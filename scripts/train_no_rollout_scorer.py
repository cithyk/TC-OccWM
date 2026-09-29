from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rwc.config import bev_spec_from_config, load_config
from rwc.data.nuscenes_dataset import build_dataset
from rwc.data.perturbations import clean_perturb_tokens
from rwc.data.trajectory import comfort_cost, trajectory_box_collision
from rwc.models.encoder import BEVEncoder
from rwc.models.no_rollout_scorer import NoRolloutTrajectoryScorer
from rwc.models.proposal import TrajectoryProposalCVAE
from rwc.training.trainer import move_batch_to_device
from rwc.utils.seed import seed_everything


class RunningMean:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.count = 0

    def update(self, values: dict[str, float], n: int) -> None:
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * n
        self.count += n

    def compute(self) -> dict[str, float]:
        return {key: value / max(self.count, 1) for key, value in self.sums.items()}


def build_proposal(cfg: Any, spec: Any, device: torch.device, ckpt_path: str, num_candidates: int):
    encoder = BEVEncoder(spec.num_channels, int(cfg.model.latent_dim)).to(device).eval()
    proposal = TrajectoryProposalCVAE(
        int(cfg.model.latent_dim),
        int(cfg.planning.horizon_steps),
        int(cfg.model.trajectory_dim),
        int(cfg.model.proposal_latent_dim),
        num_candidates=num_candidates,
    ).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    proposal.load_state_dict(ckpt["proposal"])
    return encoder, proposal


def freeze(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def perturb_bev(bev: torch.Tensor, cfg: Any, bev_drop: float, agent_drop: float) -> torch.Tensor:
    out = bev.clone()
    if bev_drop > 0:
        keep = torch.rand(out.shape[0], 1, out.shape[2], out.shape[3], device=out.device) > bev_drop
        out = out * keep.float()
    if agent_drop > 0 and "agents" in list(cfg.bev.channels):
        agent_idx = list(cfg.bev.channels).index("agents")
        keep = torch.rand(out.shape[0], 1, out.shape[2], out.shape[3], device=out.device) > agent_drop
        out[:, agent_idx : agent_idx + 1] *= keep.float()
    return out


def perturb_candidates(candidates: torch.Tensor, xy_std: float, speed_std: float, delay: int) -> torch.Tensor:
    out = candidates.clone()
    if xy_std > 0:
        out[..., :2] += torch.randn(out.shape[0], out.shape[1], 1, 2, device=out.device) * xy_std
    if speed_std > 0:
        out[..., 3:4] += torch.randn(out.shape[0], out.shape[1], 1, 1, device=out.device) * speed_std
    if delay > 0:
        tail = out[:, :, -1:].expand(-1, -1, delay, -1)
        out = torch.cat([out[:, :, delay:], tail], dim=2)
    return out


def perturb_token(
    batch_size: int,
    device: torch.device,
    delay: int,
    bev_drop: float,
    agent_drop: float,
    xy_std: float,
    speed_std: float,
) -> torch.Tensor:
    token = clean_perturb_tokens(batch_size, device)
    token[:, 0] = float(delay)
    token[:, 1] = float(bev_drop)
    token[:, 2] = float(agent_drop)
    token[:, 3] = float(xy_std)
    token[:, 4] = float(xy_std)
    token[:, 6] = float(speed_std)
    token[:, 7] = 1.0 if any(v > 0 for v in [delay, bev_drop, agent_drop, xy_std, speed_std]) else 0.0
    return token


def ade_fde(candidates: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    err = torch.linalg.norm(candidates[..., :2] - target[:, None, :, :2], dim=-1)
    return err.mean(dim=-1), err[..., -1]


def gt_offroad(candidates: torch.Tensor, bev: torch.Tensor, cfg: Any) -> torch.Tensor:
    b, n, t, _ = candidates.shape
    outs = []
    for bi in range(b):
        rr = ((candidates[bi, ..., 0] - float(cfg.bev.x_min)) / float(cfg.bev.resolution)).long()
        cc = ((candidates[bi, ..., 1] - float(cfg.bev.y_min)) / float(cfg.bev.resolution)).long()
        valid = (rr >= 0) & (rr < bev.shape[-2]) & (cc >= 0) & (cc < bev.shape[-1])
        sampled = torch.zeros(n, t, device=bev.device)
        sampled[valid] = bev[bi, 0, rr[valid], cc[valid]]
        outs.append(((~valid) | (sampled < 0.5)).any(dim=1).float())
    return torch.stack(outs, dim=0)


def target_score(candidates: torch.Tensor, batch: dict[str, Any], cfg: Any, weights: dict[str, float]):
    ade, fde = ade_fde(candidates, batch["future_ego"])
    collision = trajectory_box_collision(candidates, batch["future_agents"])
    offroad = gt_offroad(candidates, batch["bev"], cfg)
    progress = torch.clamp(candidates[..., -1, 0] / max(float(cfg.bev.x_max), 1.0), 0.0, 1.0)
    comfort = comfort_cost(candidates, float(cfg.planning.dt))
    score = (
        -weights["collision"] * collision
        -weights["offroad"] * offroad
        -weights["ade"] * ade
        -weights["fde"] * fde
        + weights["progress"] * progress
        -weights["comfort"] * comfort
    )
    return score, {
        "ade": ade,
        "fde": fde,
        "collision": collision,
        "offroad": offroad,
        "progress": progress,
        "comfort": comfort,
    }


def scorer_score(model: NoRolloutTrajectoryScorer, z: torch.Tensor, candidates: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    raw = model(z, candidates, token)["risk"]
    risk = torch.sigmoid(raw[..., :3]).amax(dim=2)
    progress = raw[..., 3].mean(dim=2)
    uncertainty = F.softplus(raw[..., 4]).mean(dim=2)
    return progress - risk.sum(dim=-1) - 0.1 * uncertainty


def listwise_loss(pred_score: torch.Tensor, target_score: torch.Tensor, tau: float) -> torch.Tensor:
    target_prob = F.softmax(target_score / tau, dim=1)
    return F.kl_div(F.log_softmax(pred_score / tau, dim=1), target_prob, reduction="batchmean")


def margin_loss(pred_score: torch.Tensor, target_score: torch.Tensor, margin: float) -> torch.Tensor:
    best = target_score.argmax(dim=1)
    worst = target_score.argmin(dim=1)
    ar = torch.arange(pred_score.shape[0], device=pred_score.device)
    return F.relu(margin - pred_score[ar, best] + pred_score[ar, worst]).mean()


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    seed_everything(int(args.seed if args.seed is not None else cfg.seed))
    spec = bev_spec_from_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_candidates = int(args.num_candidates or cfg.planning.num_candidates)
    dataset = build_dataset(cfg, spec, args.split, args.synthetic)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or cfg.training.batch_size),
        shuffle=True,
        num_workers=0 if args.synthetic else int(cfg.data.num_workers),
        drop_last=True,
    )

    proposal_encoder, proposal = build_proposal(cfg, spec, device, args.proposal_ckpt, num_candidates)
    freeze(proposal_encoder)
    freeze(proposal)

    encoder = BEVEncoder(spec.num_channels, int(cfg.model.latent_dim)).to(device)
    scorer = NoRolloutTrajectoryScorer(
        latent_dim=int(cfg.model.latent_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        traj_dim=int(cfg.model.trajectory_dim),
        perturb_dim=int(cfg.model.perturb_dim),
        horizon_steps=int(cfg.planning.horizon_steps),
        num_risks=len(cfg.model.risk_names),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(scorer.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    weights = {
        "collision": args.w_collision,
        "offroad": args.w_offroad,
        "ade": args.w_ade,
        "fde": args.w_fde,
        "progress": args.w_progress,
        "comfort": args.w_comfort,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    settings = [
        ("clean", 0.0, 0.0, 0.0, 0.0, 0, args.clean_weight),
        ("mild", args.mild_bev_drop, args.mild_agent_drop, args.mild_xy_noise, args.mild_speed_noise, args.mild_delay, args.mild_weight),
        ("severe", args.severe_bev_drop, args.severe_agent_drop, args.severe_xy_noise, args.severe_speed_noise, args.severe_delay, args.severe_weight),
    ]

    for epoch in range(args.epochs):
        encoder.train()
        scorer.train()
        totals = RunningMean()
        for step, batch in enumerate(tqdm(loader, desc=f"no-rollout scorer {epoch + 1}/{args.epochs}")):
            if args.max_batches is not None and step >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            b = batch["bev"].shape[0]
            with torch.no_grad():
                candidates = proposal.sample(proposal_encoder(batch["bev"]), num_candidates)

            loss = torch.zeros((), device=device)
            log_values: dict[str, float] = {}
            for name, bev_drop, agent_drop, xy_noise, speed_noise, delay, weight in settings:
                if weight <= 0:
                    continue
                perturbed_candidates = perturb_candidates(candidates, xy_noise, speed_noise, delay)
                perturbed_bev = perturb_bev(batch["bev"], cfg, bev_drop, agent_drop)
                token = perturb_token(b, device, delay, bev_drop, agent_drop, xy_noise, speed_noise)
                target, components = target_score(perturbed_candidates, batch, cfg, weights)
                pred = scorer_score(scorer, encoder(perturbed_bev), perturbed_candidates, token)
                rank = listwise_loss(pred, target, args.tau)
                margin = margin_loss(pred, target, args.margin)
                loss = loss + weight * (rank + args.margin_weight * margin)

                with torch.no_grad():
                    ar = torch.arange(b, device=device)
                    best = pred.argmax(dim=1)
                    log_values[f"{name}_rank"] = float(rank.detach().cpu())
                    log_values[f"{name}_ADE"] = float(components["ade"][ar, best].mean().detach().cpu())
                    log_values[f"{name}_collision"] = float(components["collision"][ar, best].mean().detach().cpu())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(scorer.parameters()), args.grad_clip_norm)
            optimizer.step()
            log_values["loss"] = float(loss.detach().cpu())
            totals.update(log_values, b)

        metrics = totals.compute()
        print({"epoch": epoch + 1, **metrics})
        ckpt = {
            "encoder": encoder.state_dict(),
            "scorer": scorer.state_dict(),
            "proposal_checkpoint": args.proposal_ckpt,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "latest.pt")
        if metrics.get("loss", float("inf")) < best_loss:
            best_loss = metrics["loss"]
            torch.save(ckpt, out_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a direct no-rollout trajectory scorer baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--proposal-ckpt", default="outputs/proposal/latest.pt")
    parser.add_argument("--output-dir", default="outputs/no_rollout_scorer")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--clean-weight", type=float, default=0.7)
    parser.add_argument("--mild-weight", type=float, default=0.8)
    parser.add_argument("--severe-weight", type=float, default=1.0)
    parser.add_argument("--mild-bev-drop", type=float, default=0.05)
    parser.add_argument("--mild-agent-drop", type=float, default=0.15)
    parser.add_argument("--mild-xy-noise", type=float, default=0.2)
    parser.add_argument("--mild-speed-noise", type=float, default=0.2)
    parser.add_argument("--mild-delay", type=int, default=1)
    parser.add_argument("--severe-bev-drop", type=float, default=0.10)
    parser.add_argument("--severe-agent-drop", type=float, default=0.30)
    parser.add_argument("--severe-xy-noise", type=float, default=0.5)
    parser.add_argument("--severe-speed-noise", type=float, default=0.5)
    parser.add_argument("--severe-delay", type=int, default=2)
    parser.add_argument("--w-collision", type=float, default=5.0)
    parser.add_argument("--w-offroad", type=float, default=3.0)
    parser.add_argument("--w-ade", type=float, default=0.3)
    parser.add_argument("--w-fde", type=float, default=0.2)
    parser.add_argument("--w-progress", type=float, default=1.0)
    parser.add_argument("--w-comfort", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
