from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.tc_occwm.model import CANDIDATE_SOURCE_IDS, TCOccWM
from experiments.tc_occwm.occupancy_cost import (
    build_occupancy_selector_features,
    occupancy_cost_alignment_loss,
    sample_trajectory_occupancy_cost,
)
from rwc.config import bev_spec_from_config, load_config
from rwc.data.nuscenes_dataset import build_dataset
from rwc.models.encoder import BEVEncoder
from rwc.models.no_rollout_scorer import NoRolloutTrajectoryScorer
from rwc.training.trainer import move_batch_to_device
from rwc.utils.seed import seed_everything
from scripts.train_no_rollout_scorer import (
    RunningMean,
    build_proposal,
    freeze,
    listwise_loss,
    margin_loss,
    perturb_bev,
    perturb_candidates,
    perturb_token,
    scorer_score,
    target_score,
)
from scripts.evaluate_val_table import counterfactual_candidates, ego_hold_candidates, lattice_candidates


def parse_candidate_source_mix(raw: str) -> list[str]:
    sources = [item.strip() for item in raw.split(",") if item.strip()]
    if not sources:
        raise ValueError("--candidate-source-mix must contain at least one source")
    unknown = [source for source in sources if source not in CANDIDATE_SOURCE_IDS]
    if unknown:
        raise ValueError(f"unsupported candidate source(s): {unknown}; choose from {sorted(CANDIDATE_SOURCE_IDS)}")
    return sources


def _repeat_to_num_candidates(candidates: torch.Tensor, num_candidates: int) -> torch.Tensor:
    if candidates.shape[1] == num_candidates:
        return candidates
    if candidates.shape[1] > num_candidates:
        return candidates[:, :num_candidates]
    repeats = (num_candidates + candidates.shape[1] - 1) // max(candidates.shape[1], 1)
    repeat_shape = [1] * candidates.ndim
    repeat_shape[1] = repeats
    return candidates.repeat(*repeat_shape)[:, :num_candidates]


def build_training_candidates(
    sources: list[str],
    batch: dict[str, torch.Tensor],
    cfg: Any,
    num_candidates: int,
    proposal_encoder: BEVEncoder,
    proposal: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks: list[torch.Tensor] = []
    source_ids: list[torch.Tensor] = []
    per_source = max(1, num_candidates // max(len(sources), 1))
    remaining = num_candidates
    for idx, source in enumerate(sources):
        take = remaining if idx == len(sources) - 1 else per_source
        remaining -= take
        if source == "proposal":
            candidates = proposal.sample(proposal_encoder(batch["bev"]), take)
            ids = torch.full((batch["bev"].shape[0], candidates.shape[1]), CANDIDATE_SOURCE_IDS[source], device=batch["bev"].device)
        elif source == "lattice":
            candidates = lattice_candidates(batch, cfg, take)
            ids = torch.full((batch["bev"].shape[0], candidates.shape[1]), CANDIDATE_SOURCE_IDS[source], device=batch["bev"].device)
        elif source == "combined":
            proposal_n = max(1, take // 4)
            lattice_n = max(1, take // 4)
            counterfactual_n = max(1, take // 4)
            ego_hold_n = max(1, take - proposal_n - lattice_n - counterfactual_n)
            prop = proposal.sample(proposal_encoder(batch["bev"]), proposal_n)
            lat = lattice_candidates(batch, cfg, lattice_n)
            cf = counterfactual_candidates(batch, counterfactual_n)
            hold = ego_hold_candidates(batch, cfg, ego_hold_n)
            candidates = torch.cat([prop, lat, cf, hold], dim=1)
            ids = torch.full((batch["bev"].shape[0], candidates.shape[1]), CANDIDATE_SOURCE_IDS["combined"], device=batch["bev"].device)
        else:
            raise ValueError(f"unsupported candidate source: {source}")
        chunks.append(candidates)
        source_ids.append(ids)
    candidates = _repeat_to_num_candidates(torch.cat(chunks, dim=1), num_candidates)
    ids = _repeat_to_num_candidates(torch.cat(source_ids, dim=1)[..., None].float(), num_candidates).squeeze(-1).long()
    return candidates, ids


def build_teacher(cfg: Any, spec: Any, device: torch.device, ckpt_path: str):
    path = Path(ckpt_path)
    if not ckpt_path or not path.exists():
        print(f"skip teacher checkpoint: {ckpt_path}")
        return None
    encoder = BEVEncoder(spec.num_channels, int(cfg.model.latent_dim)).to(device).eval()
    scorer = NoRolloutTrajectoryScorer(
        int(cfg.model.latent_dim),
        int(cfg.model.hidden_dim),
        int(cfg.model.trajectory_dim),
        int(cfg.model.perturb_dim),
        int(cfg.planning.horizon_steps),
        len(cfg.model.risk_names),
    ).to(device).eval()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    scorer.load_state_dict(ckpt["scorer"])
    freeze(encoder)
    freeze(scorer)
    return encoder, scorer, ckpt


def init_from_teacher(encoder: BEVEncoder, model: TCOccWM, teacher: tuple[Any, Any, dict[str, Any]] | None) -> None:
    if teacher is None:
        return
    _, _, ckpt = teacher
    encoder.load_state_dict(ckpt["encoder"])
    model.direct.load_state_dict(ckpt["scorer"])
    print("initialized encoder and direct branch from no-rollout teacher")


def occupancy_prediction_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], pos_weight: float) -> torch.Tensor:
    occ_pred = outputs["future_occupancy"]
    b, n, t, c, h, w = occ_pred.shape
    occ_target = batch["future_occupancy"]
    if occ_target.shape[-2:] != (h, w):
        occ_target = F.interpolate(
            occ_target.flatten(0, 1),
            size=(h, w),
            mode="nearest",
        ).view(occ_target.shape[0], occ_target.shape[1], occ_target.shape[2], h, w)
    occ_target = occ_target[:, None].expand_as(occ_pred)
    if pos_weight <= 1.0:
        return F.binary_cross_entropy_with_logits(occ_pred, occ_target)
    weight = torch.where(occ_target > 0.5, torch.full_like(occ_target, pos_weight), torch.ones_like(occ_target))
    return F.binary_cross_entropy_with_logits(occ_pred, occ_target, weight=weight)


def risk_prediction_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], num_risks: int) -> torch.Tensor:
    risk = outputs["world_risk"]
    risk_target = batch["risk"][:, None, None, :num_risks].expand(*risk.shape[:3], num_risks)
    progress_target = batch["risk"][:, None, None, num_risks : num_risks + 1].expand(*risk.shape[:3], 1)
    risk_loss = F.binary_cross_entropy_with_logits(risk[..., :num_risks], risk_target)
    progress_loss = F.smooth_l1_loss(risk[..., num_risks : num_risks + 1], progress_target)
    return risk_loss + 0.25 * progress_loss


def soft_target_loss(pred_score: torch.Tensor, teacher_score: torch.Tensor, tau: float) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(pred_score / tau, dim=1),
        F.softmax(teacher_score / tau, dim=1),
        reduction="batchmean",
    ) * (tau * tau)


def mask_selector_features(selector_features: torch.Tensor, disable_occupancy_features: bool) -> torch.Tensor:
    if not disable_occupancy_features:
        return selector_features
    out = selector_features.clone()
    # Columns 0:5 are occupancy-derived costs: collision, mean collision,
    # offroad, route, and uncertainty. Keep progress/comfort visible.
    out[..., :5] = 0.0
    return out


def consistency_loss(
    model: TCOccWM,
    encoder: BEVEncoder,
    batch: dict[str, torch.Tensor],
    candidates: torch.Tensor,
    token: torch.Tensor,
    noisy_occ: torch.Tensor,
) -> torch.Tensor:
    was_encoder_training = encoder.training
    was_model_training = model.training
    encoder.eval()
    model.eval()
    with torch.no_grad():
        clean_token = torch.zeros_like(token)
        clean_outputs = model(encoder(batch["bev"]), candidates, clean_token)
        clean_occ = torch.sigmoid(clean_outputs["future_occupancy"])
    encoder.train(was_encoder_training)
    model.train(was_model_training)
    return F.mse_loss(torch.sigmoid(noisy_occ), clean_occ)


def build_model(args: argparse.Namespace, cfg: Any, spec: Any, device: torch.device) -> tuple[BEVEncoder, TCOccWM]:
    encoder = BEVEncoder(spec.num_channels, int(cfg.model.latent_dim)).to(device)
    model = TCOccWM(
        latent_dim=int(cfg.model.latent_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        traj_dim=int(cfg.model.trajectory_dim),
        perturb_dim=int(cfg.model.perturb_dim),
        horizon_steps=int(cfg.planning.horizon_steps),
        occupancy_channels=int(cfg.model.occupancy_channels),
        num_risks=len(cfg.model.risk_names),
        num_candidate_sources=len(CANDIDATE_SOURCE_IDS) if args.use_source_embedding else 0,
        source_embed_dim=args.source_embed_dim,
    ).to(device)
    return encoder, model


def resume_training_state(
    args: argparse.Namespace,
    encoder: BEVEncoder,
    model: TCOccWM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    if not args.resume:
        return 0, float("inf")

    resume_path = Path(args.resume)
    if not resume_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    model.load_state_dict(ckpt["tc_occwm"])

    ckpt_args = ckpt.get("args", {})
    if "disable_occupancy_features" in ckpt_args:
        ckpt_disabled = bool(ckpt_args["disable_occupancy_features"])
        current_disabled = bool(args.disable_occupancy_features)
        if ckpt_disabled != current_disabled:
            raise ValueError(
                "resume checkpoint and current command disagree on "
                f"--disable-occupancy-features: checkpoint={ckpt_disabled}, current={current_disabled}. "
                "Keep the same ablation flag when resuming."
            )
    if "disable_uncertainty_features" in ckpt_args:
        ckpt_disabled = bool(ckpt_args["disable_uncertainty_features"])
        current_disabled = bool(args.disable_uncertainty_features)
        if ckpt_disabled != current_disabled:
            raise ValueError(
                "resume checkpoint and current command disagree on "
                f"--disable-uncertainty-features: checkpoint={ckpt_disabled}, current={current_disabled}. "
                "Keep the same ablation flag when resuming."
            )
    for key in ("candidate_source_mix", "use_source_embedding", "source_embed_dim", "normalize_occupancy_features"):
        if key not in ckpt_args:
            continue
        old = ckpt_args[key]
        new = getattr(args, key)
        if old != new:
            raise ValueError(f"resume checkpoint and current command disagree on --{key.replace('_', '-')}: checkpoint={old}, current={new}")

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer_loaded = True
    else:
        optimizer_loaded = False

    if "epoch" in ckpt:
        start_epoch = int(ckpt["epoch"])
        epoch_source = "checkpoint"
    elif args.resume_completed_epochs is not None:
        start_epoch = int(args.resume_completed_epochs)
        epoch_source = "--resume-completed-epochs"
    else:
        start_epoch = 0
        epoch_source = "legacy checkpoint without epoch"

    best_loss = float(ckpt.get("best_loss", float("inf")))
    print(
        {
            "resume": str(resume_path),
            "start_epoch": start_epoch,
            "epoch_source": epoch_source,
            "optimizer_loaded": optimizer_loaded,
            "best_loss": best_loss,
        }
    )
    return start_epoch, best_loss


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
    teacher = build_teacher(cfg, spec, device, args.teacher_ckpt)
    candidate_sources = parse_candidate_source_mix(args.candidate_source_mix)
    if len(candidate_sources) > 1 and not args.use_source_embedding:
        print("candidate source mix has multiple sources; enabling source embedding")
        args.use_source_embedding = True

    encoder, model = build_model(args, cfg, spec, device)
    if args.init_from_teacher:
        init_from_teacher(encoder, model, teacher)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(model.parameters()),
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
    start_epoch, best_loss = resume_training_state(args, encoder, model, optimizer, device)

    settings = [
        ("clean", 0.0, 0.0, 0.0, 0.0, 0, args.clean_weight),
        ("mild", args.mild_bev_drop, args.mild_agent_drop, args.mild_xy_noise, args.mild_speed_noise, args.mild_delay, args.mild_weight),
        ("severe", args.severe_bev_drop, args.severe_agent_drop, args.severe_xy_noise, args.severe_speed_noise, args.severe_delay, args.severe_weight),
    ]

    if start_epoch >= args.epochs:
        print(
            {
                "message": "checkpoint already reached the requested target epochs",
                "completed_epochs": start_epoch,
                "target_epochs": args.epochs,
            }
        )
        return

    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        model.train()
        totals = RunningMean()
        for step, batch in enumerate(tqdm(loader, desc=f"TC-OccWM {epoch + 1}/{args.epochs}")):
            if args.max_batches is not None and step >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            b = batch["bev"].shape[0]
            with torch.no_grad():
                candidates, source_ids = build_training_candidates(
                    candidate_sources,
                    batch,
                    cfg,
                    num_candidates,
                    proposal_encoder,
                    proposal,
                )

            loss = torch.zeros((), device=device)
            log_values: dict[str, float] = {}
            for name, bev_drop, agent_drop, xy_noise, speed_noise, delay, setting_weight in settings:
                if setting_weight <= 0:
                    continue
                perturbed_candidates = perturb_candidates(candidates, xy_noise, speed_noise, delay)
                perturbed_bev = perturb_bev(batch["bev"], cfg, bev_drop, agent_drop)
                token = perturb_token(b, device, delay, bev_drop, agent_drop, xy_noise, speed_noise)
                target, components = target_score(perturbed_candidates, batch, cfg, weights)

                z = encoder(perturbed_bev)
                outputs = model(z, perturbed_candidates, token)
                costs = sample_trajectory_occupancy_cost(
                    outputs["future_occupancy"],
                    perturbed_candidates,
                    cfg,
                    uncertainty_logits=outputs["world_uncertainty"],
                )
                selector_features = build_occupancy_selector_features(
                    costs,
                    components["progress"],
                    components["comfort"],
                    normalize=args.normalize_occupancy_features,
                    disable_uncertainty=args.disable_uncertainty_features,
                )
                selector_features = mask_selector_features(selector_features, args.disable_occupancy_features)
                pred = model.score(outputs, selector_features, source_ids if args.use_source_embedding else None)

                rank = listwise_loss(pred, target, args.tau)
                margin = margin_loss(pred, target, args.margin)
                occ = occupancy_prediction_loss(outputs, batch, args.occ_pos_weight)
                risk = risk_prediction_loss(outputs, batch, model.num_risks)
                align = occupancy_cost_alignment_loss(costs, components["collision"])
                setting_loss = (
                    rank
                    + args.margin_weight * margin
                    + args.occ_weight * occ
                    + args.risk_weight * risk
                    + args.cost_align_weight * align
                )

                if args.distill_weight > 0 and teacher is not None:
                    teacher_encoder, teacher_scorer, _ = teacher
                    with torch.no_grad():
                        teacher_pred = scorer_score(teacher_scorer, teacher_encoder(perturbed_bev), perturbed_candidates, token)
                    distill = soft_target_loss(pred, teacher_pred, args.distill_tau)
                    setting_loss = setting_loss + args.distill_weight * distill
                    log_values[f"{name}_distill"] = float(distill.detach().cpu())

                if args.consistency_weight > 0 and name != "clean":
                    cons = consistency_loss(model, encoder, batch, perturbed_candidates, token, outputs["future_occupancy"])
                    setting_loss = setting_loss + args.consistency_weight * cons
                    log_values[f"{name}_consistency"] = float(cons.detach().cpu())

                loss = loss + setting_weight * setting_loss
                with torch.no_grad():
                    ar = torch.arange(b, device=device)
                    best = pred.argmax(dim=1)
                    log_values[f"{name}_rank"] = float(rank.detach().cpu())
                    log_values[f"{name}_occ"] = float(occ.detach().cpu())
                    log_values[f"{name}_align"] = float(align.detach().cpu())
                    log_values[f"{name}_ADE"] = float(components["ade"][ar, best].mean().detach().cpu())
                    log_values[f"{name}_collision"] = float(components["collision"][ar, best].mean().detach().cpu())
                    log_values[f"{name}_occ_cost"] = float(costs["collision_cost"][ar, best].mean().detach().cpu())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(model.parameters()), args.grad_clip_norm)
            optimizer.step()
            log_values["loss"] = float(loss.detach().cpu())
            totals.update(log_values, b)

        metrics = totals.compute()
        print({"epoch": epoch + 1, **metrics})
        current_best_loss = best_loss
        if metrics.get("loss", float("inf")) < current_best_loss:
            current_best_loss = metrics["loss"]
        ckpt = {
            "encoder": encoder.state_dict(),
            "tc_occwm": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_loss": current_best_loss,
            "metrics": metrics,
            "proposal_checkpoint": args.proposal_ckpt,
            "teacher_checkpoint": args.teacher_ckpt,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "latest.pt")
        torch.save(ckpt, out_dir / "last.pt")
        if current_best_loss < best_loss:
            best_loss = current_best_loss
            torch.save(ckpt, out_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TC-OccWM for robust trajectory reranking.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--proposal-ckpt", default="outputs/proposal/latest.pt")
    parser.add_argument("--teacher-ckpt", default="outputs/no_rollout_scorer/best.pt")
    parser.add_argument("--output-dir", default="outputs/tc_occwm")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume TC-OccWM training from last.pt/latest.pt/best.pt.",
    )
    parser.add_argument(
        "--resume-completed-epochs",
        type=int,
        default=None,
        help="Completed epochs for legacy checkpoints that do not store an epoch field.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--candidate-source-mix",
        default="proposal",
        help="Comma-separated training candidate sources from proposal,lattice,combined.",
    )
    parser.add_argument(
        "--use-source-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Condition the selector on candidate-source IDs. Auto-enabled when multiple training sources are used.",
    )
    parser.add_argument("--source-embed-dim", type=int, default=4)
    parser.add_argument(
        "--normalize-occupancy-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize occupancy-derived selector features within each scene's candidate set.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--occ-weight", type=float, default=0.08)
    parser.add_argument("--risk-weight", type=float, default=0.03)
    parser.add_argument("--cost-align-weight", type=float, default=0.10)
    parser.add_argument("--consistency-weight", type=float, default=0.01)
    parser.add_argument("--distill-weight", type=float, default=0.10)
    parser.add_argument("--distill-tau", type=float, default=0.7)
    parser.add_argument("--occ-pos-weight", type=float, default=3.0)
    parser.add_argument(
        "--disable-occupancy-features",
        action="store_true",
        help="Zero occupancy-derived selector features; use for the without-occupancy-cost ablation.",
    )
    parser.add_argument(
        "--disable-uncertainty-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zero only the occupancy-derived uncertainty selector feature.",
    )
    parser.add_argument("--init-from-teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-weight", type=float, default=0.8)
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
