from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rwc.config import bev_spec_from_config, load_config
from rwc.data.nuscenes_dataset import build_dataset
from rwc.data.perturbations import PerturbationSpec, clean_perturb_tokens, sample_perturb_tokens
from rwc.data.trajectory import make_counterfactual_trajectories
from rwc.models.encoder import BEVEncoder
from rwc.models.world_model import LatentWorldModel
from rwc.training.losses import world_model_loss
from rwc.training.metrics import RunningMean
from rwc.training.trainer import autocast_context, grad_accum_steps, move_batch_to_device
from rwc.utils.io import ensure_dir, save_checkpoint
from rwc.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None, help="Override candidates for low-memory smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    spec = bev_spec_from_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(cfg, spec, args.split, args.synthetic)
    loader = DataLoader(dataset, batch_size=int(cfg.training.batch_size), shuffle=True, num_workers=0 if args.synthetic else int(cfg.data.num_workers))

    encoder = BEVEncoder(spec.num_channels, int(cfg.model.latent_dim)).to(device)
    world = LatentWorldModel(
        latent_dim=int(cfg.model.latent_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        traj_dim=int(cfg.model.trajectory_dim),
        perturb_dim=int(cfg.model.perturb_dim),
        horizon_steps=int(cfg.planning.horizon_steps),
        occupancy_channels=int(cfg.model.occupancy_channels),
        num_risks=len(cfg.model.risk_names),
    ).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(world.parameters()), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    accum = grad_accum_steps(int(cfg.training.batch_size), int(cfg.training.effective_batch_size))
    perturb_spec = PerturbationSpec(
        max_delay_steps=int(cfg.perturbations.max_delay_steps),
        bev_dropout_prob=float(cfg.perturbations.bev_dropout_prob),
        agent_dropout_prob=float(cfg.perturbations.agent_dropout_prob),
        ego_noise_std=(
            float(cfg.perturbations.ego_noise_std.x),
            float(cfg.perturbations.ego_noise_std.y),
            float(cfg.perturbations.ego_noise_std.yaw),
            float(cfg.perturbations.ego_noise_std.speed),
        ),
    )
    epochs = int(args.epochs or cfg.training.epochs)
    out_dir = ensure_dir(Path(cfg.training.output_dir) / "world_model")

    for epoch in range(epochs):
        encoder.train()
        world.train()
        meter = RunningMean()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"world epoch {epoch+1}/{epochs}")):
            batch = move_batch_to_device(batch, device)
            b = batch["bev"].shape[0]
            with autocast_context(bool(cfg.training.amp), device):
                z = encoder(batch["bev"])
                num_candidates = int(args.num_candidates or cfg.planning.num_candidates)
                candidates = torch.stack(
                    [make_counterfactual_trajectories(batch["future_ego"][i], num_candidates) for i in range(b)],
                    dim=0,
                )
                is_expert = torch.zeros(b, num_candidates, dtype=torch.bool, device=device)
                is_expert[:, 0] = True
                noisy = torch.rand((), device=device) > float(cfg.training.clean_ratio)
                perturb = sample_perturb_tokens(b, perturb_spec, device) if noisy else clean_perturb_tokens(b, device)
                rollout = world(z, candidates, perturb)
                losses = world_model_loss(rollout, batch["future_occupancy"], batch["risk"], is_expert)
                loss = losses["loss"] / accum
            loss.backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(world.parameters()), float(cfg.training.grad_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            meter.update(losses, n=b)
        print({"epoch": epoch + 1, **meter.compute()})
        save_checkpoint(out_dir / "latest.pt", encoder=encoder.state_dict(), world_model=world.state_dict(), cfg=dict(cfg))


if __name__ == "__main__":
    main()
