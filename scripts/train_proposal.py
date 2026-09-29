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
from rwc.models.encoder import BEVEncoder
from rwc.models.proposal import TrajectoryProposalCVAE, cvae_loss
from rwc.training.metrics import RunningMean, proposal_metrics
from rwc.training.trainer import autocast_context, grad_accum_steps, move_batch_to_device
from rwc.utils.io import ensure_dir, save_checkpoint
from rwc.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
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
    proposal = TrajectoryProposalCVAE(
        bev_channels=int(cfg.model.latent_dim),
        horizon_steps=int(cfg.planning.horizon_steps),
        trajectory_dim=int(cfg.model.trajectory_dim),
        latent_dim=int(cfg.model.proposal_latent_dim),
        num_candidates=int(cfg.planning.num_candidates),
    ).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(proposal.parameters()), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    accum = grad_accum_steps(int(cfg.training.batch_size), int(cfg.training.effective_batch_size))
    epochs = int(args.epochs or cfg.training.epochs)

    out_dir = ensure_dir(Path(cfg.training.output_dir) / "proposal")
    for epoch in range(epochs):
        encoder.train()
        proposal.train()
        meter = RunningMean()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"proposal epoch {epoch+1}/{epochs}")):
            batch = move_batch_to_device(batch, device)
            with autocast_context(bool(cfg.training.amp), device):
                z = encoder(batch["bev"])
                pred = proposal(z, batch["future_ego"])
                losses = cvae_loss(pred, batch["future_ego"])
                loss = losses["loss"] / accum
            loss.backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(proposal.parameters()), float(cfg.training.grad_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            meter.update(losses, n=batch["bev"].shape[0])
        encoder.eval()
        proposal.eval()
        with torch.no_grad():
            batch = move_batch_to_device(next(iter(loader)), device)
            z = encoder(batch["bev"])
            candidates = proposal.sample(z, int(cfg.planning.num_candidates))
            metrics = proposal_metrics(candidates, batch["future_ego"])
        print({"epoch": epoch + 1, **meter.compute(), **metrics})
        save_checkpoint(out_dir / "latest.pt", encoder=encoder.state_dict(), proposal=proposal.state_dict(), cfg=dict(cfg))


if __name__ == "__main__":
    main()
