from __future__ import annotations

import torch
from torch import nn


class NoRolloutTrajectoryScorer(nn.Module):
    """Direct trajectory scorer without future latent or occupancy rollout."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        traj_dim: int,
        perturb_dim: int,
        horizon_steps: int,
        num_risks: int,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.traj_dim = traj_dim
        self.scene_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
        )
        self.traj_encoder = nn.Sequential(
            nn.Linear(horizon_steps * traj_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + perturb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_risks + 2),  # risks + progress + uncertainty
        )

    def forward(self, z: torch.Tensor, trajectories: torch.Tensor, perturb: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score every candidate directly.

        Args:
            z: [B,C,H,W]
            trajectories: [B,N,T,D]
            perturb: [B,P]
        """
        b, n, t, d = trajectories.shape
        if t != self.horizon_steps or d != self.traj_dim:
            raise ValueError(f"Expected trajectories [B,N,{self.horizon_steps},{self.traj_dim}], got {tuple(trajectories.shape)}")

        scene = self.scene_pool(z)
        scene = scene[:, None].expand(b, n, scene.shape[-1])
        traj = self.traj_encoder(trajectories.reshape(b, n, t * d))
        perturb_rep = perturb[:, None].expand(b, n, perturb.shape[-1])
        raw = self.head(torch.cat([scene, traj, perturb_rep], dim=-1))
        risk = raw[:, :, None, :].expand(b, n, t, raw.shape[-1])
        return {"risk": risk}
