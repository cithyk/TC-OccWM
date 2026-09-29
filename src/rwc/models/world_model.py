from __future__ import annotations

import torch
from torch import nn

from rwc.models.encoder import BEVDecoder


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, 3, padding=1)
        self.candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=1)
        reset, update = self.gates(combined).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        cand = torch.tanh(self.candidate(torch.cat([x, reset * h], dim=1)))
        return (1.0 - update) * h + update * cand


class TrajectoryConditioner(nn.Module):
    def __init__(self, traj_dim: int, perturb_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(traj_dim + perturb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
        )

    def forward(self, traj_step: torch.Tensor, perturb: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = torch.cat([traj_step, perturb], dim=-1)
        token = self.net(x)
        return token[..., :, None, None].expand(*token.shape, h, w)


class LatentWorldModel(nn.Module):
    """Action-conditioned latent rollout model with occupancy and risk heads."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        traj_dim: int,
        perturb_dim: int,
        horizon_steps: int,
        occupancy_channels: int,
        num_risks: int,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.input_proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.conditioner = TrajectoryConditioner(traj_dim, perturb_dim, hidden_dim)
        self.gru = ConvGRUCell(hidden_dim, hidden_dim)
        self.to_latent = nn.Conv2d(hidden_dim, latent_dim, 1)
        self.occ_head = BEVDecoder(hidden_dim, occupancy_channels)
        self.risk_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim + traj_dim + perturb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_risks + 2),  # risks + progress + uncertainty
        )

    def forward(self, z: torch.Tensor, trajectories: torch.Tensor, perturb: torch.Tensor) -> dict[str, torch.Tensor]:
        """Roll out every candidate.

        Args:
            z: [B,C,H,W]
            trajectories: [B,N,T,D]
            perturb: [B,8]
        """
        b, n, t, d = trajectories.shape
        _, _, h, w = z.shape
        z_rep = z[:, None].expand(b, n, *z.shape[1:]).reshape(b * n, *z.shape[1:])
        traj = trajectories.reshape(b * n, t, d)
        perturb_rep = perturb[:, None].expand(b, n, perturb.shape[-1]).reshape(b * n, perturb.shape[-1])

        hidden = self.input_proj(z_rep)
        occ_steps = []
        latent_steps = []
        risk_steps = []
        for step in range(self.horizon_steps):
            cond = self.conditioner(traj[:, step], perturb_rep, h, w)
            hidden = self.gru(cond, hidden)
            latent_steps.append(self.to_latent(hidden))
            occ_steps.append(self.occ_head(hidden))
            pooled = self.risk_pool(hidden)
            risk_steps.append(self.risk_head(torch.cat([pooled, traj[:, step], perturb_rep], dim=-1)))

        occ = torch.stack(occ_steps, dim=1)
        latent = torch.stack(latent_steps, dim=1)
        risk = torch.stack(risk_steps, dim=1)
        occ = occ.view(b, n, t, *occ.shape[2:])
        latent = latent.view(b, n, t, *latent.shape[2:])
        risk = risk.view(b, n, t, risk.shape[-1])
        return {"future_latent": latent, "future_occupancy": occ, "risk": risk}
