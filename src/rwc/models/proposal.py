from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TrajectoryProposalCVAE(nn.Module):
    """Lightweight CVAE for multi-modal ego trajectory proposals."""

    def __init__(
        self,
        bev_channels: int,
        horizon_steps: int,
        trajectory_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        num_candidates: int = 32,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.trajectory_dim = trajectory_dim
        self.latent_dim = latent_dim
        self.num_candidates = num_candidates
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bev_channels, hidden_dim),
            nn.SiLU(),
        )
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + horizon_steps * trajectory_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon_steps * trajectory_dim),
        )

    def encode(self, z: torch.Tensor, future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self.context(z)
        posterior = self.posterior(torch.cat([ctx, future.flatten(1)], dim=-1))
        mu = self.mu(posterior)
        logvar = self.logvar(posterior).clamp(-8.0, 4.0)
        eps = torch.randn_like(mu)
        sample = mu + eps * torch.exp(0.5 * logvar)
        return sample, mu, logvar

    def decode(self, z: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        ctx = self.context(z)
        out = self.decoder(torch.cat([ctx, latent], dim=-1))
        traj = out.view(z.shape[0], self.horizon_steps, self.trajectory_dim)
        traj_xy = torch.cumsum(traj[..., :2], dim=1)
        return torch.cat([traj_xy, traj[..., 2:]], dim=-1)

    def forward(self, z: torch.Tensor, future: torch.Tensor) -> dict[str, torch.Tensor]:
        latent, mu, logvar = self.encode(z, future)
        recon = self.decode(z, latent)
        return {"traj": recon, "mu": mu, "logvar": logvar}

    @torch.no_grad()
    def sample(self, z: torch.Tensor, num_candidates: int | None = None) -> torch.Tensor:
        n = num_candidates or self.num_candidates
        b = z.shape[0]
        z_rep = z[:, None].expand(b, n, *z.shape[1:]).reshape(b * n, *z.shape[1:])
        latent = torch.randn(b * n, self.latent_dim, device=z.device, dtype=z.dtype)
        traj = self.decode(z_rep, latent)
        return traj.view(b, n, self.horizon_steps, self.trajectory_dim)


def cvae_loss(pred: dict[str, torch.Tensor], target: torch.Tensor, beta: float = 0.01) -> dict[str, torch.Tensor]:
    recon = F.smooth_l1_loss(pred["traj"], target)
    mu = pred["mu"]
    logvar = pred["logvar"]
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return {"loss": recon + beta * kl, "recon": recon.detach(), "kl": kl.detach()}
