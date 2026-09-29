from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rwc.models.encoder import BEVDecoder
from rwc.models.no_rollout_scorer import NoRolloutTrajectoryScorer
from rwc.models.world_model import ConvGRUCell, TrajectoryConditioner


class HybridWorldScorer(nn.Module):
    """World-regularized trajectory scorer.

    The direct branch keeps the strong no-rollout scorer behavior. The world
    branch rolls a candidate-conditioned latent state forward and feeds its
    summary into a learned selector residual. The final score is intentionally
    initialized as the direct scorer, so adding the world branch should not
    destroy a good teacher checkpoint at step zero.
    """

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
        self.traj_dim = traj_dim
        self.num_risks = num_risks
        self.out_dim = num_risks + 2

        self.direct = NoRolloutTrajectoryScorer(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            traj_dim=traj_dim,
            perturb_dim=perturb_dim,
            horizon_steps=horizon_steps,
            num_risks=num_risks,
        )

        self.input_proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.conditioner = TrajectoryConditioner(traj_dim, perturb_dim, hidden_dim)
        self.gru = ConvGRUCell(hidden_dim, hidden_dim)
        self.occ_head = BEVDecoder(hidden_dim, occupancy_channels)
        self.risk_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.world_head = nn.Sequential(
            nn.Linear(hidden_dim + traj_dim + perturb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )

        direct_context_dim = hidden_dim * 2 + perturb_dim
        self.direct_context_proj = nn.Sequential(
            nn.Linear(direct_context_dim, hidden_dim),
            nn.SiLU(),
        )
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim * 2 + self.out_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )
        self._zero_init_selector_residual()

    def _zero_init_selector_residual(self) -> None:
        last = self.selector[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _direct_raw_and_context(
        self,
        z: torch.Tensor,
        trajectories: torch.Tensor,
        perturb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, t, d = trajectories.shape
        if t != self.horizon_steps or d != self.traj_dim:
            raise ValueError(f"Expected trajectories [B,N,{self.horizon_steps},{self.traj_dim}], got {tuple(trajectories.shape)}")

        scene = self.direct.scene_pool(z)[:, None].expand(b, n, -1)
        traj = self.direct.traj_encoder(trajectories.reshape(b, n, t * d))
        perturb_rep = perturb[:, None].expand(b, n, perturb.shape[-1])
        direct_context = torch.cat([scene, traj, perturb_rep], dim=-1)
        direct_raw = self.direct.head(direct_context)
        return direct_raw, direct_context

    def forward(
        self,
        z: torch.Tensor,
        trajectories: torch.Tensor,
        perturb: torch.Tensor,
        return_world_outputs: bool = False,
    ) -> dict[str, torch.Tensor]:
        b, n, t, d = trajectories.shape
        _, _, h, w = z.shape

        direct_raw, direct_context = self._direct_raw_and_context(z, trajectories, perturb)

        z_rep = z[:, None].expand(b, n, *z.shape[1:]).reshape(b * n, *z.shape[1:])
        traj = trajectories.reshape(b * n, t, d)
        perturb_rep = perturb[:, None].expand(b, n, perturb.shape[-1]).reshape(b * n, perturb.shape[-1])

        hidden = self.input_proj(z_rep)
        pooled_steps = []
        risk_steps = []
        occ_steps = []
        for step in range(self.horizon_steps):
            cond = self.conditioner(traj[:, step], perturb_rep, h, w)
            hidden = self.gru(cond, hidden)
            pooled = self.risk_pool(hidden)
            pooled_steps.append(pooled)
            risk_steps.append(self.world_head(torch.cat([pooled, traj[:, step], perturb_rep], dim=-1)))
            if return_world_outputs:
                occ_steps.append(self.occ_head(hidden))

        world_risk = torch.stack(risk_steps, dim=1).view(b, n, t, self.out_dim)
        world_summary = torch.stack(pooled_steps, dim=1).mean(dim=1).view(b, n, -1)
        world_risk_max = torch.sigmoid(world_risk[..., : self.num_risks]).amax(dim=2)
        world_progress = world_risk[..., self.num_risks : self.num_risks + 1].mean(dim=2)
        world_uncertainty = F.softplus(world_risk[..., self.num_risks + 1 : self.num_risks + 2]).mean(dim=2)
        world_stats = torch.cat([world_risk_max, world_progress, world_uncertainty], dim=-1)

        direct_context = self.direct_context_proj(direct_context)
        selector_input = torch.cat([direct_context, world_summary, direct_raw, world_stats], dim=-1)
        fused_raw = direct_raw + self.selector(selector_input)

        outputs = {
            "risk": fused_raw[:, :, None].expand(b, n, t, self.out_dim),
            "direct_risk": direct_raw[:, :, None].expand(b, n, t, self.out_dim),
            "world_risk": world_risk,
        }
        if return_world_outputs:
            occ = torch.stack(occ_steps, dim=1).view(b, n, t, *occ_steps[0].shape[1:])
            outputs["future_occupancy"] = occ
        return outputs
