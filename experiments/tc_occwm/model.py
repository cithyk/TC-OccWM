from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rwc.models.encoder import BEVDecoder
from rwc.models.no_rollout_scorer import NoRolloutTrajectoryScorer
from rwc.models.world_model import ConvGRUCell, TrajectoryConditioner


CANDIDATE_SOURCE_IDS = {
    "proposal": 0,
    "lattice": 1,
    "combined": 2,
}


def direct_score_from_raw(raw: torch.Tensor, num_risks: int) -> torch.Tensor:
    risk = torch.sigmoid(raw[..., :num_risks]).sum(dim=-1)
    progress = raw[..., num_risks]
    uncertainty = F.softplus(raw[..., num_risks + 1])
    return progress - risk - 0.1 * uncertainty


class TCOccWM(nn.Module):
    """Trajectory-conditioned occupancy world model for candidate reranking."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        traj_dim: int,
        perturb_dim: int,
        horizon_steps: int,
        occupancy_channels: int,
        num_risks: int,
        selector_feature_dim: int = 7,
        num_candidate_sources: int = 0,
        source_embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.traj_dim = traj_dim
        self.num_risks = num_risks
        self.out_dim = num_risks + 2
        self.num_candidate_sources = int(num_candidate_sources)
        self.source_embed_dim = int(source_embed_dim) if self.num_candidate_sources > 0 else 0

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
        self.uncertainty_head = BEVDecoder(hidden_dim, 1)
        self.risk_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.world_risk_head = nn.Sequential(
            nn.Linear(hidden_dim + traj_dim + perturb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )
        self.source_embedding = (
            nn.Embedding(self.num_candidate_sources, self.source_embed_dim)
            if self.num_candidate_sources > 0
            else None
        )
        self.selector = nn.Sequential(
            nn.Linear(self.out_dim + selector_feature_dim + self.source_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._zero_init_selector_residual()

    def _zero_init_selector_residual(self) -> None:
        last = self.selector[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _direct_raw(self, z: torch.Tensor, trajectories: torch.Tensor, perturb: torch.Tensor) -> torch.Tensor:
        b, n, t, d = trajectories.shape
        if t != self.horizon_steps or d != self.traj_dim:
            raise ValueError(f"Expected trajectories [B,N,{self.horizon_steps},{self.traj_dim}], got {tuple(trajectories.shape)}")
        raw = self.direct(z, trajectories, perturb)["risk"]
        return raw[:, :, 0]

    def forward(self, z: torch.Tensor, trajectories: torch.Tensor, perturb: torch.Tensor) -> dict[str, torch.Tensor]:
        b, n, t, d = trajectories.shape
        _, _, h, w = z.shape
        direct_raw = self._direct_raw(z, trajectories, perturb)

        z_rep = z[:, None].expand(b, n, *z.shape[1:]).reshape(b * n, *z.shape[1:])
        traj = trajectories.reshape(b * n, t, d)
        perturb_rep = perturb[:, None].expand(b, n, perturb.shape[-1]).reshape(b * n, perturb.shape[-1])
        hidden = self.input_proj(z_rep)

        occ_steps = []
        uncertainty_steps = []
        risk_steps = []
        for step in range(self.horizon_steps):
            cond = self.conditioner(traj[:, step], perturb_rep, h, w)
            hidden = self.gru(cond, hidden)
            occ_steps.append(self.occ_head(hidden))
            uncertainty_steps.append(self.uncertainty_head(hidden))
            pooled = self.risk_pool(hidden)
            risk_steps.append(self.world_risk_head(torch.cat([pooled, traj[:, step], perturb_rep], dim=-1)))

        occ = torch.stack(occ_steps, dim=1).view(b, n, t, *occ_steps[0].shape[1:])
        uncertainty = torch.stack(uncertainty_steps, dim=1).view(b, n, t, *uncertainty_steps[0].shape[1:])
        world_risk = torch.stack(risk_steps, dim=1).view(b, n, t, self.out_dim)
        return {
            "direct_raw": direct_raw,
            "future_occupancy": occ,
            "world_uncertainty": uncertainty,
            "world_risk": world_risk,
        }

    def _source_features(self, selector_features: torch.Tensor, source_ids: torch.Tensor | None) -> torch.Tensor | None:
        if self.source_embedding is None:
            return None
        if source_ids is None:
            source_ids = torch.zeros(selector_features.shape[:2], device=selector_features.device, dtype=torch.long)
        if source_ids.ndim == 1:
            source_ids = source_ids[:, None].expand(-1, selector_features.shape[1])
        source_ids = source_ids.to(device=selector_features.device, dtype=torch.long).clamp(0, self.num_candidate_sources - 1)
        return self.source_embedding(source_ids)

    def score(
        self,
        outputs: dict[str, torch.Tensor],
        selector_features: torch.Tensor,
        source_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        direct_raw = outputs["direct_raw"]
        direct_score = direct_score_from_raw(direct_raw, self.num_risks)
        extra = self._source_features(selector_features, source_ids)
        inputs = [direct_raw, selector_features]
        if extra is not None:
            inputs.append(extra)
        residual = self.selector(torch.cat(inputs, dim=-1)).squeeze(-1)
        return direct_score + residual
