from __future__ import annotations

import torch

from rwc.models.critic import WorldCritic
from rwc.planning.candidate_filter import FilterSpec, candidate_valid_mask


class TrajectoryReranker:
    def __init__(self, world_model, critic: WorldCritic, filter_spec: FilterSpec) -> None:
        self.world_model = world_model
        self.critic = critic
        self.filter_spec = filter_spec

    @torch.no_grad()
    def select(self, z: torch.Tensor, candidates: torch.Tensor, perturb: torch.Tensor) -> dict[str, torch.Tensor]:
        rollout = self.world_model(z, candidates, perturb)
        scores = self.critic.score(rollout, candidates)
        valid = candidate_valid_mask(candidates, self.filter_spec)
        masked_score = scores["score"].masked_fill(~valid, -1e6)
        best_idx = masked_score.argmax(dim=1)
        batch_idx = torch.arange(candidates.shape[0], device=candidates.device)
        return {
            "best_trajectory": candidates[batch_idx, best_idx],
            "best_index": best_idx,
            "scores": masked_score,
            "valid": valid,
            "rollout": rollout,
            **scores,
        }
