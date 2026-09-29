from __future__ import annotations

from dataclasses import dataclass

import torch

from rwc.data.trajectory import comfort_cost


@dataclass(frozen=True)
class CriticWeights:
    collision: float = 8.0
    offroad: float = 5.0
    redlight: float = 3.0
    uncertainty: float = 1.0
    progress: float = 1.0
    comfort: float = 0.2


class WorldCritic:
    """Converts world-model rollout heads into scalar trajectory scores."""

    def __init__(self, weights: CriticWeights, dt: float) -> None:
        self.weights = weights
        self.dt = dt

    def score(self, rollout: dict[str, torch.Tensor], trajectories: torch.Tensor) -> dict[str, torch.Tensor]:
        risk = rollout["risk"]
        risk_prob = torch.sigmoid(risk[..., :3]).amax(dim=2)
        progress = risk[..., 3].mean(dim=2)
        uncertainty = torch.nn.functional.softplus(risk[..., 4]).mean(dim=2)
        comfort = comfort_cost(trajectories, self.dt)
        score = (
            self.weights.progress * progress
            - self.weights.collision * risk_prob[..., 0]
            - self.weights.offroad * risk_prob[..., 1]
            - self.weights.redlight * risk_prob[..., 2]
            - self.weights.uncertainty * uncertainty
            - self.weights.comfort * comfort
        )
        return {
            "score": score,
            "risk_prob": risk_prob,
            "progress": progress,
            "uncertainty": uncertainty,
            "comfort": comfort,
        }
