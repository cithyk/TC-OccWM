from __future__ import annotations

import torch

from rwc.data.trajectory import min_ade_fde


@torch.no_grad()
def proposal_metrics(candidates: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    min_ade, min_fde = min_ade_fde(candidates, target)
    return {"minADE": float(min_ade.mean().cpu()), "minFDE": float(min_fde.mean().cpu())}


@torch.no_grad()
def ranking_accuracy(scores: torch.Tensor, candidate_is_expert: torch.Tensor) -> float:
    best = scores.argmax(dim=1)
    labels = candidate_is_expert.float().argmax(dim=1)
    return float((best == labels).float().mean().cpu())


class RunningMean:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, metrics: dict[str, float | torch.Tensor], n: int = 1) -> None:
        for key, value in metrics.items():
            scalar = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            self.values[key] = self.values.get(key, 0.0) + scalar * n
            self.counts[key] = self.counts.get(key, 0) + n

    def compute(self) -> dict[str, float]:
        return {key: self.values[key] / max(self.counts[key], 1) for key in self.values}
