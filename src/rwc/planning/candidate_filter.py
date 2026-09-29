from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FilterSpec:
    max_speed_mps: float = 18.0
    max_abs_y: float = 38.0
    min_x: float = -10.0
    max_x: float = 80.0


def candidate_valid_mask(trajectories: torch.Tensor, spec: FilterSpec) -> torch.Tensor:
    x = trajectories[..., 0]
    y = trajectories[..., 1]
    v = trajectories[..., 3]
    return (
        (x >= spec.min_x).all(dim=-1)
        & (x <= spec.max_x).all(dim=-1)
        & (y.abs() <= spec.max_abs_y).all(dim=-1)
        & (v.abs() <= spec.max_speed_mps).all(dim=-1)
    )
