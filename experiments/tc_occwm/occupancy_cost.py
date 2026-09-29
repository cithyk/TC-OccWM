from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _footprint_offsets(
    device: torch.device,
    dtype: torch.dtype,
    ego_length: float,
    ego_width: float,
    samples_x: int,
    samples_y: int,
) -> torch.Tensor:
    xs = torch.linspace(-ego_length * 0.5, ego_length * 0.5, samples_x, device=device, dtype=dtype)
    ys = torch.linspace(-ego_width * 0.5, ego_width * 0.5, samples_y, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def trajectory_footprint_points(
    trajectories: torch.Tensor,
    ego_length: float = 4.8,
    ego_width: float = 2.0,
    samples_x: int = 5,
    samples_y: int = 3,
) -> torch.Tensor:
    """Return sampled ego-footprint points for every trajectory step.

    Args:
        trajectories: [B,N,T,4] with x, y, yaw, speed.

    Returns:
        points: [B,N,T,K,2].
    """
    offsets = _footprint_offsets(
        trajectories.device,
        trajectories.dtype,
        ego_length=ego_length,
        ego_width=ego_width,
        samples_x=samples_x,
        samples_y=samples_y,
    )
    yaw = trajectories[..., 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    ox = offsets[:, 0]
    oy = offsets[:, 1]
    dx = cos_yaw[..., None] * ox - sin_yaw[..., None] * oy
    dy = sin_yaw[..., None] * ox + cos_yaw[..., None] * oy
    center = trajectories[..., None, :2]
    return center + torch.stack([dx, dy], dim=-1)


def _xy_to_grid(points: torch.Tensor, cfg: Any, height: int, width: int) -> torch.Tensor:
    x_min = float(cfg.bev.x_min)
    x_max = float(cfg.bev.x_max)
    y_min = float(cfg.bev.y_min)
    y_max = float(cfg.bev.y_max)
    row = (points[..., 0] - x_min) / max(x_max - x_min, 1e-6) * 2.0 - 1.0
    col = (points[..., 1] - y_min) / max(y_max - y_min, 1e-6) * 2.0 - 1.0
    # grid_sample expects x=column and y=row.
    return torch.stack([col, row], dim=-1).clamp(-1.2, 1.2)


def _sample_channel(
    probs: torch.Tensor,
    trajectories: torch.Tensor,
    cfg: Any,
    channel: int,
    ego_length: float,
    ego_width: float,
    samples_x: int,
    samples_y: int,
) -> torch.Tensor:
    """Sample one occupancy channel at ego-footprint points.

    Args:
        probs: [B,N,T,C,H,W].

    Returns:
        sampled values [B,N,T,K].
    """
    b, n, t, c, h, w = probs.shape
    points = trajectory_footprint_points(
        trajectories,
        ego_length=ego_length,
        ego_width=ego_width,
        samples_x=samples_x,
        samples_y=samples_y,
    )
    grid = _xy_to_grid(points, cfg, h, w).reshape(b * n * t, -1, 1, 2)
    src = probs[:, :, :, channel].reshape(b * n * t, 1, h, w)
    sampled = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.reshape(b, n, t, -1)


def sample_trajectory_occupancy_cost(
    occupancy_logits: torch.Tensor,
    trajectories: torch.Tensor,
    cfg: Any,
    uncertainty_logits: torch.Tensor | None = None,
    ego_length: float = 4.8,
    ego_width: float = 2.0,
    samples_x: int = 5,
    samples_y: int = 3,
) -> dict[str, torch.Tensor]:
    """Convert candidate-conditioned occupancy maps into structured trajectory costs.

    Channel convention follows the existing cache: 0=agents, 1=off-drivable,
    2=route. Missing channels fall back to zeros.
    """
    probs = torch.sigmoid(occupancy_logits)
    b, n, t, c, _, _ = probs.shape
    zeros = torch.zeros(b, n, device=probs.device, dtype=probs.dtype)

    agent_sampled = _sample_channel(
        probs,
        trajectories,
        cfg,
        channel=0,
        ego_length=ego_length,
        ego_width=ego_width,
        samples_x=samples_x,
        samples_y=samples_y,
    )
    collision_cost = agent_sampled.amax(dim=(2, 3))
    collision_mean = agent_sampled.mean(dim=(2, 3))

    if c > 1:
        offroad_sampled = _sample_channel(
            probs,
            trajectories,
            cfg,
            channel=1,
            ego_length=ego_length,
            ego_width=ego_width,
            samples_x=samples_x,
            samples_y=samples_y,
        )
        offroad_cost = offroad_sampled.amax(dim=(2, 3))
    else:
        offroad_cost = zeros

    if c > 2:
        route_sampled = _sample_channel(
            probs,
            trajectories,
            cfg,
            channel=2,
            ego_length=ego_length,
            ego_width=ego_width,
            samples_x=samples_x,
            samples_y=samples_y,
        )
        route_score = route_sampled.mean(dim=(2, 3))
    else:
        route_score = zeros

    if uncertainty_logits is not None:
        uncertainty = torch.sigmoid(uncertainty_logits)
        uncertainty_sampled = _sample_channel(
            uncertainty,
            trajectories,
            cfg,
            channel=0,
            ego_length=ego_length,
            ego_width=ego_width,
            samples_x=samples_x,
            samples_y=samples_y,
        )
        uncertainty_cost = uncertainty_sampled.mean(dim=(2, 3))
    else:
        bernoulli_entropy = -(probs * torch.log(probs.clamp_min(1e-6)) + (1.0 - probs) * torch.log((1.0 - probs).clamp_min(1e-6)))
        uncertainty_cost = bernoulli_entropy[:, :, :, 0].mean(dim=(2, 3, 4)) if bernoulli_entropy.ndim == 6 else zeros

    return {
        "collision_cost": collision_cost,
        "collision_mean": collision_mean,
        "offroad_cost": offroad_cost,
        "route_score": route_score,
        "uncertainty_cost": uncertainty_cost,
    }


def build_occupancy_selector_features(
    costs: dict[str, torch.Tensor],
    progress: torch.Tensor,
    comfort: torch.Tensor,
    normalize: bool = False,
    disable_uncertainty: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    features = torch.stack(
        [
            costs["collision_cost"],
            costs["collision_mean"],
            costs["offroad_cost"],
            costs["route_score"],
            costs["uncertainty_cost"],
            progress,
            comfort,
        ],
        dim=-1,
    )
    if normalize:
        occ = features[..., :5]
        mean = occ.mean(dim=1, keepdim=True)
        std = occ.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        features = torch.cat([(occ - mean) / std, features[..., 5:]], dim=-1)
    if disable_uncertainty:
        features = features.clone()
        # Column 4 is the occupancy-derived uncertainty cost.
        features[..., 4] = 0.0
    return features


def occupancy_cost_alignment_loss(costs: dict[str, torch.Tensor], collision_label: torch.Tensor) -> torch.Tensor:
    pred = costs["collision_cost"].clamp(1e-4, 1.0 - 1e-4)
    return F.binary_cross_entropy(pred, collision_label.float())
