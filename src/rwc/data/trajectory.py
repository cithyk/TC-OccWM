from __future__ import annotations

import numpy as np
import torch


def make_counterfactual_trajectories(
    expert: torch.Tensor,
    num_candidates: int,
    max_lateral_shift: float = 4.0,
    speed_scale_range: tuple[float, float] = (0.5, 1.5),
) -> torch.Tensor:
    """Create simple risky/near-risky trajectory candidates from an expert path.

    Args:
        expert: Tensor [T, 4] with x, y, yaw, speed in ego frame.
        num_candidates: Number of trajectories to create.
    """
    if expert.ndim != 2 or expert.shape[-1] != 4:
        raise ValueError("expert must have shape [T, 4]")

    candidates = expert.unsqueeze(0).repeat(num_candidates, 1, 1)
    device = expert.device
    dtype = expert.dtype

    shifts = torch.linspace(-max_lateral_shift, max_lateral_shift, num_candidates, device=device, dtype=dtype)
    candidates[..., 1] += shifts[:, None]

    speed_scales = torch.linspace(speed_scale_range[0], speed_scale_range[1], num_candidates, device=device, dtype=dtype)
    candidates[..., 0] *= speed_scales[:, None]
    candidates[..., 3] *= speed_scales[:, None]

    if num_candidates > 1:
        candidates[0] = expert
    return candidates


def comfort_cost(traj: torch.Tensor, dt: float) -> torch.Tensor:
    """Return jerk/yaw-rate comfort cost for trajectories [B,N,T,4] or [N,T,4]."""
    if traj.shape[-2] < 3:
        return torch.zeros(traj.shape[:-2], device=traj.device, dtype=traj.dtype)
    vel = torch.diff(traj[..., :2], dim=-2) / dt
    acc = torch.diff(vel, dim=-2) / dt
    jerk = torch.diff(acc, dim=-2) / dt if acc.shape[-2] > 1 else acc.new_zeros(acc.shape[:-2] + (1, 2))
    yaw_rate = torch.diff(traj[..., 2], dim=-1) / dt
    return jerk.norm(dim=-1).mean(dim=-1) + 0.1 * yaw_rate.abs().mean(dim=-1)


def min_ade_fde(candidates: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute minADE/minFDE for candidates [B,N,T,4] and target [B,T,4]."""
    err = torch.linalg.norm(candidates[..., :2] - target[:, None, :, :2], dim=-1)
    ade = err.mean(dim=-1)
    fde = err[..., -1]
    return ade.min(dim=1).values, fde.min(dim=1).values


def trajectory_box_collision(
    candidates: torch.Tensor,
    agents: torch.Tensor,
    ego_length: float = 4.8,
    ego_width: float = 2.0,
    margin: float = 0.0,
) -> torch.Tensor:
    """Return oriented footprint collision flags for candidates [B,N,T,4].

    Agents are [B,T,A,5] with x, y, length, width, yaw. The result is [B,N].
    """
    if candidates.ndim != 4 or candidates.shape[-1] < 3:
        raise ValueError("candidates must have shape [B,N,T,D>=3]")
    if agents.ndim != 4 or agents.shape[-1] < 5:
        raise ValueError("agents must have shape [B,T,A,5]")

    ego_xy = candidates[..., :2].unsqueeze(3)
    agent_xy = agents[:, None, :, :, :2]
    delta = agent_xy - ego_xy

    ego_yaw = candidates[..., 2].unsqueeze(3)
    agent_yaw = agents[:, None, :, :, 4]
    ego_u, ego_v = _torch_box_axes(ego_yaw)
    agent_u, agent_v = _torch_box_axes(agent_yaw)

    ego_hl = candidates.new_tensor(float(ego_length) * 0.5 + float(margin))
    ego_hw = candidates.new_tensor(float(ego_width) * 0.5 + float(margin))
    agent_hl = agents[:, None, :, :, 2].clamp_min(0.0) * 0.5 + float(margin)
    agent_hw = agents[:, None, :, :, 3].clamp_min(0.0) * 0.5 + float(margin)
    valid_agent = (agents[:, None, :, :, 2] > 0) & (agents[:, None, :, :, 3] > 0)

    overlap = (
        _overlap_on_axis(delta, ego_u, ego_hl, ego_hw, ego_u, ego_v, agent_hl, agent_hw, agent_u, agent_v)
        & _overlap_on_axis(delta, ego_v, ego_hl, ego_hw, ego_u, ego_v, agent_hl, agent_hw, agent_u, agent_v)
        & _overlap_on_axis(delta, agent_u, ego_hl, ego_hw, ego_u, ego_v, agent_hl, agent_hw, agent_u, agent_v)
        & _overlap_on_axis(delta, agent_v, ego_hl, ego_hw, ego_u, ego_v, agent_hl, agent_hw, agent_u, agent_v)
        & valid_agent
    )
    return overlap.any(dim=(2, 3)).float()


def trajectory_box_collision_np(
    future_ego: np.ndarray,
    future_agents: np.ndarray,
    ego_length: float = 4.8,
    ego_width: float = 2.0,
    margin: float = 0.0,
) -> float:
    """Numpy oriented footprint collision for one expert trajectory."""
    for step, ego in enumerate(future_ego):
        for agent in future_agents[step]:
            if agent[2] <= 0 or agent[3] <= 0:
                continue
            if _rectangles_overlap_np(
                float(ego[0]),
                float(ego[1]),
                ego_length,
                ego_width,
                float(ego[2]),
                float(agent[0]),
                float(agent[1]),
                float(agent[2]),
                float(agent[3]),
                float(agent[4]),
                margin,
            ):
                return 1.0
    return 0.0


def _torch_box_axes(yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    length_axis = torch.stack([c, s], dim=-1)
    width_axis = torch.stack([-s, c], dim=-1)
    return length_axis, width_axis


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1)


def _overlap_on_axis(
    delta: torch.Tensor,
    axis: torch.Tensor,
    ego_hl: torch.Tensor,
    ego_hw: torch.Tensor,
    ego_u: torch.Tensor,
    ego_v: torch.Tensor,
    agent_hl: torch.Tensor,
    agent_hw: torch.Tensor,
    agent_u: torch.Tensor,
    agent_v: torch.Tensor,
) -> torch.Tensor:
    ego_extent = ego_hl * _dot(ego_u, axis).abs() + ego_hw * _dot(ego_v, axis).abs()
    agent_extent = agent_hl * _dot(agent_u, axis).abs() + agent_hw * _dot(agent_v, axis).abs()
    return _dot(delta, axis).abs() <= ego_extent + agent_extent


def _rectangles_overlap_np(
    x1: float,
    y1: float,
    length1: float,
    width1: float,
    yaw1: float,
    x2: float,
    y2: float,
    length2: float,
    width2: float,
    yaw2: float,
    margin: float,
) -> bool:
    def axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        return np.array([c, s], dtype=np.float32), np.array([-s, c], dtype=np.float32)

    c1 = np.array([x1, y1], dtype=np.float32)
    c2 = np.array([x2, y2], dtype=np.float32)
    delta = c2 - c1
    u1, v1 = axes(yaw1)
    u2, v2 = axes(yaw2)
    h1 = (float(length1) * 0.5 + margin, float(width1) * 0.5 + margin)
    h2 = (float(length2) * 0.5 + margin, float(width2) * 0.5 + margin)

    for axis in (u1, v1, u2, v2):
        e1 = h1[0] * abs(float(np.dot(u1, axis))) + h1[1] * abs(float(np.dot(v1, axis)))
        e2 = h2[0] * abs(float(np.dot(u2, axis))) + h2[1] * abs(float(np.dot(v2, axis)))
        if abs(float(np.dot(delta, axis))) > e1 + e2:
            return False
    return True
