from __future__ import annotations

import math

import numpy as np
import torch


def wrap_angle(angle: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def pairwise_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a - b, dim=-1)


def trajectory_ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return pairwise_l2(pred[..., :2], target[..., :2]).mean(dim=-1)


def trajectory_fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return pairwise_l2(pred[..., -1, :2], target[..., -1, :2])


def polygon_mask(height: int, width: int, polygon_rc: np.ndarray) -> np.ndarray:
    """Pure-numpy polygon fill for small BEV rasters."""
    rr_min = max(int(np.floor(polygon_rc[:, 0].min())), 0)
    rr_max = min(int(np.ceil(polygon_rc[:, 0].max())) + 1, height)
    cc_min = max(int(np.floor(polygon_rc[:, 1].min())), 0)
    cc_max = min(int(np.ceil(polygon_rc[:, 1].max())) + 1, width)
    mask = np.zeros((height, width), dtype=bool)
    if rr_min >= rr_max or cc_min >= cc_max:
        return mask

    ys, xs = np.mgrid[rr_min:rr_max, cc_min:cc_max]
    points = np.stack([ys + 0.5, xs + 0.5], axis=-1)
    poly = polygon_rc
    inside = np.zeros(points.shape[:2], dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        yi, xi = poly[i]
        yj, xj = poly[j]
        crosses = ((xi > points[..., 1]) != (xj > points[..., 1])) & (
            points[..., 0]
            < (yj - yi) * (points[..., 1] - xi) / (xj - xi + 1e-6) + yi
        )
        inside ^= crosses
        j = i
    mask[rr_min:rr_max, cc_min:cc_max] = inside
    return mask


def oriented_box_corners(x: float, y: float, length: float, width: float, yaw: float) -> np.ndarray:
    local = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ],
        dtype=np.float32,
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)
