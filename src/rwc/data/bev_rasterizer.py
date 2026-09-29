from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rwc.config import BEVSpec
from rwc.utils.geometry import oriented_box_corners, polygon_mask


@dataclass(frozen=True)
class AgentBox:
    x: float
    y: float
    length: float
    width: float
    yaw: float


class BEVRasterizer:
    """Rasterizes symbolic map/agent state into an ego-centric BEV tensor."""

    def __init__(self, spec: BEVSpec) -> None:
        self.spec = spec
        self.channel_to_idx = {name: i for i, name in enumerate(spec.channels)}

    def xy_to_rc(self, xy: np.ndarray) -> np.ndarray:
        x = xy[..., 0]
        y = xy[..., 1]
        row = (x - self.spec.x_min) / self.spec.resolution
        col = (y - self.spec.y_min) / self.spec.resolution
        return np.stack([row, col], axis=-1)

    def empty(self) -> np.ndarray:
        return np.zeros((self.spec.num_channels, self.spec.height, self.spec.width), dtype=np.float32)

    def rasterize_boxes(self, bev: np.ndarray, boxes: list[AgentBox], channel: str = "agents") -> None:
        if channel not in self.channel_to_idx:
            return
        c = self.channel_to_idx[channel]
        for box in boxes:
            corners = oriented_box_corners(box.x, box.y, box.length, box.width, box.yaw)
            polygon_rc = self.xy_to_rc(corners)
            bev[c] = np.maximum(bev[c], polygon_mask(self.spec.height, self.spec.width, polygon_rc).astype(np.float32))

    def rasterize_polyline(self, bev: np.ndarray, points_xy: np.ndarray, channel: str, radius: int = 1) -> None:
        if channel not in self.channel_to_idx or len(points_xy) == 0:
            return
        c = self.channel_to_idx[channel]
        rc = np.round(self.xy_to_rc(points_xy)).astype(int)
        for r, col in rc:
            r0, r1 = max(r - radius, 0), min(r + radius + 1, self.spec.height)
            c0, c1 = max(col - radius, 0), min(col + radius + 1, self.spec.width)
            bev[c, r0:r1, c0:c1] = 1.0

    def rasterize_ego(self, bev: np.ndarray) -> None:
        if "ego" not in self.channel_to_idx:
            return
        ego = AgentBox(x=0.0, y=0.0, length=4.8, width=2.0, yaw=0.0)
        self.rasterize_boxes(bev, [ego], channel="ego")

    def synthetic_scene(
        self,
        rng: np.random.Generator,
        horizon_steps: int,
        dt: float,
        num_agents: int = 8,
    ) -> dict[str, np.ndarray]:
        bev = self.empty()
        if "drivable" in self.channel_to_idx:
            bev[self.channel_to_idx["drivable"], :, :] = 1.0
        if "lane" in self.channel_to_idx:
            for y in [-6.0, -2.0, 2.0, 6.0]:
                pts = np.stack([np.linspace(self.spec.x_min, self.spec.x_max, 120), np.full(120, y)], axis=-1)
                self.rasterize_polyline(bev, pts, "lane", radius=0)
        if "route" in self.channel_to_idx:
            route = np.stack([np.linspace(0.0, min(60.0, self.spec.x_max - 5.0), 80), np.zeros(80)], axis=-1)
            self.rasterize_polyline(bev, route, "route", radius=1)

        boxes = []
        agent_trajs = np.zeros((horizon_steps, num_agents, 5), dtype=np.float32)
        for i in range(num_agents):
            x = float(rng.uniform(5.0, 65.0))
            y = float(rng.choice([-6.0, -2.0, 2.0, 6.0]) + rng.normal(0, 0.4))
            v = float(rng.uniform(1.0, 10.0))
            boxes.append(AgentBox(x=x, y=y, length=4.6, width=2.0, yaw=0.0))
            for t in range(horizon_steps):
                agent_trajs[t, i] = np.array([x + v * dt * t, y, 4.6, 2.0, 0.0], dtype=np.float32)
        self.rasterize_boxes(bev, boxes)
        self.rasterize_ego(bev)

        speed = float(rng.uniform(4.0, 10.0))
        future = np.zeros((horizon_steps, 4), dtype=np.float32)
        for t in range(horizon_steps):
            future[t] = np.array([speed * dt * (t + 1), 0.0, 0.0, speed], dtype=np.float32)

        occ = np.zeros((horizon_steps, 3, self.spec.height, self.spec.width), dtype=np.float32)
        for t in range(horizon_steps):
            step_boxes = [
                AgentBox(x=float(a[0]), y=float(a[1]), length=float(a[2]), width=float(a[3]), yaw=float(a[4]))
                for a in agent_trajs[t]
            ]
            tmp = self.empty()
            self.rasterize_boxes(tmp, step_boxes)
            occ[t, 0] = tmp[self.channel_to_idx.get("agents", 0)]
            occ[t, 1] = 1.0 - bev[self.channel_to_idx.get("drivable", 0)]
            occ[t, 2] = bev[self.channel_to_idx.get("route", 0)]

        risk = np.array([0.0, 0.0, 0.0, future[-1, 0] / 60.0], dtype=np.float32)
        return {
            "bev": bev,
            "future_ego": future,
            "future_agents": agent_trajs,
            "future_occupancy": occ,
            "risk": risk,
            "ego": np.array([0.0, 0.0, 0.0, speed], dtype=np.float32),
        }
