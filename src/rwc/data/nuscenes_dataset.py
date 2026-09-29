from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rwc.config import BEVSpec
from rwc.data.bev_rasterizer import BEVRasterizer


class SyntheticDrivingDataset(Dataset):
    """Tiny procedural dataset for smoke testing without nuScenes files."""

    def __init__(self, spec: BEVSpec, size: int, horizon_steps: int, dt: float, seed: int = 7) -> None:
        self.spec = spec
        self.size = size
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.seed = seed
        self.rasterizer = BEVRasterizer(spec)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed + idx)
        sample = self.rasterizer.synthetic_scene(rng, self.horizon_steps, self.dt)
        return {
            "bev": torch.from_numpy(sample["bev"]),
            "ego": torch.from_numpy(sample["ego"]),
            "future_ego": torch.from_numpy(sample["future_ego"]),
            "future_agents": torch.from_numpy(sample["future_agents"]),
            "future_occupancy": torch.from_numpy(sample["future_occupancy"]),
            "risk": torch.from_numpy(sample["risk"]),
            "scene_id": f"synthetic_{idx:06d}",
            "sample_token": f"synthetic_{idx:06d}",
        }


class CachedNuScenesDataset(Dataset):
    """Reads prepared `.npz` samples from `prepare_nuscenes_cache.py`."""

    def __init__(self, cache_root: str | Path, split: str) -> None:
        root = Path(cache_root)
        self.split_dir = root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Cache split not found: {self.split_dir}. Run scripts/prepare_nuscenes_cache.py first."
            )
        self.files = sorted(self.split_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz cache samples found under {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        data = np.load(path, allow_pickle=False)
        return {
            "bev": torch.from_numpy(data["bev"].astype(np.float32)),
            "ego": torch.from_numpy(data["ego"].astype(np.float32)),
            "future_ego": torch.from_numpy(data["future_ego"].astype(np.float32)),
            "future_agents": torch.from_numpy(data["future_agents"].astype(np.float32)),
            "future_occupancy": torch.from_numpy(data["future_occupancy"].astype(np.float32)),
            "risk": torch.from_numpy(data["risk"].astype(np.float32)),
            "scene_id": str(data["scene_id"]),
            "sample_token": str(data["sample_token"]),
        }


def build_dataset(cfg: Any, spec: BEVSpec, split: str, synthetic: bool) -> Dataset:
    if synthetic:
        return SyntheticDrivingDataset(
            spec=spec,
            size=int(cfg.data.synthetic_size),
            horizon_steps=int(cfg.planning.horizon_steps),
            dt=float(cfg.planning.dt),
            seed=int(cfg.seed) + (0 if split == "train" else 10000),
        )
    cache_root = str(cfg.data.cache_root)
    if not cache_root:
        raise ValueError("data.cache_root is empty. Fill it in configs/default.yaml or use --synthetic.")
    return CachedNuScenesDataset(cache_root, split)
