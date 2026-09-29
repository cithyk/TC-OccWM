from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Small dict wrapper with attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_node(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({k: _to_node(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_node(v) for v in value]
    return value


def load_config(path: str | Path) -> ConfigNode:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _to_node(data)


@dataclass(frozen=True)
class BEVSpec:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float
    channels: tuple[str, ...]

    @property
    def height(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def width(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def num_channels(self) -> int:
        return len(self.channels)


def bev_spec_from_config(cfg: ConfigNode) -> BEVSpec:
    return BEVSpec(
        x_min=float(cfg.bev.x_min),
        x_max=float(cfg.bev.x_max),
        y_min=float(cfg.bev.y_min),
        y_max=float(cfg.bev.y_max),
        resolution=float(cfg.bev.resolution),
        channels=tuple(str(c) for c in cfg.bev.channels),
    )
