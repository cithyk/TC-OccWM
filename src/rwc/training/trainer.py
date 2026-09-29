from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch


def autocast_context(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def grad_accum_steps(batch_size: int, effective_batch_size: int) -> int:
    return max(1, effective_batch_size // max(batch_size, 1))
