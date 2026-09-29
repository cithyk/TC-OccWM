from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PerturbationSpec:
    max_delay_steps: int = 2
    bev_dropout_prob: float = 0.1
    agent_dropout_prob: float = 0.15
    ego_noise_std: tuple[float, float, float, float] = (0.15, 0.15, 0.02, 0.2)


def sample_perturb_tokens(batch_size: int, spec: PerturbationSpec, device: torch.device) -> torch.Tensor:
    """Return compact perturb tokens [B, 8]."""
    delay = torch.randint(0, spec.max_delay_steps + 1, (batch_size, 1), device=device).float()
    bev_dropout = torch.rand(batch_size, 1, device=device) * spec.bev_dropout_prob
    agent_dropout = torch.rand(batch_size, 1, device=device) * spec.agent_dropout_prob
    ego_noise = torch.randn(batch_size, 4, device=device) * torch.tensor(spec.ego_noise_std, device=device)
    is_noisy = torch.ones(batch_size, 1, device=device)
    return torch.cat([delay, bev_dropout, agent_dropout, ego_noise, is_noisy], dim=-1)


def clean_perturb_tokens(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch_size, 8, device=device)


def apply_bev_dropout(bev: torch.Tensor, dropout_prob: torch.Tensor) -> torch.Tensor:
    """Drop random BEV cells. dropout_prob is [B,1]."""
    if bev.ndim != 4:
        raise ValueError("bev must have shape [B,C,H,W]")
    keep = torch.rand(bev.shape[0], 1, bev.shape[2], bev.shape[3], device=bev.device) > dropout_prob[:, :, None, None]
    return bev * keep.to(bev.dtype)
