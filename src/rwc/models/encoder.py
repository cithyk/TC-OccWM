from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BEVEncoder(nn.Module):
    """Small symbolic-BEV encoder."""

    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, latent_dim),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.net(bev)


class BEVDecoder(nn.Module):
    """Predict occupancy-style maps from latent BEV."""

    def __init__(self, latent_dim: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(latent_dim, latent_dim),
            nn.Conv2d(latent_dim, out_channels, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
