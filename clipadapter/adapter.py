from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """2-layer MLP adapter with bottleneck.

    Implements: f_adapter = W2(ReLU(W1(f_clip)))
    """

    def __init__(self, in_dim: int, bottleneck_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, self.in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

