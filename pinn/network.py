"""
Phase 3, Part 1: reusable PINN architecture.

One fully-connected network, (x, t) -> u_theta(x, t). This same class
is used for every hypothesis (diffusion, advection, ...) tested in this
phase -- only the physics RESIDUAL changes between hypotheses (see
pinn/residuals.py), never the network architecture. This directly
implements the "IMPORTANT ARCHITECTURAL REQUIREMENT" from the spec.
"""

from typing import List

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class PINN(nn.Module):
    def __init__(self, hidden_dims: List[int] = (64, 64, 64, 64), activation: str = "tanh"):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        dims = [2] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x, t: shape (N, 1) each. Returns u_theta(x,t), shape (N, 1)."""
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)
