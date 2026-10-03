"""
Phase 3, Part 4 & Part 5: loss terms.

    L = lambda_data    * L_data
      + lambda_physics * L_physics
      + lambda_bc       * L_bc
      + lambda_ic       * L_ic

All four are mean-squared-error terms; the only thing that differs
between them is which points and which target values go in.
"""

from typing import Dict

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def data_loss(model, x: torch.Tensor, t: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    u_pred = model(x, t)
    return mse(u_pred, u_true)


def physics_loss(residual: torch.Tensor) -> torch.Tensor:
    return torch.mean(residual ** 2)


def bc_loss(model, x_bc: torch.Tensor, t_bc: torch.Tensor, u_bc_target: torch.Tensor) -> torch.Tensor:
    u_pred = model(x_bc, t_bc)
    return mse(u_pred, u_bc_target)


def ic_loss(model, x_ic: torch.Tensor, t_ic: torch.Tensor, u_ic_target: torch.Tensor) -> torch.Tensor:
    u_pred = model(x_ic, t_ic)
    return mse(u_pred, u_ic_target)


def total_loss(components: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    total = 0.0
    for name, value in components.items():
        total = total + weights.get(f"lambda_{name}", 1.0) * value
    return total
