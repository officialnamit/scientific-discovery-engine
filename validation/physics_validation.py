"""
Phase 3, Part 11-12 support: evaluate physics-residual consistency of a
trained PINN on a FRESH grid of points (not the training collocation
points), so the reported physics_residual reflects generalized physical
consistency rather than just how well training converged on the exact
points it saw.
"""

from typing import Dict

import numpy as np
import torch

from pinn.derivatives import compute_derivatives


def evaluate_physics_residual(
    model, residual_fn, params: Dict[str, torch.Tensor],
    x_min: float, x_max: float, t_min: float, t_max: float,
    n_x: int = 60, n_t: int = 60,
) -> float:
    """Mean squared PDE residual over a fresh (n_x * n_t) grid spanning
    [x_min,x_max] x [t_min,t_max]. Returns a plain float."""
    x_lin = np.linspace(x_min, x_max, n_x)
    t_lin = np.linspace(t_min, t_max, n_t)
    Xg, Tg = np.meshgrid(x_lin, t_lin)
    x = torch.tensor(Xg.ravel(), dtype=torch.float32).reshape(-1, 1)
    t = torch.tensor(Tg.ravel(), dtype=torch.float32).reshape(-1, 1)

    derivs = compute_derivatives(model, x, t)
    residual = residual_fn(derivs, params)
    return float(torch.mean(residual ** 2).item())
