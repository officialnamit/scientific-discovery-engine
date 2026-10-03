"""
Phase 3, Part 12: out-of-distribution evaluation.

Splits the (already-generated, Phase 1) clean reference field into an
in-distribution region (t <= t_cutoff, used for training) and an OOD
region (t > t_cutoff, NEVER used for training), and compares PINN
predictions against the clean numerical solution in each region.
"""

from typing import Dict, Tuple

import numpy as np
import torch


def split_by_time(
    x_flat: np.ndarray, t_flat: np.ndarray, u_flat: np.ndarray, t_cutoff: float
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    in_mask = t_flat <= t_cutoff
    ood_mask = ~in_mask
    in_dist = {"x": x_flat[in_mask], "t": t_flat[in_mask], "u": u_flat[in_mask]}
    ood = {"x": x_flat[ood_mask], "t": t_flat[ood_mask], "u": u_flat[ood_mask]}
    return in_dist, ood


def prediction_error(model, x: np.ndarray, t: np.ndarray, u_true: np.ndarray) -> Dict[str, float]:
    """RMSE and MAE of model(x,t) against a reference solution u_true.
    x, t, u_true: 1D numpy arrays of matching length."""
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        tt = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
        u_pred = model(xt, tt).numpy().ravel()
    err = u_pred - u_true
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }
