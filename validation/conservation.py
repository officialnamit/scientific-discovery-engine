"""
Conservation-law validation: does a trained model respect the GLOBAL balance
law implied by a candidate equation?

Integrating a candidate u_t = f(u, u_x, u_xx, ...) over the domain gives

    dM/dt = integral of f dx,      M(t) = integral of u dx

i.e. the total amount of u may change only through what the law says
(boundary flux of the u_xx / u_x terms, plus any source/sink terms). The
pointwise physics residual can be small almost everywhere while still
leaking or creating "mass" systematically; this checks the integrated
(weak-form) balance instead.

    violation(t)  = integral of residual(x, t) dx      (= dM/dt - integral f dx)
    relative      = RMS_t violation / RMS_t (dM/dt)

Like physics_validation.py, it is evaluated on a fresh (x, t) grid, never on
training points, and works for ANY compiled candidate residual.
"""

from typing import Callable, Dict

import numpy as np
import torch

from pinn.derivatives import compute_derivatives

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz


def conservation_violation(
    model: torch.nn.Module,
    residual_fn: Callable,
    params: Dict[str, torch.Tensor],
    x_min: float, x_max: float, t_min: float, t_max: float,
    n_x: int = 101, n_t: int = 41,
) -> Dict:
    """Global balance violation of `model` w.r.t. the law encoded by `residual_fn`
    (residual = u_t - f, unit u_t coefficient) on t in [t_min, t_max]."""
    x = np.linspace(x_min, x_max, n_x)
    t = np.linspace(t_min, t_max, n_t)
    X, T = np.meshgrid(x, t)  # shape (n_t, n_x)
    xt = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
    tt = torch.tensor(T.reshape(-1, 1), dtype=torch.float32)

    derivs = compute_derivatives(model, xt, tt)
    residual = residual_fn(derivs, params).detach().numpy().reshape(n_t, n_x)
    u_t = derivs["u_t"].detach().numpy().reshape(n_t, n_x)

    violation = _trapz(residual, x, axis=1)  # dM/dt - integral f dx, per time
    dM_dt = _trapz(u_t, x, axis=1)
    rms_violation = float(np.sqrt(np.mean(violation ** 2)))
    rms_dM_dt = float(np.sqrt(np.mean(dM_dt ** 2)))
    return {
        "rms_balance_violation": rms_violation,
        "rms_dM_dt": rms_dM_dt,
        "relative_violation": rms_violation / rms_dM_dt if rms_dM_dt > 0 else float("nan"),
        "max_abs_violation": float(np.max(np.abs(violation))),
        "t_window": [t_min, t_max],
    }
