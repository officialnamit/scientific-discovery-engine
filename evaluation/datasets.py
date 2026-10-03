"""
Phase 7: ground-truth benchmark datasets for FOUR mechanism families.

Exact analytic solutions are used wherever they exist (diffusion, advection,
advection-diffusion), so the answer key carries no numerical-solver error.
Reaction-diffusion has no closed form; it is integrated with the generic
method-of-lines solver (validation/forward_solver.py, RK4) on a fine grid.

Ground truth (`truth` dict) is for SCORING ONLY. Discovery code receives
only (x, t, U) -- see evaluation/benchmark.py.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np


@dataclass
class Dataset:
    name: str
    family: str                 # "diffusion" | "advection" | "reaction-diffusion" | "advection-diffusion"
    truth: Dict[str, float]     # true parameter values (scoring only)
    true_equation: str
    x: np.ndarray
    t: np.ndarray
    U: np.ndarray               # shape (n_t, n_x)
    ic: Callable = field(repr=False, default=None)
    exact: Callable = field(repr=False, default=None)   # exact(x, t) for any t (None if numerical)
    x_range: tuple = (0.0, 1.0)


def _diffusion(D, nx=101, nt=101, t_end=1.0):
    x, t = np.linspace(0, 1, nx), np.linspace(0, t_end, nt)
    ex = lambda X, T: (np.sin(np.pi * X) * np.exp(-np.pi ** 2 * D * T)
                       + 0.5 * np.sin(3 * np.pi * X) * np.exp(-9 * np.pi ** 2 * D * T))
    X, T = np.meshgrid(x, t)
    return Dataset(f"diffusion_D{D}", "diffusion", {"D": D}, "u_t = D*u_xx",
                   x, t, ex(X, T), lambda xx: ex(xx, 0.0), ex)


def _gauss(x0, c, D, sigma=0.08, L=1.5, nx=151, nt=101, t_end=1.0):
    def ex(X, T):
        s2 = sigma ** 2 + 2 * D * T
        return sigma / np.sqrt(s2) * np.exp(-(X - x0 - c * T) ** 2 / (2 * s2))
    x, t = np.linspace(0, L, nx), np.linspace(0, t_end, nt)
    X, T = np.meshgrid(x, t)
    return x, t, ex(X, T), ex


def _advection(c, x0):
    x, t, U, ex = _gauss(x0, c, 0.0)
    return Dataset(f"advection_c{c}", "advection", {"c": c}, "u_t + c*u_x = 0",
                   x, t, U, lambda xx: ex(xx, 0.0), ex, (0.0, 1.5))


def _adv_diff(c, D, x0):
    x, t, U, ex = _gauss(x0, c, D)
    return Dataset(f"advdiff_c{c}_D{D}", "advection-diffusion", {"c": c, "D": D}, "u_t + c*u_x = D*u_xx",
                   x, t, U, lambda xx: ex(xx, 0.0), ex, (0.0, 1.5))


def _reaction_diffusion(D, r, K, nx=101, nt=101, t_end=1.0):
    from llm.schemas import Hypothesis
    from validation.forward_solver import solve_forward
    h = Hypothesis(id="RD", name="rd", equation="u_t = D*u_xx + r*u*(1-u/K)",
                   parameters=[{"name": "D"}, {"name": "r"}, {"name": "K"}])
    x = np.linspace(0, 1, nx)
    ic = lambda xx: 0.3 * (np.sin(np.pi * xx) + 0.5 * np.sin(3 * np.pi * xx))
    sol = solve_forward(h, {"D": D, "r": r, "K": K}, ic, x, t_end, n_out=nt, method="rk4")
    return Dataset(f"reacdiff_D{D}_r{r}_K{K}", "reaction-diffusion", {"D": D, "r": r, "K": K},
                   "u_t = D*u_xx + r*u*(1-u/K)", x, sol["t"], sol["U"], ic, None)


CACHE = "data/generated/benchmark_datasets.npz"


def benchmark_datasets(use_cache: bool = True):
    """12 datasets: 4 families x 3 parameter settings. Parameter values are chosen
    to span the family, NOT tuned to make discovery succeed. Cached to disk because
    reaction-diffusion integration takes ~1 min; callables (ic/exact) are rebuilt."""
    import os
    built = _build_all() if not (use_cache and os.path.exists(CACHE)) else None
    if built is not None:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, **{f"{d.name}__U": d.U for d in built}, **{f"{d.name}__t": d.t for d in built})
        return built
    data = np.load(CACHE)
    shells = _build_all(skip_numerical=True)
    for d in shells:
        d.U, d.t = data[f"{d.name}__U"], data[f"{d.name}__t"]
    return shells


def _build_all(skip_numerical: bool = False):
    rd = (lambda *a: _rd_shell(*a)) if skip_numerical else _reaction_diffusion
    return [
        _diffusion(0.05), _diffusion(0.1), _diffusion(0.2),
        _advection(0.3, 0.4), _advection(0.5, 0.35), _advection(-0.4, 1.1),
        rd(0.05, 1.0, 1.0), rd(0.02, 2.0, 1.0), rd(0.1, 1.5, 0.8),
        _adv_diff(0.3, 0.01, 0.4), _adv_diff(0.4, 0.02, 0.35), _adv_diff(-0.3, 0.015, 1.1),
    ]


def _rd_shell(D, r, K, nx=101):
    x = np.linspace(0, 1, nx)
    ic = lambda xx: 0.3 * (np.sin(np.pi * xx) + 0.5 * np.sin(3 * np.pi * xx))
    return Dataset(f"reacdiff_D{D}_r{r}_K{K}", "reaction-diffusion", {"D": D, "r": r, "K": K},
                   "u_t = D*u_xx + r*u*(1-u/K)", x, None, None, ic, None)


def add_noise(ds: Dataset, frac: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ds.U + rng.normal(0, frac * np.std(ds.U), ds.U.shape)
