"""
Phase 2, Step 4: Sparse regression / SINDy.

Solves, for target y (here y = u_t) and candidate library Theta:

    y ≈ Theta @ xi

for a SPARSE coefficient vector xi, using Sequential Thresholded Least
Squares (STLSQ; Brunton, Proctor & Kutz, PNAS 2016 -- the original SINDy
paper). Terms whose fitted coefficient magnitude falls below `threshold`
after each least-squares fit are zeroed out and permanently excluded,
and the remaining terms are re-fit. This repeats until the active set
stops changing (or max_iterations is reached).

This module tries to use pysindy's own STLSQ optimizer (so we are
actually "using PySINDy", as the Phase 2 spec asks) and transparently
falls back to a self-contained implementation of the same algorithm if
pysindy is not importable or its API doesn't match what's expected --
the discovery pipeline should not hard-fail just because a dependency
version drifted. Whichever path is used is reported in the result under
"backend" so runs are honest about what actually happened.

We do NOT search for the string "u_xx" or otherwise special-case any
term name -- STLSQ treats every column of Theta identically.
"""

from typing import Dict, List, Sequence

import numpy as np

try:
    from pysindy.optimizers import STLSQ as _PySindySTLSQ
    _HAS_PYSINDY = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_PYSINDY = False


def stlsq(
    Theta: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.05,
    alpha: float = 0.0,
    max_iterations: int = 20,
    n_features_min: int = 1,
) -> np.ndarray:
    """
    Self-contained Sequential Thresholded Least Squares.

    Columns of Theta are normalized before thresholding (so `threshold`
    is comparable across terms of very different physical scale, e.g.
    a constant term vs. u_xx), and the fitted coefficients are rescaled
    back to the original units before returning.
    """
    n_samples, n_features = Theta.shape
    col_norms = np.linalg.norm(Theta, axis=0)
    col_norms[col_norms == 0] = 1.0
    Theta_n = Theta / col_norms

    def _fit(active_mask):
        Ta = Theta_n[:, active_mask]
        if alpha > 0:
            A = Ta.T @ Ta + alpha * np.eye(active_mask.sum())
            b = Ta.T @ y
            return np.linalg.solve(A, b)
        sol, *_ = np.linalg.lstsq(Ta, y, rcond=None)
        return sol

    active = np.ones(n_features, dtype=bool)
    xi = np.zeros(n_features)
    xi[active] = _fit(active)

    for _ in range(max_iterations):
        small = np.abs(xi) < threshold
        newly_small = small & active
        if not newly_small.any():
            break
        if active.sum() - newly_small.sum() < n_features_min:
            break
        active[newly_small] = False
        xi[~active] = 0.0
        if not active.any():
            break
        xi[active] = _fit(active)

    return xi / col_norms


def run_sindy(
    Theta: np.ndarray,
    y: np.ndarray,
    term_names: Sequence[str],
    threshold: float = 0.05,
    alpha: float = 0.0,
    max_iterations: int = 20,
    coeff_report_tol: float = 1e-10,
) -> Dict:
    """
    Fit y ~ Theta @ xi with STLSQ and return a structured discovery result:

        {
          "backend": "pysindy.optimizers.STLSQ" | "manual_stlsq" | "manual_stlsq (pysindy unavailable)",
          "lhs": "u_t",
          "terms": [{"term": "u_xx", "coefficient": 0.098}, ...],   # nonzero only
          "all_coefficients": {"1": 0.0, "u": 0.0, "u_x": 0.0, ...} # every candidate term
        }
    """
    xi = None
    backend = None

    if _HAS_PYSINDY:
        try:
            opt = _PySindySTLSQ(threshold=threshold, alpha=alpha, max_iter=max_iterations)
            opt.fit(Theta, y.reshape(-1, 1))
            xi = np.asarray(opt.coef_).ravel()
            backend = "pysindy.optimizers.STLSQ"
        except Exception as e:  # pragma: no cover - defensive fallback
            xi = None
            backend = f"manual_stlsq (pysindy raised: {type(e).__name__})"

    if xi is None:
        xi = stlsq(Theta, y, threshold=threshold, alpha=alpha, max_iterations=max_iterations)
        if backend is None:
            backend = "manual_stlsq (pysindy unavailable)"

    all_coeffs = {name: float(c) for name, c in zip(term_names, xi)}
    terms = [
        {"term": name, "coefficient": float(c)}
        for name, c in zip(term_names, xi)
        if abs(c) > coeff_report_tol
    ]

    return {
        "backend": backend,
        "lhs": "u_t",
        "terms": terms,
        "all_coefficients": all_coeffs,
    }


def fit_restricted(Theta_subset: np.ndarray, y: np.ndarray) -> Dict:
    """
    Ordinary (non-sparse) least-squares fit restricted to a fixed set of
    terms -- used to score a SPECIFIC hypothesis (e.g. "only u_x") against
    the data, for the competing-hypothesis comparison in Step 9. No
    thresholding: every provided term is kept, however small.

    Returns fitted coefficients, residual sum of squares, and R^2.
    """
    xi, residuals, rank, _ = np.linalg.lstsq(Theta_subset, y, rcond=None)
    y_pred = Theta_subset @ xi
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"coefficients": xi.tolist(), "ss_res": ss_res, "r2": r2}
