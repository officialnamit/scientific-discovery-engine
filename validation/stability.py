"""
Phase 6: EMPIRICAL STABILITY (not Bayesian uncertainty, not a confidence
interval on the PINN's parameters).

Retraining a PINN per data subset costs ~4 min per fit; with several
candidates x several subsets that does not fit this project's budget.
Instead, each candidate's term structure is re-fitted cheaply on B
random subsets of the TRAINING observations using Phase 2's existing
derivative pipeline (scatter -> grid reconstruction -> smoothing ->
finite differences, equation_discovery/derivatives.py) and ordinary
least squares on the candidate's own field monomials.

Reported per monomial: mean coefficient, std, and sign consistency
(fraction of subsets agreeing with the majority sign). A term whose
coefficient flips sign across subsets is not stably supported by the
data. Limitation (stated in the report): this measures stability of a
noisy finite-difference regression, which Phase 2 showed is itself
fragile on this dataset -- it is supporting evidence, not a verdict.
"""

from typing import Dict, List

import numpy as np
import sympy

from equation_discovery.derivatives import finite_diff_derivatives, reconstruct_grid, smooth_grid
from hypotheses.equation_compiler import FIELD_SYMBOLS, parse_equation
from llm.schemas import Hypothesis


def candidate_monomials(hyp: Hypothesis):
    """Distinct field-only parts of the RHS terms, e.g. reaction-diffusion ->
    [u_xx, u, u**2] (parameters stripped, so OLS coefficients absorb them)."""
    param_names = [p.name for p in hyp.parameters]
    residual_expr, _, _, local_dict = parse_equation(hyp.equation, param_names)
    u_t = local_dict["u_t"]
    expanded = sympy.expand(residual_expr)
    rhs = sympy.expand(-(expanded - expanded.coeff(u_t) * u_t))
    field_syms = [local_dict[f] for f in FIELD_SYMBOLS if f != "u_t"]
    monos = []
    for term in sympy.Add.make_args(rhs):
        _, field_part = term.as_independent(*field_syms, as_Add=False)
        if field_part != 1 and field_part not in monos:
            monos.append(field_part)
    return monos, field_syms


def bootstrap_stability(hyp: Hypothesis, obs: Dict[str, np.ndarray], x_grid, t_grid,
                        smoothing_sigma: float, crop: int, n_boot: int = 20,
                        frac: float = 0.8, seed: int = 0) -> Dict:
    monos, field_syms = candidate_monomials(hyp)
    names = [str(m) for m in monos]
    funcs = [sympy.lambdify(field_syms, m, "numpy") for m in monos]
    rng = np.random.default_rng(seed)
    n = len(obs["u"])
    coefs: List[np.ndarray] = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        U, _, _ = reconstruct_grid(obs["x"][idx], obs["t"][idx], obs["u"][idx], x_grid, t_grid)
        U = smooth_grid(U, smoothing_sigma)
        f = finite_diff_derivatives(U, x_grid, t_grid, crop=crop)
        y = f["u_t"].ravel()
        cols = [np.broadcast_to(np.asarray(fn(*[f[s.name].ravel() for s in field_syms]), dtype=float), y.shape)
                for fn in funcs]
        Theta = np.column_stack(cols)
        xi, *_ = np.linalg.lstsq(Theta, y, rcond=None)
        coefs.append(xi)
    C = np.array(coefs)
    out = {}
    for j, name in enumerate(names):
        col = C[:, j]
        majority = np.sign(np.median(col)) or 1.0
        out[name] = {
            "mean": float(col.mean()), "std": float(col.std()),
            "sign_consistency": float(np.mean(np.sign(col) == majority)),
        }
    stable = all(v["sign_consistency"] >= 0.9 for v in out.values())
    return {"monomials": out, "n_boot": n_boot, "subset_fraction": frac,
            "all_terms_sign_stable": bool(stable),
            "method": "bootstrap OLS on Phase-2 finite-difference derivatives (empirical stability, not Bayesian)"}
