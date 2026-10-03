"""
Phase 2, Steps 5-6-9: orchestration, equation representation, evaluation,
and competing-hypothesis scoring.

This module is PDE-agnostic: it does not know "the answer" is u_xx. Give
it a different set of term_specs and it will happily try to discover a
different equation from whatever fields you hand it (see
tests/test_equation_discovery.py for a worked advection example).
"""

from typing import Dict, List, Sequence

import numpy as np

from equation_discovery.derivatives import (
    finite_diff_derivatives,
    reconstruct_grid,
    smooth_grid,
)
from equation_discovery.library import TermSpec, build_library
from equation_discovery.sindy import fit_restricted, run_sindy


def _fields_from_grid(U_grid, x_grid, t_grid, crop):
    fields = finite_diff_derivatives(U_grid, x_grid, t_grid, crop=crop)
    y = fields.pop("u_t").ravel()
    fields.pop("x")
    fields.pop("t")
    flat_fields = {k: v.ravel() for k, v in fields.items()}
    return flat_fields, y


def discover_from_grid(
    U_grid: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    term_specs: Sequence[TermSpec],
    crop: int = 3,
    sindy_kwargs: Dict = None,
) -> Dict:
    """Run the full discovery pipeline starting from data already on a
    regular grid (the "clean" full-field case)."""
    sindy_kwargs = sindy_kwargs or {}
    flat_fields, y = _fields_from_grid(U_grid, x_grid, t_grid, crop)
    Theta, names = build_library(flat_fields, term_specs)
    result = run_sindy(Theta, y, names, **sindy_kwargs)
    result["n_samples_used"] = int(len(y))
    return result, flat_fields, y


def discover_from_scatter(
    x: np.ndarray, t: np.ndarray, u: np.ndarray,
    x_grid: np.ndarray, t_grid: np.ndarray,
    term_specs: Sequence[TermSpec],
    crop: int = 3,
    smoothing_sigma: float = 0.0,
    interp_method: str = "linear",
    sindy_kwargs: Dict = None,
) -> Dict:
    """Run the full discovery pipeline starting from scattered (sparse
    and/or noisy) observations: reconstruct -> (optionally smooth) ->
    finite-difference -> library -> STLSQ."""
    U_grid, _, _ = reconstruct_grid(x, t, u, x_grid, t_grid, method=interp_method)
    U_grid = smooth_grid(U_grid, sigma=smoothing_sigma)
    return discover_from_grid(U_grid, x_grid, t_grid, term_specs, crop=crop, sindy_kwargs=sindy_kwargs)


def equation_to_string(result: Dict, lhs: str = "u_t") -> str:
    if not result["terms"]:
        return f"{lhs} = 0"
    pieces = []
    for term in result["terms"]:
        c = term["coefficient"]
        sign = "+" if c >= 0 else "-"
        pieces.append(f"{sign} {abs(c):.4g}*{term['term']}")
    body = " ".join(pieces)
    if body.startswith("+ "):
        body = body[2:]
    return f"{lhs} = {body}"


def evaluate_against_ground_truth(
    result: Dict,
    true_term: str = "u_xx",
    coefficient_true: float = 0.1,
    extra_term_tol: float = 1e-2,
) -> Dict:
    """
    Compares a discovered equation against a known hidden ground truth.
    Ground truth is used HERE ONLY, for grading -- never inside the
    discovery pipeline itself.

    structure_recovered = True iff:
      (a) the true term has a non-negligible coefficient, AND
      (b) every other candidate term's coefficient is below extra_term_tol.
    This directly implements "we care about whether the correct physical
    term was recovered", not just coefficient closeness.
    """
    coeffs = result["all_coefficients"]
    coefficient_pred = coeffs.get(true_term, 0.0)

    extra_terms = {
        name: c for name, c in coeffs.items()
        if name != true_term and abs(c) > extra_term_tol
    }
    structure_recovered = abs(coefficient_pred) > extra_term_tol and len(extra_terms) == 0

    rel_error = (
        abs(coefficient_pred - coefficient_true) / abs(coefficient_true)
        if coefficient_true != 0 else float("nan")
    )

    return {
        "true_term": true_term,
        "coefficient_true": coefficient_true,
        "coefficient_pred": coefficient_pred,
        "relative_parameter_error": rel_error,
        "structure_recovered": bool(structure_recovered),
        "extra_terms": extra_terms,
    }


def evaluate_competing_hypotheses(
    flat_fields: Dict[str, np.ndarray],
    y: np.ndarray,
    hypotheses: Dict[str, List[str]],
) -> Dict:
    """
    Step 9: score a fixed set of NAMED hypotheses (each a small list of
    single field names, e.g. {"H1_diffusion": ["u_xx"], "H2_advection":
    ["u_x"]}) against the data via unrestricted (non-sparse) least
    squares, and report residual / R^2 / fitted coefficient for each.

    This is EVIDENCE for the later PINN/validation stage to use -- this
    function does not itself declare a hypothesis "valid" or "invalid".
    """
    out = {}
    for name, term_names in hypotheses.items():
        cols = [flat_fields[t] for t in term_names]
        Theta_subset = np.column_stack(cols)
        fit = fit_restricted(Theta_subset, y)
        out[name] = {
            "terms": term_names,
            "coefficients": dict(zip(term_names, fit["coefficients"])),
            "r2": fit["r2"],
            "ss_res": fit["ss_res"],
        }
    return out
