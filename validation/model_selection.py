"""
Phase 6: model selection among PINN-validated candidates.

Answers three DIFFERENT questions, deliberately kept separate:

  "can it explain the data?"         -> the gate (PINN validation on held-out observations)
  "does the data REQUIRE each term?" -> term necessity (per-parameter contribution)
  "which is the most parsimonious
   supported explanation?"           -> fit-equivalence set + complexity ordering (+ BIC cross-check)

None of these is a truth score. The selected candidate is the simplest
one the data cannot distinguish from the best-fitting one -- an Occam's
razor rule applied to measured quantities, nothing more.

Everything here is computed WITHOUT the synthetic ground truth: fit
quality is measured on held-out NOISY observations, never the clean
oracle field.
"""

import math
from typing import Dict, List

import numpy as np
import sympy
import torch

from hypotheses.equation_compiler import FIELD_SYMBOLS, parse_equation
from hypotheses.validation import canonical_signature
from llm.schemas import Hypothesis
from pinn.derivatives import compute_derivatives


def parameter_contributions(model, hyp: Hypothesis, params: Dict[str, float],
                            x_range, t_range, n: int = 50) -> Dict[str, float]:
    """For each parameter p: RMS over a fresh grid of the SUM of all RHS terms
    containing p, divided by RMS(u_t). Grouping by parameter (not by single
    term) matters: r*u and -r*u**2/K can individually be sizable yet cancel;
    what the data tests is the whole mechanism r*u*(1-u/K)."""
    param_names = [p.name for p in hyp.parameters]
    residual_expr, _, used_params, local_dict = parse_equation(hyp.equation, param_names)
    u_t = local_dict["u_t"]
    expanded = sympy.expand(residual_expr)
    rhs = sympy.expand(-(expanded - expanded.coeff(u_t) * u_t))

    xs = np.linspace(*x_range, n)
    ts = np.linspace(*t_range, n)
    X, T = np.meshgrid(xs, ts)
    x = torch.tensor(X.ravel(), dtype=torch.float32).reshape(-1, 1)
    t = torch.tensor(T.ravel(), dtype=torch.float32).reshape(-1, 1)
    d = compute_derivatives(model, x, t)
    fields = {k: v.detach().numpy().ravel() for k, v in d.items()}
    ut_rms = float(np.sqrt(np.mean(fields["u_t"] ** 2))) or 1e-12

    field_syms = [local_dict[f] for f in FIELD_SYMBOLS]
    contributions = {}
    for pname in used_params:
        psym = local_dict[pname]
        group = sympy.Add(*[term for term in sympy.Add.make_args(rhs) if term.has(psym)])
        f = sympy.lambdify(field_syms + [local_dict[q] for q in used_params], group, "numpy")
        vals = f(*[fields[s] for s in FIELD_SYMBOLS], *[params[q] for q in used_params])
        vals = np.broadcast_to(np.asarray(vals, dtype=float), fields["u"].shape)
        contributions[pname] = float(np.sqrt(np.mean(vals ** 2)) / ut_rms)
    return contributions


def reduced_signature(hyp: Hypothesis, inactive_params: List[str]) -> str:
    """Canonical signature of the equation with every term containing an
    inactive parameter removed -- used to say 'H3 effectively reduces to H1'."""
    param_names = [p.name for p in hyp.parameters]
    residual_expr, _, _, local_dict = parse_equation(hyp.equation, param_names)
    expanded = sympy.expand(residual_expr)
    kept = [t for t in sympy.Add.make_args(expanded)
            if not any(t.has(local_dict[p]) for p in inactive_params)]
    reduced_expr = sympy.Add(*kept)
    remaining = [p for p in hyp.parameters if p.name not in inactive_params]
    reduced_eq = f"{sympy.sstr(reduced_expr)} = 0"
    tmp = Hypothesis(id="_reduced", name="_reduced", equation=reduced_eq,
                     parameters=[p.model_dump() for p in remaining])
    return canonical_signature(tmp), reduced_eq


def bic(mse: float, n: int, k: int) -> float:
    """Gaussian-likelihood BIC on n held-out-style observations with k fitted
    physics parameters (network weights are NOT counted: every candidate uses
    the identical network, so they cancel in comparisons)."""
    return n * math.log(max(mse, 1e-300)) + k * math.log(n)


def select_model(candidates: List[Dict], equivalence_tol: float, inactive_threshold: float,
                 prefer_reduced: bool = False) -> Dict:
    """
    candidates: list of dicts with keys id, gate_passed, val_rmse, complexity_score,
    bic, contributions, signature, reduced_signature (optional).

    Rule (stated, fixed before looking at results):
      1. Keep candidates that passed the gate (structural + physics + PINN-on-held-out-data).
      2. Fit-equivalence set: val_rmse <= (1 + equivalence_tol) * best val_rmse.
         Differences inside this band are within what measurement noise can resolve.
      3. Select the lowest complexity_score in the set; ties -> lower BIC.
      4. (prefer_reduced=True, added in Phase 7) If the selected candidate has parameters
         whose terms contribute < inactive_threshold of RMS(u_t), and the equation left after
         removing those terms is itself a SUPPORTED candidate, select that simpler candidate.
         Reason: Phase 7 found that on noise-free data the "within tol of best error" test is
         decided by discretization error, which extra terms can absorb (see README Phase 7).
         Default False so Phase 6 results are reproduced exactly.
    """
    supported = [c for c in candidates if c["gate_passed"]]
    if not supported:
        return {"selected": None, "reason": "No candidate passed validation.", "equivalence_set": []}

    best_rmse = min(c["val_rmse"] for c in supported)
    cutoff = (1.0 + equivalence_tol) * best_rmse
    equivalent = [c for c in supported if c["val_rmse"] <= cutoff]
    equivalent.sort(key=lambda c: (c["complexity_score"], c["bic"]))
    selected = equivalent[0]
    reduction = None
    if prefer_reduced:
        for _ in range(len(supported)):
            inactive = [p for p, v in selected["contributions"].items() if v < inactive_threshold]
            target = selected.get("reduced_signature")
            match = next((c for c in supported if inactive and target and c["signature"] == target
                          and c["id"] != selected["id"]), None)
            if match is None:
                break
            reduction = {"from": selected["id"], "to": match["id"], "inactive_parameters": inactive}
            selected = match

    by_bic = min(supported, key=lambda c: c["bic"])
    explanations = []
    for c in supported:
        inactive = [p for p, v in c["contributions"].items() if v < inactive_threshold]
        note = {"id": c["id"], "inactive_parameters": inactive}
        if c["id"] == selected["id"]:
            note["role"] = "selected"
        elif c in equivalent:
            note["role"] = "fits equivalently but is more complex"
        else:
            note["role"] = "supported but fits held-out data measurably worse"
        if inactive and c.get("reduced_signature") == selected["signature"]:
            note["reduces_to_selected"] = True
        explanations.append(note)

    return {
        "selected": selected["id"],
        "best_val_rmse": best_rmse,
        "equivalence_cutoff": cutoff,
        "equivalence_set": [c["id"] for c in equivalent],
        "bic_minimizer": by_bic["id"],
        "bic_agrees_with_selection": by_bic["id"] == selected["id"],
        "explanations": explanations,
        "rule": (f"gate -> val_rmse within {equivalence_tol*100:.0f}% of best -> lowest complexity -> lowest BIC"
                 + (" -> replace by reduced form if extra terms are inactive" if prefer_reduced else "")),
        "reduction_applied": reduction,
    }
