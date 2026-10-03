"""
Phase 7: mechanism identification utilities (PDE-agnostic).

- term_set(hyp): the set of field monomials on an equation's RHS, e.g.
  reaction-diffusion -> {u_xx, u, u**2}. Two equations with the same term set
  are the same mechanism family regardless of parametrization.
- discovered_to_terms(sindy_result): the same term set for a SINDy result.
- match_mechanism(): discovered term set -> the KG template with that term set.
- recover_parameters(): solve the template's physical parameters from fitted
  monomial coefficients symbolically (e.g. K = -r / coef(u**2)).
- fit_candidate_on_grid(): cheap regression "fitting engine" (OLS of u_t on a
  candidate's own monomials, scored on held-out grid points). Used where a
  PINN per item is unaffordable; the PINN path is exercised separately by
  experiments/10_generalization.py.
"""

from typing import Dict, List, Optional

import numpy as np
import sympy

from equation_discovery.derivatives import finite_diff_derivatives
from hypotheses.equation_compiler import FIELD_SYMBOLS, parse_equation
from llm.schemas import Hypothesis
from validation.stability import candidate_monomials

_LIBRARY_NAME_TO_EXPR = None


def _syms():
    return {f: sympy.Symbol(f) for f in FIELD_SYMBOLS}


def term_set(hyp: Hypothesis) -> frozenset:
    monos, _ = candidate_monomials(hyp)
    return frozenset(str(m) for m in monos)


def _label_to_expr(label: str):
    """Phase-2 library labels ('u', 'u_xx', 'u^2', 'u*u_x', '1') -> sympy expr string."""
    s = _syms()
    if label == "1":
        return sympy.Integer(1)
    expr = sympy.Integer(1)
    for part in label.split("*"):
        if "^" in part:
            name, power = part.split("^")
            expr *= s[name] ** int(power)
        else:
            expr *= s[part]
    return expr


def discovered_coefficients(sindy_result: Dict, tol: float = 1e-8) -> Dict[str, float]:
    """SINDy result -> {monomial_str: coefficient} for nonzero terms."""
    return {str(_label_to_expr(t["term"])): t["coefficient"]
            for t in sindy_result["terms"] if abs(t["coefficient"]) > tol}


def match_mechanism(coeffs: Dict[str, float], templates: List[Hypothesis]) -> Optional[Hypothesis]:
    target = frozenset(coeffs)
    for h in templates:
        if term_set(h) == target:
            return h
    return None


def recover_parameters(template: Hypothesis, coeffs: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Equate the template's RHS monomial coefficients (expressions in its
    parameters) to the fitted numbers and solve. Returns None if unsolvable."""
    names = [p.name for p in template.parameters]
    residual, _, _, local = parse_equation(template.equation, names)
    u_t = local["u_t"]
    expanded = sympy.expand(residual)
    rhs = sympy.expand(-(expanded - expanded.coeff(u_t) * u_t) / expanded.coeff(u_t))
    field_syms = [local[f] for f in FIELD_SYMBOLS if f != "u_t"]
    eqs = []
    for term in sympy.Add.make_args(rhs):
        coef_part, field_part = term.as_independent(*field_syms, as_Add=False)
        if str(field_part) in coeffs:
            eqs.append(sympy.Eq(coef_part, coeffs[str(field_part)]))
    params = [local[n] for n in names]
    try:
        sol = sympy.solve(eqs, params, dict=True)
    except Exception:
        return None
    if not sol:
        return None
    return {str(k): float(v) for k, v in sol[0].items() if v.is_number}


def grid_fields(U, x, t, crop: int = 3) -> Dict[str, np.ndarray]:
    f = finite_diff_derivatives(U, x, t, crop=crop)
    return {k: v.ravel() for k, v in f.items() if k not in ("x", "t")}


def fit_candidate_on_grid(hyp: Hypothesis, fields: Dict[str, np.ndarray], val_mask: np.ndarray) -> Dict:
    """OLS of u_t on the candidate's own monomials using train points; RMSE of
    predicted u_t on held-out val points. Recovers physical parameters from the
    fitted coefficients via recover_parameters()."""
    monos, field_syms = candidate_monomials(hyp)
    y = fields["u_t"]
    cols = [np.broadcast_to(np.asarray(sympy.lambdify(field_syms, m, "numpy")(
        *[fields[s.name] for s in field_syms]), dtype=float), y.shape) for m in monos]
    Theta = np.column_stack(cols)
    tr = ~val_mask
    xi, *_ = np.linalg.lstsq(Theta[tr], y[tr], rcond=None)
    val_rmse = float(np.sqrt(np.mean((Theta[val_mask] @ xi - y[val_mask]) ** 2)))
    coeffs = {str(m): float(c) for m, c in zip(monos, xi)}
    return {"coefficients": coeffs, "val_rmse": val_rmse,
            "params": recover_parameters(hyp, coeffs) or {}}


def library_collinearity(fields: Dict[str, np.ndarray], names=("u", "u_x", "u_xx")) -> Dict:
    """Identifiability diagnostic: max |correlation| between candidate-library
    columns. ~1.0 means the data cannot structurally distinguish those terms
    (e.g. the Phase 1 single-sine eigenfunction trap: u_xx = -pi^2 u)."""
    best = (0.0, None)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = fields[a], fields[b]
            if np.std(va) == 0 or np.std(vb) == 0:
                continue
            r = abs(float(np.corrcoef(va, vb)[0, 1]))
            if r > best[0]:
                best = (r, (a, b))
    return {"max_abs_correlation": best[0], "pair": best[1], "identifiable": best[0] < 0.99}
