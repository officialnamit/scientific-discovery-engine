"""
Phase 6: physical constraints on candidate equations.

Two kinds, applied at two different points in the search funnel:

1. PRE-PINN well-posedness checks (cheap, before any training):
   - the candidate must be an evolution equation in time (contain u_t
     with a nonzero constant coefficient), because the problem supplies
     an initial-value + boundary-value setup; an equation without u_t
     cannot be integrated forward from the given initial condition.
   - declared bounds must be self-consistent and the initial value
     must lie inside them.
   - boundary-condition compatibility is reported as a WARNING, not a
     rejection: e.g. a purely first-order-in-x equation (advection)
     with Dirichlet conditions imposed at BOTH ends is over-determined
     for a hyperbolic PDE (only the inflow boundary can be prescribed).
     The PINN will still try to satisfy both; this is recorded so the
     report can explain a resulting misfit physically, not just
     numerically.

2. POST-FIT parameter bounds: after the inverse PINN estimates the
   parameters, each estimate is checked against its bound (e.g. D > 0:
   a negative diffusivity is the ill-posed backward heat equation).
   A violation rejects the candidate regardless of how well it fit.

Bounds come from the hypothesis's own ParameterSpec (lower_bound /
upper_bound) first, then from config.yaml's phase6.parameter_constraints
keyed by parameter NAME as a fallback (a convention of this corpus:
D = diffusivity, K = carrying capacity). Parameters with no declared
bound are unconstrained -- no blanket constraint is applied.
"""

from typing import Dict, List, Optional

import sympy

from hypotheses.equation_compiler import EquationParseError, parse_equation
from llm.schemas import Hypothesis


def resolve_bounds(hyp: Hypothesis, fallback: Optional[Dict] = None) -> Dict[str, Dict]:
    fallback = fallback or {}
    out = {}
    for p in hyp.parameters:
        fb = fallback.get(p.name, {}) or {}
        lower = p.lower_bound if p.lower_bound is not None else fb.get("lower")
        upper = p.upper_bound if p.upper_bound is not None else fb.get("upper")
        source = "hypothesis" if (p.lower_bound is not None or p.upper_bound is not None) else (
            "config_fallback" if fb else "unconstrained")
        out[p.name] = {"lower": lower, "upper": upper, "source": source}
    return out


def precheck_physics(hyp: Hypothesis, problem_spec: Dict, fallback_bounds: Optional[Dict] = None) -> Dict:
    """problem_spec: {"has_ic": bool, "dirichlet_both_ends": bool} -- describes the
    conditions the OBSERVATIONAL SETUP provides (not anything about the answer)."""
    errors: List[str] = []
    warnings: List[str] = []
    param_names = [p.name for p in hyp.parameters]

    try:
        residual_expr, used_fields, _, local_dict = parse_equation(hyp.equation, param_names)
    except EquationParseError as e:
        return {"id": hyp.id, "passed": False, "errors": [str(e)], "warnings": []}

    u_t = local_dict["u_t"]
    coeff_ut = sympy.expand(residual_expr).coeff(u_t)
    if coeff_ut == 0:
        errors.append("Not an evolution equation: no u_t term, cannot be integrated from the given initial condition.")
    elif coeff_ut.free_symbols:
        errors.append(f"u_t coefficient '{coeff_ut}' is not constant (depends on {sorted(map(str, coeff_ut.free_symbols))}); "
                      "not a standard explicit evolution equation.")
    # any nonlinearity in u_t itself (u_t**2 etc.)
    if sympy.degree(sympy.expand(residual_expr), u_t) > 1:
        errors.append("Equation is nonlinear in u_t; not supported as an explicit evolution equation.")

    bounds = resolve_bounds(hyp, fallback_bounds)
    for p in hyp.parameters:
        b = bounds[p.name]
        lo, hi = b["lower"], b["upper"]
        if lo is not None and hi is not None and lo >= hi:
            errors.append(f"Parameter {p.name}: inconsistent bounds ({lo}, {hi}).")
        if p.initial_value is not None:
            if lo is not None and not p.initial_value > lo:
                errors.append(f"Parameter {p.name}: initial value {p.initial_value} violates bound > {lo}.")
            if hi is not None and not p.initial_value < hi:
                errors.append(f"Parameter {p.name}: initial value {p.initial_value} violates bound < {hi}.")

    max_x_order = max([{"u_x": 1, "u_xx": 2, "u_xxx": 3}.get(f, 0) for f in used_fields] + [0])
    if problem_spec.get("dirichlet_both_ends"):
        if max_x_order == 0:
            warnings.append("No spatial derivative: boundary conditions are not coupled to the dynamics "
                            "(each point evolves independently); BCs can hold only if the dynamics preserve them.")
        elif max_x_order == 1:
            warnings.append("First order in x with Dirichlet data at BOTH ends: over-determined for a "
                            "hyperbolic equation (only the inflow boundary can be prescribed).")
    if problem_spec.get("has_ic") and not hyp.required_initial_conditions:
        warnings.append("Problem provides an initial condition but hypothesis declares none required.")

    return {"id": hyp.id, "passed": len(errors) == 0, "errors": errors, "warnings": warnings,
            "bounds": bounds, "max_spatial_derivative_order": max_x_order}


def check_fitted_bounds(param_estimate: Dict[str, float], bounds: Dict[str, Dict]) -> Dict:
    violations = []
    for name, value in param_estimate.items():
        b = bounds.get(name, {})
        lo, hi = b.get("lower"), b.get("upper")
        if lo is not None and not value > lo:
            violations.append(f"{name}={value:.4g} violates physical bound {name} > {lo}")
        if hi is not None and not value < hi:
            violations.append(f"{name}={value:.4g} violates physical bound {name} < {hi}")
    return {"passed": len(violations) == 0, "violations": violations}
