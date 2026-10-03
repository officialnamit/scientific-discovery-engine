"""
Phase 4: deterministic (non-LLM) validation of generated hypotheses.

These checks run BEFORE anything is handed to PINN validation. None of
them decide scientific correctness -- they only catch malformed,
inconsistent, or duplicate hypotheses so the (expensive) PINN stage
never wastes time on something that can't even be compiled or is a
structural repeat of a hypothesis already in the set.
"""

from typing import Dict, List, Tuple

import sympy

from hypotheses.equation_compiler import EquationParseError, parse_equation
from llm.schemas import Hypothesis


def validate_hypothesis(hyp: Hypothesis) -> Dict:
    """Runs checks 1-5 from the Phase 4 spec on a single hypothesis.
    Returns a dict; 'valid' is False only for hard errors (unparseable
    equation, undeclared symbols, missing 'u'). Everything else is a
    warning -- informative, non-fatal."""
    errors: List[str] = []
    warnings_list: List[str] = []
    param_names = [p.name for p in hyp.parameters]

    # 1. Syntax validation + (side effect) 4. auto-detected derivatives
    try:
        _, used_fields, used_params, _ = parse_equation(hyp.equation, param_names)
    except EquationParseError as e:
        return {
            "id": hyp.id, "valid": False, "errors": [str(e)], "warnings": [],
            "used_fields": [], "used_params": [], "dimensional_completeness": 0.0,
        }

    # 2. Variable validation
    if "u" not in hyp.dependent_variables:
        errors.append("dependent_variables does not include 'u'")
    for v in ("x", "t"):
        if v not in hyp.independent_variables:
            warnings_list.append(f"independent_variables missing '{v}'")

    # 3. Parameter validation
    declared = set(param_names)
    used = set(used_params)
    undeclared_used = used - declared  # parse_equation already forbids this, kept for defense-in-depth
    unused_declared = declared - used
    if undeclared_used:
        errors.append(f"Equation uses undeclared parameter(s): {sorted(undeclared_used)}")
    if unused_declared:
        warnings_list.append(f"Declared parameter(s) never appear in the equation: {sorted(unused_declared)}")

    # 4. Derivative validation: LLM-STATED required_derivatives vs. what sympy actually finds
    stated = set(hyp.required_derivatives)
    actual_derivs = {f for f in used_fields if f != "u"}
    if stated != actual_derivs:
        missing = actual_derivs - stated
        extra = stated - actual_derivs
        parts = []
        if missing:
            parts.append(f"equation uses {sorted(missing)} but they're not listed in required_derivatives")
        if extra:
            parts.append(f"required_derivatives lists {sorted(extra)} but the equation doesn't use them")
        warnings_list.append("Derivative mismatch: " + "; ".join(parts))

    # 5. Dimensional/unit metadata -- honestly partial, never claimed complete
    n_params = len(hyp.parameters)
    n_with_units = sum(1 for p in hyp.parameters if p.units)
    dimensional_completeness = (n_with_units / n_params) if n_params > 0 else 1.0
    if dimensional_completeness < 1.0:
        warnings_list.append(
            f"Dimensional metadata incomplete: {n_with_units}/{n_params} parameters have units specified"
        )

    return {
        "id": hyp.id,
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings_list,
        "used_fields": used_fields,
        "used_params": used_params,
        "dimensional_completeness": dimensional_completeness,
    }


def canonical_signature(hyp: Hypothesis) -> str:
    """
    6. Duplicate detection support: a structural signature that is
    independent of (a) parameter NAMES (D vs k) and (b) which side of
    the equation terms were written on / overall sign (u_t=D*u_xx is
    the same statement as -u_t+D*u_xx=0). Two hypotheses sharing a
    signature are structural duplicates regardless of surface wording.
    """
    param_names = [p.name for p in hyp.parameters]
    try:
        residual_expr, _, used_params, local_dict = parse_equation(hyp.equation, param_names)
    except EquationParseError:
        return f"UNPARSEABLE::{hyp.equation.strip()}"

    subs = {local_dict[name]: sympy.Symbol(f"_p{i}") for i, name in enumerate(sorted(used_params))}
    generic_expr = sympy.expand(residual_expr.subs(subs))
    s_pos = sympy.sstr(generic_expr)
    s_neg = sympy.sstr(sympy.expand(-generic_expr))
    return min(s_pos, s_neg)  # sign-invariant canonical form


def deduplicate(hypotheses: List[Hypothesis]) -> Tuple[List[Hypothesis], List[Dict]]:
    """Returns (unique_hypotheses, removed_records). A later hypothesis
    with the same structural signature as an earlier one is removed;
    order of first appearance is preserved."""
    seen: Dict[str, str] = {}
    unique: List[Hypothesis] = []
    removed: List[Dict] = []
    for h in hypotheses:
        sig = canonical_signature(h)
        if sig in seen:
            removed.append({"id": h.id, "name": h.name, "duplicate_of": seen[sig], "signature": sig})
        else:
            seen[sig] = h.id
            unique.append(h)
    return unique, removed
