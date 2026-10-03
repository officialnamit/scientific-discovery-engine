"""
Phase 4: generic equation -> PINN residual compiler.

This is what makes the architecture PDE-agnostic in fact, not just in
principle: rather than hand-adding an entry to pinn/residuals.py's
RESIDUAL_REGISTRY for every hypothesis the LLM might propose, this
module PARSES the hypothesis's own equation string (sympy) and
compiles it directly into a residual_fn with the exact interface
pinn/trainer.py already expects: residual_fn(derivs, params).

Supported field symbols: u, u_t, u_x, u_xx, u_xxx (matching what
pinn/derivatives.py actually computes). Parameters are whatever names
the hypothesis declares (D, c, r, K, ...).

This covers every example in the Phase 4 spec without modification:
    diffusion:            u_t = D*u_xx
    advection:             u_t + c*u_x = 0
    advection-diffusion:    u_t + c*u_x = D*u_xx
    reaction-diffusion:      u_t = D*u_xx + r*u*(1-u/K)
and anything else expressible as an algebraic combination of
{u, u_t, u_x, u_xx, u_xxx} and named parameters -- no code change
needed for a new equation, only a new hypothesis object.

LIMITATION (stated honestly): this compiles ALGEBRAIC PDEs in these
five symbols. It does not handle integral/non-local terms, terms in
u_xxxx or higher, or systems of multiple dependent variables. That
covers 1D scalar-field transport/reaction equations, which is the
scope of this project so far.
"""

from typing import Dict, List, Tuple

import sympy
import torch
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

FIELD_SYMBOLS = ["u", "u_t", "u_x", "u_xx", "u_xxx"]

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
_TORCH_FUNCTIONS = {
    "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
    "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt, "tanh": torch.tanh,
    "Abs": torch.abs,
}


class EquationParseError(ValueError):
    pass


def parse_equation(equation_str: str, parameter_names: List[str]):
    """Parse 'lhs = rhs' into a sympy residual expression (lhs - rhs),
    using ONLY the fixed field symbols plus the hypothesis's own
    declared parameter names -- nothing else is in scope, so a typo'd
    or undeclared symbol raises a clear parse error rather than
    silently being treated as a new free variable.
    """
    if "=" not in equation_str:
        raise EquationParseError(f"Equation must contain '=': {equation_str!r}")
    lhs_str, rhs_str = equation_str.split("=", 1)

    local_dict = {name: sympy.Symbol(name) for name in FIELD_SYMBOLS + list(parameter_names)}
    try:
        lhs = parse_expr(lhs_str.strip(), local_dict=local_dict, transformations=_TRANSFORMATIONS)
        rhs = parse_expr(rhs_str.strip(), local_dict=local_dict, transformations=_TRANSFORMATIONS)
    except (sympy.SympifyError, SyntaxError, TypeError) as e:
        raise EquationParseError(f"Could not parse equation {equation_str!r}: {e}") from e

    residual_expr = sympy.expand(lhs - rhs)

    free_names = {s.name for s in residual_expr.free_symbols}
    unknown = free_names - set(FIELD_SYMBOLS) - set(parameter_names)
    if unknown:
        raise EquationParseError(
            f"Equation {equation_str!r} references undeclared symbol(s) {sorted(unknown)}. "
            f"Declared parameters: {parameter_names}, known fields: {FIELD_SYMBOLS}."
        )

    used_fields = [f for f in FIELD_SYMBOLS if f in free_names]
    used_params = [p for p in parameter_names if p in free_names]
    return residual_expr, used_fields, used_params, local_dict


def compile_residual(equation_str: str, parameter_names: List[str]):
    """
    Returns (residual_fn, used_fields, used_params, residual_expr).

    residual_fn(derivs: Dict[str, Tensor], params: Dict[str, Tensor]) -> Tensor
    matches the exact interface pinn/trainer.py calls for every
    hard-coded RESIDUAL_REGISTRY entry -- a compiled hypothesis is a
    drop-in replacement, no trainer changes needed.
    """
    residual_expr, used_fields, used_params, local_dict = parse_equation(equation_str, parameter_names)

    field_syms = [local_dict[f] for f in FIELD_SYMBOLS]
    param_syms = [local_dict[p] for p in parameter_names]

    numeric_fn = sympy.lambdify(
        field_syms + param_syms, residual_expr, modules=[_TORCH_FUNCTIONS, "numpy"]
    )

    def residual_fn(derivs: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]) -> torch.Tensor:
        zeros = torch.zeros_like(derivs["u_t"])
        field_vals = [derivs.get(f, zeros) for f in FIELD_SYMBOLS]
        param_vals = [params[p] for p in parameter_names]
        return numeric_fn(*field_vals, *param_vals)

    return residual_fn, used_fields, used_params, residual_expr


def equation_str_to_sympy_str(equation_str: str, parameter_names: List[str]) -> str:
    """Human-readable normalized form, e.g. for logging/reports."""
    residual_expr, _, _, _ = parse_equation(equation_str, parameter_names)
    return f"{sympy.sstr(residual_expr)} = 0"
