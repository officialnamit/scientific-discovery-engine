"""
Phase 6: interpretable equation complexity.

Every component is a plain count read off the parsed (sympy-expanded)
equation, so a reviewer can verify the score by hand:

    n_terms               additive terms on the right-hand side (u_t excluded)
    n_free_parameters     trainable parameters actually used in the equation
    max_derivative_order  highest spatial derivative order (u_x=1, u_xx=2, u_xxx=3)
    n_nonlinear_terms     terms whose field part has total degree > 1 (e.g. u**2, u*u_x)
    expression_ops        sympy.count_ops of the RHS (overall expression size)

score = n_terms + n_free_parameters + max_derivative_order + 2*n_nonlinear_terms
        + 0.1*expression_ops

Weights are a stated, fixed choice (nonlinear terms counted double because
they change the qualitative behavior of an equation more than an extra
linear term). The score is only ever used to ORDER candidates that fit
the data equivalently well -- it never overrides fit quality.

Example: u_t = D*u_xx scores far below u_t = D*u_xx + r*u*(1-u/K).
"""

from typing import Dict

import sympy

from hypotheses.equation_compiler import FIELD_SYMBOLS, parse_equation
from llm.schemas import Hypothesis

_ORDER = {"u": 0, "u_t": 0, "u_x": 1, "u_xx": 2, "u_xxx": 3}


def equation_complexity(hyp: Hypothesis) -> Dict:
    param_names = [p.name for p in hyp.parameters]
    residual_expr, used_fields, used_params, local_dict = parse_equation(hyp.equation, param_names)
    u_t = local_dict["u_t"]
    expanded = sympy.expand(residual_expr)
    rhs = sympy.expand(-(expanded - expanded.coeff(u_t) * u_t))  # u_t = rhs (for unit u_t coefficient)

    field_syms = [local_dict[f] for f in FIELD_SYMBOLS if f != "u_t"]
    terms = sympy.Add.make_args(rhs) if rhs != 0 else ()
    n_nonlinear = 0
    for term in terms:
        _, field_part = term.as_independent(*field_syms, as_Add=False)
        degree = sum(sympy.degree(field_part, s) for s in field_syms if field_part.has(s))
        if degree > 1:
            n_nonlinear += 1

    trainable_used = [p.name for p in hyp.parameters if p.trainable and p.name in used_params]
    max_order = max([_ORDER[f] for f in used_fields if f != "u_t"] + [0])
    ops = int(sympy.count_ops(rhs))

    score = len(terms) + len(trainable_used) + max_order + 2 * n_nonlinear + 0.1 * ops
    return {
        "n_terms": len(terms),
        "n_free_parameters": len(trainable_used),
        "max_derivative_order": max_order,
        "n_nonlinear_terms": n_nonlinear,
        "expression_ops": ops,
        "complexity_score": round(score, 3),
        "rhs_expanded": str(rhs),
    }
