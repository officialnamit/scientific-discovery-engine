"""
Phase 4 -> Phase 3 adapter.

This is the ONLY place that translates a Hypothesis object into
something pinn/trainer.py can run. Nothing in pinn/ or validation/ was
modified to know about Hypothesis objects (aside from the earlier,
general-purpose make_params per-parameter-trainable extension) -- this
keeps Phase 3 a reusable, hypothesis-agnostic engine, per the "Do not
duplicate the PINN implementation" / "implement a clean adapter rather
than modifying the PINN architecture unnecessarily" constraints.
"""

from typing import Dict

from hypotheses.equation_compiler import compile_residual
from llm.schemas import Hypothesis


def hypothesis_to_pinn_config(hyp: Hypothesis) -> Dict:
    """
    Returns a dict with everything pinn.trainer.build_and_train needs:
        {
          "hypothesis_cfg": {"fn": residual_fn, "param_names": [...], "equation_str": ...},
          "param_init":     {name: initial_value, ...},
          "trainable":      {name: bool, ...},
        }
    This has the exact shape of a RESIDUAL_REGISTRY entry
    (pinn/residuals.py) plus a param_init/trainable pair -- so it can
    be passed anywhere a hand-written hypothesis_cfg from the registry
    is accepted (see pinn/forward.py, pinn/inverse.py, pinn/trainer.py:build_and_train).
    """
    param_names = [p.name for p in hyp.parameters]
    residual_fn, used_fields, used_params, residual_expr = compile_residual(hyp.equation, param_names)

    hypothesis_cfg = {
        "fn": residual_fn,
        "param_names": param_names,
        "equation_str": hyp.equation,
    }
    param_init = {
        p.name: (p.initial_value if p.initial_value is not None else 1.0)
        for p in hyp.parameters
    }
    trainable = {p.name: p.trainable for p in hyp.parameters}

    return {
        "hypothesis_id": hyp.id,
        "hypothesis_name": hyp.name,
        "hypothesis_cfg": hypothesis_cfg,
        "param_init": param_init,
        "trainable": trainable,
        "used_fields": used_fields,
        "used_params": used_params,
        "residual_expr_str": str(residual_expr),
    }
