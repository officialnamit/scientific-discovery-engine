"""
Phase 3, Part 6: forward PINN.

Physics parameters (e.g. D) are KNOWN and FIXED -- not trained. The
network learns u_theta(x,t) to jointly satisfy the (noisy, sparse)
observations and the known governing equation.
"""

from typing import Dict

from pinn.residuals import RESIDUAL_REGISTRY
from pinn.trainer import build_and_train


def run_forward_pinn(
    hypothesis_name: str,
    param_values: Dict[str, float],
    architecture_cfg: Dict,
    weights: Dict[str, float],
    data, collocation, bc, ic,
    training_cfg: Dict,
    resample_collocation_fn=None,
    seed: int = 0,
):
    hyp_cfg = RESIDUAL_REGISTRY[hypothesis_name]
    trainer, history = build_and_train(
        hyp_cfg, param_values, trainable=False,
        architecture_cfg=architecture_cfg, weights=weights,
        data=data, collocation=collocation, bc=bc, ic=ic,
        training_cfg=training_cfg,
        resample_collocation_fn=resample_collocation_fn, seed=seed,
    )
    return trainer, history
