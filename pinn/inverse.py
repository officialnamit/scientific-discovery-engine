"""
Phase 3, Part 7: inverse PINN.

Physics parameters (e.g. D) are UNKNOWN and TRAINABLE -- optimized
jointly with the network weights. Initialize away from the true value
(the caller should NOT pass the true D here) so convergence to the
right value is actually demonstrated, not assumed.
"""

from typing import Dict

from pinn.residuals import RESIDUAL_REGISTRY
from pinn.trainer import build_and_train


def run_inverse_pinn(
    hypothesis_name: str,
    param_init: Dict[str, float],
    architecture_cfg: Dict,
    weights: Dict[str, float],
    data, collocation, bc, ic,
    training_cfg: Dict,
    resample_collocation_fn=None,
    seed: int = 0,
):
    hyp_cfg = RESIDUAL_REGISTRY[hypothesis_name]
    trainer, history = build_and_train(
        hyp_cfg, param_init, trainable=True,
        architecture_cfg=architecture_cfg, weights=weights,
        data=data, collocation=collocation, bc=bc, ic=ic,
        training_cfg=training_cfg,
        resample_collocation_fn=resample_collocation_fn, seed=seed,
    )
    return trainer, history
