"""
Phase 3, Part 11: validation result object.

Status is decided PURELY from numeric thresholds on measured metrics --
no LLM, no human judgement call embedded here. Thresholds are a config
choice (see config.yaml: pinn.validation.thresholds), fixed BEFORE
looking at results for a given run, not tuned after the fact to force
an outcome.

Terminology deliberately avoids overclaiming: "supported by the
available observations and physics constraints" / "rejected under the
tested conditions" -- never "proven".
"""

from typing import Dict, Optional

SUPPORTED = "supported by the available observations and physics constraints"
REJECTED = "rejected under the tested conditions"


def build_validation_result(
    hypothesis_name: str,
    equation_str: str,
    data_loss: float,
    physics_residual: float,
    bc_loss: float,
    ic_loss: float,
    prediction_error: float,
    thresholds: Dict[str, float],
    parameter_estimate: Optional[Dict[str, float]] = None,
    parameter_true: Optional[Dict[str, float]] = None,
    ood_error: Optional[float] = None,
) -> Dict:
    checks = {
        "data_loss": data_loss <= thresholds["data_loss_max"],
        "physics_residual": physics_residual <= thresholds["physics_residual_max"],
        "bc_loss": bc_loss <= thresholds["bc_loss_max"],
        "ic_loss": ic_loss <= thresholds["ic_loss_max"],
        "prediction_error": prediction_error <= thresholds["prediction_error_max"],
    }
    status = SUPPORTED if all(checks.values()) else REJECTED

    parameter_error = None
    if parameter_estimate is not None and parameter_true is not None:
        parameter_error = {}
        for name, est in parameter_estimate.items():
            if name in parameter_true and parameter_true[name] != 0:
                true_val = parameter_true[name]
                parameter_error[name] = {
                    "absolute_error": abs(est - true_val),
                    "relative_error": abs(est - true_val) / abs(true_val),
                }

    return {
        "hypothesis": hypothesis_name,
        "equation": equation_str,
        "status": status,
        "failed_checks": [k for k, ok in checks.items() if not ok],
        "data_loss": data_loss,
        "physics_residual": physics_residual,
        "bc_loss": bc_loss,
        "ic_loss": ic_loss,
        "prediction_error": prediction_error,
        "ood_error": ood_error,
        "parameter_estimate": parameter_estimate,
        "parameter_true": parameter_true,
        "parameter_error": parameter_error,
        "thresholds_used": thresholds,
    }
