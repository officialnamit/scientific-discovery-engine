"""
Phase 3 tests.

1. test_network_output_shape: basic sanity on the reusable PINN class.
2. test_autograd_derivatives_match_analytic: verifies pinn/derivatives.py
   against a function with a KNOWN closed-form derivative
   (u = sin(x)*exp(-t)  =>  u_x=cos(x)*exp(-t), u_t=-sin(x)*exp(-t),
   u_xx=-sin(x)*exp(-t)) using a tiny untrained network wrapper replaced
   by a direct closed-form nn.Module, so this test does not depend on
   training convergence at all -- it isolates the autograd machinery.
3. test_residual_registry: diffusion/advection residual functions
   compute the expected arithmetic on synthetic tensors.
4. test_trainer_smoke: a FAST (small, few-epoch) end-to-end run of
   PINNTrainer on synthetic linear data, just to confirm the training
   loop (data+physics+bc+ic losses, backward, optimizer step) runs
   without error and total loss decreases.
5. test_phase3_reported_results: if the (slower, full) experiment
   scripts have already been run and left their JSON reports in
   results/pinn/, sanity-check the headline numbers -- D_hat close to
   D_true for both forward/inverse, and H1 clearly better than H2 on
   every metric. Skipped (not failed) if those files aren't present.

Run:
    python tests/test_pinn.py
"""

import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinn.derivatives import compute_derivatives
from pinn.losses import bc_loss, data_loss, ic_loss, physics_loss, total_loss
from pinn.network import PINN
from pinn.residuals import RESIDUAL_REGISTRY, make_params
from pinn.trainer import PINNTrainer


def test_network_output_shape():
    model = PINN(hidden_dims=[16, 16], activation="tanh")
    x = torch.rand(10, 1)
    t = torch.rand(10, 1)
    u = model(x, t)
    assert u.shape == (10, 1)
    print("[PASS] test_network_output_shape")


class _ClosedFormWrapper(nn.Module):
    """u(x,t) = sin(x)*exp(-t), with known derivatives, used to test
    the autograd machinery independent of any trained network."""
    def forward(self, x, t):
        return torch.sin(x) * torch.exp(-t)


def test_autograd_derivatives_match_analytic():
    model = _ClosedFormWrapper()
    x = torch.linspace(0.1, 2.0, 20).reshape(-1, 1)
    t = torch.linspace(0.1, 1.0, 20).reshape(-1, 1)

    derivs = compute_derivatives(model, x, t)

    u_x_true = torch.cos(x) * torch.exp(-t)
    u_t_true = -torch.sin(x) * torch.exp(-t)
    u_xx_true = -torch.sin(x) * torch.exp(-t)

    assert torch.allclose(derivs["u_x"], u_x_true, atol=1e-5)
    assert torch.allclose(derivs["u_t"], u_t_true, atol=1e-5)
    assert torch.allclose(derivs["u_xx"], u_xx_true, atol=1e-5)
    print("[PASS] test_autograd_derivatives_match_analytic")


def test_residual_registry():
    derivs = {
        "u": torch.tensor([1.0, 2.0]),
        "u_t": torch.tensor([0.5, -0.3]),
        "u_x": torch.tensor([2.0, 1.0]),
        "u_xx": torch.tensor([4.0, -1.0]),
    }
    params = make_params({"D": 0.1}, trainable=False)
    residual = RESIDUAL_REGISTRY["diffusion"]["fn"](derivs, params)
    expected = derivs["u_t"] - 0.1 * derivs["u_xx"]
    assert torch.allclose(residual, expected)

    params_c = make_params({"c": 0.5}, trainable=False)
    residual_adv = RESIDUAL_REGISTRY["advection"]["fn"](derivs, params_c)
    expected_adv = derivs["u_t"] - 0.5 * derivs["u_x"]
    assert torch.allclose(residual_adv, expected_adv)
    print("[PASS] test_residual_registry")


def test_trainer_smoke():
    """Fast smoke test: does the full training loop run and reduce loss?
    Ground truth here is u(x,t)=x*t (so u_t=x, u_x=t, u_xx=0), used only
    to build a synthetic data+physics problem with a KNOWN closed-form
    answer, purely to keep this test fast and self-contained."""
    torch.manual_seed(0)
    n = 50
    x = torch.rand(n, 1)
    t = torch.rand(n, 1)
    u = x * t

    model = PINN(hidden_dims=[16, 16], activation="tanh")
    params = make_params({"D": 0.5}, trainable=True)  # arbitrary, will be trained
    weights = {"lambda_data": 1.0, "lambda_physics": 0.1, "lambda_bc": 1.0, "lambda_ic": 1.0}

    def residual_fn(derivs, p):
        # not physically meaningful here -- just needs to be differentiable
        return derivs["u_t"] - p["D"] * derivs["u_xx"]

    trainer = PINNTrainer(model, residual_fn, params, weights)
    data = {"x": x, "t": t, "u": u}
    collocation = {"x": torch.rand(50, 1), "t": torch.rand(50, 1)}
    bc = {"x": torch.zeros(10, 1), "t": torch.rand(10, 1), "u": torch.zeros(10, 1)}
    ic = {"x": torch.rand(10, 1), "t": torch.zeros(10, 1), "u": torch.zeros(10, 1)}

    history = trainer.train(data, collocation, bc, ic, adam_epochs=100, adam_lr=1e-2, lbfgs_epochs=0, log_every=25)

    assert len(history) > 0
    assert history[-1]["total"] < history[0]["total"], "Loss did not decrease during smoke training"
    print(f"[PASS] test_trainer_smoke: loss {history[0]['total']:.4f} -> {history[-1]['total']:.4f}")


def test_phase3_reported_results():
    forward_path = "results/pinn/phase3_forward_report.json"
    inverse_path = "results/pinn/phase3_inverse_report.json"
    hyp_path = "results/pinn/phase3_hypothesis_validation_report.json"

    if not (os.path.exists(forward_path) and os.path.exists(inverse_path) and os.path.exists(hyp_path)):
        print("[SKIP] test_phase3_reported_results: experiment reports not found, run experiments/03-05 first")
        return

    with open(forward_path) as f:
        fwd = json.load(f)
    with open(inverse_path) as f:
        inv = json.load(f)
    with open(hyp_path) as f:
        hyp = json.load(f)

    assert fwd["prediction_error_vs_clean"]["rmse"] < 0.01, "Forward PINN prediction error too high"
    assert inv["relative_parameter_error"] < 0.05, "Inverse PINN D estimate off by more than 5%"

    h1 = hyp["H1_diffusion"]
    h2 = hyp["H2_advection"]
    assert h1["status"] == "supported by the available observations and physics constraints"
    assert h2["status"] == "rejected under the tested conditions"
    assert h1["ood_rmse"] < h2["ood_rmse"], "H1 should generalize better OOD than H2"
    assert h1["data_loss"] < h2["data_loss"]
    assert h1["physics_residual"] < h2["physics_residual"]

    print(f"[PASS] test_phase3_reported_results: forward RMSE={fwd['prediction_error_vs_clean']['rmse']:.5f}, "
          f"inverse D_hat={inv['D_hat']:.5f} (rel_err={inv['relative_parameter_error']*100:.3f}%), "
          f"H1 OOD RMSE={h1['ood_rmse']:.5f} vs H2 OOD RMSE={h2['ood_rmse']:.5f}")


if __name__ == "__main__":
    test_network_output_shape()
    test_autograd_derivatives_match_analytic()
    test_residual_registry()
    test_trainer_smoke()
    test_phase3_reported_results()
    print("\nAll Phase 3 tests passed (or skipped where experiment reports weren't found).")
