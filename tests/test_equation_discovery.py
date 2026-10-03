"""
Phase 2 tests.

1. test_clean_diffusion_recovery: the key demonstration -- clean data
   from the Phase 1 generator should yield u_t ~= D*u_xx with D within
   5% of the true 0.1, and no other significant candidate terms.
2. test_generality_on_advection: PROVES the "IMPORTANT ARCHITECTURAL
   REQUIREMENT" from the spec -- the exact same discovery_engine /
   library / sindy code, given a completely different synthetic dataset
   (1D advection, u_t = -c*u_x) and told nothing about which terms
   matter, recovers the advection structure instead. Nothing about
   equation_discovery/ is diffusion-specific.
3. test_stlsq_recovers_known_sparse_signal: a fast, fully-synthetic unit
   test of the regression core in isolation (no PDE data at all).
4. test_library_shapes_and_labels: unit test for the candidate library.

Run:
    python -m pytest tests/test_equation_discovery.py -v
or:
    python tests/test_equation_discovery.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equation_discovery.derivatives import finite_diff_derivatives
from equation_discovery.discovery_engine import (
    discover_from_grid,
    evaluate_against_ground_truth,
)
from equation_discovery.library import build_library, term_label
from equation_discovery.sindy import stlsq


DIFFUSION_TERM_SPECS = [(), ("u",), ("u_x",), ("u_xx",), ("u_xxx",), ("u", "u"), ("u", "u_x"), ("u", "u_xx")]


def test_clean_diffusion_recovery():
    data_path = "data/generated/full.npz"
    assert os.path.exists(data_path), "Run data/generate_diffusion.py first."
    full = np.load(data_path)
    x_grid, t_grid, U = full["x_grid"], full["t_grid"], full["U"]
    D_true = float(full["D_true"])

    result, _, _ = discover_from_grid(
        U, x_grid, t_grid, DIFFUSION_TERM_SPECS, crop=3,
        sindy_kwargs=dict(threshold=0.05, alpha=0.0, max_iterations=20),
    )
    eval_result = evaluate_against_ground_truth(result, true_term="u_xx", coefficient_true=D_true)

    assert eval_result["structure_recovered"], (
        f"Expected only u_xx to be selected, got extra terms: {eval_result['extra_terms']}"
    )
    assert eval_result["relative_parameter_error"] < 0.05, (
        f"D estimate off by {eval_result['relative_parameter_error']*100:.2f}% (expected < 5%)"
    )
    print(f"[PASS] test_clean_diffusion_recovery: D_pred={eval_result['coefficient_pred']:.4f}, "
          f"rel_error={eval_result['relative_parameter_error']*100:.2f}%")


def test_generality_on_advection():
    """
    Same engine, different physics: u_t + c*u_x = 0  =>  u_t = -c*u_x.
    u(x,t) = f(x - c*t) for a smooth bump f. No diffusion term at all.
    This dataset is built in-memory here -- it has nothing to do with
    the Phase 1 diffusion generator -- to prove the discovery code is
    not secretly hard-coded for u_xx.
    """
    c_true = 0.5
    x = np.linspace(0.0, 2.0, 121)
    t = np.linspace(0.0, 1.0, 161)
    Xg, Tg = np.meshgrid(x, t)

    def bump(s):
        return np.exp(-((s - 0.5) ** 2) / (2 * 0.1 ** 2))

    U = bump(Xg - c_true * Tg)  # exact solution of u_t = -c u_x

    term_specs = [(), ("u",), ("u_x",), ("u_xx",), ("u", "u_x")]
    result, _, _ = discover_from_grid(
        U, x, t, term_specs, crop=3,
        sindy_kwargs=dict(threshold=0.05, alpha=0.0, max_iterations=20),
    )
    eval_result = evaluate_against_ground_truth(result, true_term="u_x", coefficient_true=-c_true)

    assert eval_result["structure_recovered"], (
        f"Expected only u_x to be selected for advection data, got: {result['all_coefficients']}"
    )
    assert eval_result["relative_parameter_error"] < 0.05, (
        f"c estimate off by {eval_result['relative_parameter_error']*100:.2f}% (expected < 5%)"
    )
    print(f"[PASS] test_generality_on_advection: c_pred={-eval_result['coefficient_pred']:.4f} "
          f"(true c={c_true}), rel_error={eval_result['relative_parameter_error']*100:.2f}%")


def test_stlsq_recovers_known_sparse_signal():
    rng = np.random.default_rng(0)
    n, n_features = 500, 6
    Theta = rng.normal(size=(n, n_features))
    true_xi = np.array([0.0, 0.0, 2.5, 0.0, -1.2, 0.0])
    y = Theta @ true_xi + rng.normal(scale=0.01, size=n)

    xi = stlsq(Theta, y, threshold=0.1, alpha=0.0, max_iterations=20)

    nonzero_true = set(np.nonzero(true_xi)[0].tolist())
    nonzero_pred = set(np.nonzero(np.abs(xi) > 1e-6)[0].tolist())
    assert nonzero_pred == nonzero_true, f"Expected support {nonzero_true}, got {nonzero_pred}"
    assert np.allclose(xi[list(nonzero_true)], true_xi[list(nonzero_true)], atol=0.05)
    print(f"[PASS] test_stlsq_recovers_known_sparse_signal: recovered {xi}")


def test_library_shapes_and_labels():
    n = 10
    fields = {"u": np.arange(n, dtype=float), "u_x": np.ones(n)}
    term_specs = [(), ("u",), ("u", "u"), ("u", "u_x")]
    Theta, names = build_library(fields, term_specs)

    assert Theta.shape == (n, 4)
    assert names == ["1", "u", "u^2", "u*u_x"]
    assert np.allclose(Theta[:, 0], 1.0)
    assert np.allclose(Theta[:, 1], fields["u"])
    assert np.allclose(Theta[:, 2], fields["u"] ** 2)
    assert np.allclose(Theta[:, 3], fields["u"] * fields["u_x"])
    assert term_label(()) == "1"
    print("[PASS] test_library_shapes_and_labels")


if __name__ == "__main__":
    test_library_shapes_and_labels()
    test_stlsq_recovers_known_sparse_signal()
    test_generality_on_advection()
    test_clean_diffusion_recovery()
    print("\nAll Phase 2 tests passed.")
