"""
Phase 1, Step 6: Load the generated diffusion dataset, run sanity checks,
and produce a verification plot.

Run (from repo root, after generating data):
    python tests/test_diffusion_data.py --config config.yaml

What this checks:
  1. Files exist and load correctly.
  2. Shapes are consistent (full field vs. flattened arrays vs. samples).
  3. Boundary conditions are ~0 in the full field (dirichlet_zero).
  4. The sparse/noisy subsets are genuinely a strict subset of the full grid
     size (i.e. sparsity actually happened).
  5. Noisy data's mean absolute deviation from clean data is in the
     right ballpark given noise_std (catches silent noise-generation bugs).
  6. Saves a PNG with three panels: full-field heatmap, sparse points,
     noisy points (colored by u) so you can eyeball that it looks like
     diffusion (peak decaying and smoothing out over time).
"""

import argparse
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def check(condition: bool, message: str):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        raise AssertionError(message)


def run_checks(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    out_dir = cfg["paths"]["output_dir"]

    full_path = os.path.join(out_dir, "full.npz")
    sparse_path = os.path.join(out_dir, "sparse.npz")
    noisy_path = os.path.join(out_dir, "noisy.npz")

    for p in [full_path, sparse_path, noisy_path]:
        check(os.path.exists(p), f"File exists: {p}")

    full = np.load(full_path)
    sparse = np.load(sparse_path)
    noisy = np.load(noisy_path)

    x_grid, t_grid, U = full["x_grid"], full["t_grid"], full["U"]
    X, T, Uf = full["X"], full["T"], full["U_flat"]
    D_true = float(full["D_true"])

    n_t, n_x = U.shape
    check(U.shape == (len(t_grid), len(x_grid)), "Full field shape matches grid sizes")
    check(len(X) == len(T) == len(Uf) == n_t * n_x, "Flattened arrays match full field size")

    # Boundary condition check: u(x_min, t) and u(x_max, t) should be ~0 for all t
    left_edge = U[:, 0]
    right_edge = U[:, -1]
    check(np.allclose(left_edge, 0.0, atol=1e-8), "Left boundary u(x_min, t) ~= 0")
    check(np.allclose(right_edge, 0.0, atol=1e-8), "Right boundary u(x_max, t) ~= 0")

    # Diffusion sanity: the field should decay over time (max |u| shrinks)
    max_abs_per_t = np.max(np.abs(U), axis=1)
    check(
        max_abs_per_t[-1] < max_abs_per_t[0],
        f"Field amplitude decays over time (max|u| at t0={max_abs_per_t[0]:.4f} "
        f"-> t_end={max_abs_per_t[-1]:.4f})",
    )

    # Sparsity checks
    n_full = n_t * n_x
    check(len(sparse["u"]) < n_full, f"Sparse set ({len(sparse['u'])} pts) is a strict subset of full grid ({n_full} pts)")
    check(len(noisy["u"]) < n_full, f"Noisy set ({len(noisy['u'])} pts) is a strict subset of full grid ({n_full} pts)")

    # Noise sanity: mean abs deviation from clean should be roughly consistent
    # with noise_std (within a generous factor, since it's a random sample)
    noise_std = float(noisy["noise_std"])
    actual_mad = np.mean(np.abs(noisy["u"] - noisy["u_clean"]))
    expected_mad = noise_std * np.sqrt(2 / np.pi)  # E[|N(0,std)|]
    ratio = actual_mad / expected_mad if expected_mad > 0 else float("nan")
    check(
        0.5 < ratio < 2.0,
        f"Noisy data's mean abs deviation ({actual_mad:.5f}) is in expected "
        f"range around theoretical {expected_mad:.5f} (ratio={ratio:.2f})",
    )

    print(f"\nD_true (hidden ground truth, for grading only) = {D_true}")
    print(f"Full field: {n_t} time steps x {n_x} spatial points = {n_full} total points")
    print(f"Sparse (clean) points handed nowhere yet: {len(sparse['u'])}")
    print(f"Noisy points that Phase 1 discovery will actually use: {len(noisy['u'])} "
          f"(noise_std={noise_std:.5f})")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(
        U, aspect="auto", origin="lower",
        extent=[x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()],
        cmap="viridis",
    )
    axes[0].set_title("Full clean field u(x,t)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0])

    sc1 = axes[1].scatter(sparse["x"], sparse["t"], c=sparse["u"], cmap="viridis", s=10)
    axes[1].set_title(f"Sparse, noise-free ({len(sparse['u'])} pts)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    fig.colorbar(sc1, ax=axes[1])

    sc2 = axes[2].scatter(noisy["x"], noisy["t"], c=noisy["u"], cmap="viridis", s=10)
    axes[2].set_title(f"Sparse + noisy ({len(noisy['u'])} pts) — used by discovery system")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    fig.colorbar(sc2, ax=axes[2])

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "verification_plot.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nSaved verification plot to: {os.path.abspath(plot_path)}")

    print("\nAll checks passed.")


def test_phase1_dataset():
    """pytest entry point (previously this file was a main()-only script, so
    `pytest tests/` collected ZERO Phase 1 tests -- found in the Phase 6 audit).
    Also checks the two-mode IC is what the stored dataset actually contains."""
    run_checks("config.yaml")
    d = np.load("data/generated/full.npz")
    x = d["x_grid"]
    two_mode = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    assert np.allclose(d["U"][0], two_mode, atol=1e-12), "Stored IC is not the two-mode IC"
    noisy = np.load("data/generated/noisy.npz")
    assert set(["x", "t", "u"]).issubset(noisy.files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    run_checks(args.config)


if __name__ == "__main__":
    main()
