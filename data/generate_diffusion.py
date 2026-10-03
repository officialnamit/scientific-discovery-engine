"""
Phase 1, Step 2-3: Synthetic 1D diffusion dataset generator.

Ground truth (hidden from the discovery system):
    u_t = D * u_xx,   D = config.diffusion.D_true

This script:
  1. Solves the diffusion equation on a fine grid using an explicit
     finite-difference scheme (this doubles as the "numerical solver"
     baseline referenced later in the project).
  2. Saves the FULL clean field (x, t, u) -> full.npz
  3. Saves a SPARSE, noise-free subsample -> sparse.npz
  4. Saves a SPARSE, NOISY subsample -> noisy.npz  <- this is what the
     discovery system is actually allowed to see.

Run:
    python data/generate_diffusion.py --config config.yaml

Output directory: config.paths.output_dir (default: data/generated/)
"""

import argparse
import os

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def initial_condition(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "sin":
        # NOTE: a single sine mode is an EIGENFUNCTION of d^2/dx^2 on this
        # domain (sin(pi x) satisfies u_xx = -pi^2 * u exactly). That makes
        # "u" and "u_xx" perfectly collinear in the resulting dataset, so
        # equation discovery cannot structurally distinguish D*u_xx from
        # -D*pi^2*u no matter how good the regression is. Kept here for
        # reference/testing; NOT used as the discovery-demo default.
        return np.sin(np.pi * x)
    elif kind == "two_mode_sin":
        # sin(pi x) + 0.5*sin(3 pi x): still exactly zero at both
        # boundaries (so dirichlet_zero holds with no extra masking), but
        # a mix of two Laplacian eigenmodes with different eigenvalues
        # (-pi^2 and -9pi^2) breaks the u/u_xx collinearity above and
        # makes equation discovery well-posed. This is the default used
        # for the equation-discovery experiments (Phase 2).
        return np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    elif kind == "gaussian_bump":
        bump = np.exp(-((x - 0.5) ** 2) / (2 * 0.05 ** 2))
        # force zero at boundaries so it's consistent with dirichlet_zero
        bump[0] = 0.0
        bump[-1] = 0.0
        return bump
    else:
        raise ValueError(f"Unknown initial_condition kind: {kind}")


def solve_diffusion_fd(D: float, x: np.ndarray, t: np.ndarray, ic_kind: str) -> np.ndarray:
    """
    Explicit (forward-time, centered-space) finite-difference solver for
    u_t = D u_xx on [x_min, x_max] x [t_min, t_max] with u=0 at both
    boundaries (dirichlet_zero).

    Returns U of shape (n_t, n_x).
    """
    n_x = len(x)
    n_t = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    r = D * dt / dx ** 2
    if r > 0.5:
        raise ValueError(
            f"Stability condition violated: r = D*dt/dx^2 = {r:.4f} > 0.5. "
            f"Increase n_x/n_t resolution ratio or lower D_true / t_max in config.yaml."
        )

    U = np.zeros((n_t, n_x))
    U[0, :] = initial_condition(x, ic_kind)
    U[0, 0] = 0.0
    U[0, -1] = 0.0

    for n in range(0, n_t - 1):
        u = U[n, :]
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        U[n + 1, :] = u_new

    return U


def build_full_field(cfg: dict):
    dcfg = cfg["diffusion"]
    x = np.linspace(dcfg["x_min"], dcfg["x_max"], dcfg["n_x"])
    t = np.linspace(dcfg["t_min"], dcfg["t_max"], dcfg["n_t"])
    U = solve_diffusion_fd(dcfg["D_true"], x, t, dcfg["initial_condition"])
    return x, t, U


def flatten_field(x: np.ndarray, t: np.ndarray, U: np.ndarray):
    """Return (X, T, Uflat) as 1D arrays of matching length, one row per (x,t) pair."""
    Xg, Tg = np.meshgrid(x, t)  # shape (n_t, n_x) each, matches U
    return Xg.ravel(), Tg.ravel(), U.ravel()


def make_sparse_and_noisy(X, T, Uf, cfg: dict, rng: np.random.Generator):
    scfg = cfg["sampling"]
    n_total = len(Uf)

    # --- sparse (noise-free) subsample ---
    n_sparse = int(scfg["sparse_fraction_of_grid"] * n_total)
    sparse_idx = rng.choice(n_total, size=n_sparse, replace=False)
    sparse = {
        "x": X[sparse_idx],
        "t": T[sparse_idx],
        "u": Uf[sparse_idx],
    }

    # --- sparse + noisy subsample (this is what the discovery system sees) ---
    n_noisy = min(scfg["n_samples"], n_total)
    noisy_idx = rng.choice(n_total, size=n_noisy, replace=False)
    u_clean_subset = Uf[noisy_idx]
    noise_std = scfg["noise_std_fraction"] * np.std(Uf)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_noisy)
    noisy = {
        "x": X[noisy_idx],
        "t": T[noisy_idx],
        "u": u_clean_subset + noise,
        "u_clean": u_clean_subset,  # kept ONLY for evaluation/plots, not for discovery
        "noise_std": noise_std,
    }

    return sparse, noisy


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 1D diffusion dataset.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["seed"])

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    x, t, U = build_full_field(cfg)
    X, T, Uf = flatten_field(x, t, U)

    np.savez(
        os.path.join(out_dir, "full.npz"),
        x_grid=x, t_grid=t, U=U, X=X, T=T, U_flat=Uf,
        D_true=cfg["diffusion"]["D_true"],
    )

    sparse, noisy = make_sparse_and_noisy(X, T, Uf, cfg, rng)

    np.savez(os.path.join(out_dir, "sparse.npz"), **sparse)
    np.savez(
        os.path.join(out_dir, "noisy.npz"),
        x=noisy["x"], t=noisy["t"], u=noisy["u"],
        u_clean=noisy["u_clean"], noise_std=noisy["noise_std"],
    )

    print("Generated dataset in:", os.path.abspath(out_dir))
    print(f"  full.npz   : full field, shape U={U.shape} (n_t x n_x)")
    print(f"  sparse.npz : {len(sparse['u'])} noise-free (x,t,u) points")
    print(f"  noisy.npz  : {len(noisy['u'])} noisy (x,t,u) points "
          f"(noise_std={noisy['noise_std']:.5f})")
    print(f"  D_true = {cfg['diffusion']['D_true']} (hidden from discovery system)")
    print("\nNOTE: only noisy.npz (x, t, u — NOT u_clean, NOT D_true) should be")
    print("handed to the equation-discovery / PINN pipeline in later phases.")


if __name__ == "__main__":
    main()
