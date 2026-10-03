"""
Phase 2, Step 2: Derivative estimation.

Two situations are handled:

1. DATA ALREADY ON A REGULAR GRID (the "clean" full-field case).
   Just apply finite differences directly.

2. SCATTERED / SPARSE DATA (the "sparse" and "noisy" cases).
   Sparse or noisy (x, t, u) triples are NOT on a convenient regular
   grid to difference. The standard approach (used e.g. in the PDE-FIND
   literature) is:
       a. interpolate the scattered points onto a regular reconstruction
          grid (scipy.interpolate.griddata),
       b. optionally smooth that reconstructed field to denoise it
          (numerical differentiation amplifies noise -- a lot, and more
          so for higher-order derivatives),
       c. apply finite differences to the (smoothed) reconstructed grid.

LIMITATIONS (please read before trusting derivative estimates):
  - Interpolation from a genuinely sparse point cloud onto a fine grid
    introduces reconstruction error that has nothing to do with the
    underlying PDE -- it's a property of the sampling, not the physics.
    Using a COARSER reconstruction grid than the original simulation
    grid trades resolution for stability; this is a deliberate config
    choice (see config.yaml: discovery.reconstruction).
  - Gaussian smoothing reduces noise amplification but introduces bias
    (it will slightly flatten sharp features and shrink coefficient
    estimates for terms tied to high spatial-frequency structure).
  - Third derivatives (u_xxx) are especially unreliable on noisy data:
    each np.gradient call roughly doubles the noise amplification, so
    u_xxx has amplified noise ~3x more than u_x. Treat u_xxx results
    on noisy data with real skepticism -- this is reported, not hidden.
  - Finite differences near the domain boundary use one-sided stencils
    (via np.gradient) which are less accurate. We crop a configurable
    margin off every edge before handing points to the regression so
    the discovered equation isn't distorted by boundary artifacts.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def reconstruct_grid(
    x: np.ndarray, t: np.ndarray, u: np.ndarray,
    x_grid: np.ndarray, t_grid: np.ndarray,
    method: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate scattered (x, t, u) samples onto a regular (t_grid x x_grid)
    mesh. Falls back to nearest-neighbor fill for any points griddata
    can't interpolate (typically right at the convex-hull boundary).

    Returns (U_grid, X_mesh, T_mesh), all shape (len(t_grid), len(x_grid)).
    """
    Xg, Tg = np.meshgrid(x_grid, t_grid)  # shape (n_t, n_x)
    points = np.column_stack([x, t])

    U_grid = griddata(points, u, xi=(Xg, Tg), method=method)
    if np.isnan(U_grid).any():
        U_nearest = griddata(points, u, xi=(Xg, Tg), method="nearest")
        U_grid = np.where(np.isnan(U_grid), U_nearest, U_grid)

    return U_grid, Xg, Tg


def smooth_grid(U_grid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing (denoising) of a reconstructed grid. sigma is in
    grid-index units (not physical units). sigma<=0 disables smoothing."""
    if sigma and sigma > 0:
        return gaussian_filter(U_grid, sigma=sigma, mode="nearest")
    return U_grid


def finite_diff_derivatives(
    U_grid: np.ndarray, x_grid: np.ndarray, t_grid: np.ndarray, crop: int = 3
) -> Dict[str, np.ndarray]:
    """
    Central finite differences (via np.gradient) for u, u_t, u_x, u_xx, u_xxx
    on a regular grid of shape (n_t, n_x).

    `crop` grid cells are trimmed from every edge of both axes before
    returning, to avoid the less-accurate one-sided stencils at the
    boundary (and their compounding effect on the twice- and
    thrice-differentiated fields).

    Returns a dict of 2D arrays, all shape (n_t - 2*crop, n_x - 2*crop):
        "u", "u_t", "u_x", "u_xx", "u_xxx", "x", "t"
    """
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    u_t = np.gradient(U_grid, dt, axis=0)
    u_x = np.gradient(U_grid, dx, axis=1)
    u_xx = np.gradient(u_x, dx, axis=1)
    u_xxx = np.gradient(u_xx, dx, axis=1)

    Xg, Tg = np.meshgrid(x_grid, t_grid)

    n_t, n_x = U_grid.shape
    c = crop
    if n_t - 2 * c <= 0 or n_x - 2 * c <= 0:
        raise ValueError(
            f"crop={c} is too large for grid shape {U_grid.shape}; "
            "reduce discovery.reconstruction crop or increase grid resolution."
        )
    sl = (slice(c, n_t - c), slice(c, n_x - c))

    return {
        "u": U_grid[sl],
        "u_t": u_t[sl],
        "u_x": u_x[sl],
        "u_xx": u_xx[sl],
        "u_xxx": u_xxx[sl],
        "x": Xg[sl],
        "t": Tg[sl],
    }
