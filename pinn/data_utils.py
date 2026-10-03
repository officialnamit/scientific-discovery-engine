"""
Not required by the Phase 3 file list itself, but factored out to avoid
copy-pasting the same tensor-construction logic into all three
experiment scripts (03/04/05). Nothing PDE-specific lives here except
the two facts explicitly given as part of the experimental setup: the
boundary condition (u=0 at x=0,1) and the initial condition
(sin(pi*x)+0.5*sin(3*pi*x)) -- both come from config/the data
generator, never from a "discovered" or hidden equation.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_diffusion import initial_condition as _ic_fn  # noqa: E402


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32).reshape(-1, 1)


def load_observational_data(out_dir: str, which: str = "noisy") -> Dict[str, np.ndarray]:
    """which: 'noisy' or 'sparse' -- the only files the PINN is allowed
    to see for its data-loss term."""
    d = np.load(os.path.join(out_dir, f"{which}.npz"))
    return {"x": d["x"], "t": d["t"], "u": d["u"]}


def load_clean_reference(out_dir: str) -> Dict[str, np.ndarray]:
    """Loaded ONLY for evaluation (prediction error, OOD error) -- never
    used as a PINN training target."""
    d = np.load(os.path.join(out_dir, "full.npz"))
    return {"x": d["X"], "t": d["T"], "u": d["U_flat"], "D_true": float(d["D_true"])}


def make_collocation_points(
    x_min: float, x_max: float, t_min: float, t_max: float, n: int, seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n)
    t = rng.uniform(t_min, t_max, size=n)
    return {"x": _to_tensor(x), "t": _to_tensor(t)}


def make_bc_points(x_min: float, x_max: float, t_min: float, t_max: float, n: int, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """n points at x=x_min and n points at x=x_max, random t; target u=0
    (dirichlet_zero boundary condition, as generated in Phase 1)."""
    rng = np.random.default_rng(seed)
    t_left = rng.uniform(t_min, t_max, size=n)
    t_right = rng.uniform(t_min, t_max, size=n)
    x = np.concatenate([np.full(n, x_min), np.full(n, x_max)])
    t = np.concatenate([t_left, t_right])
    u = np.zeros_like(x)
    return {"x": _to_tensor(x), "t": _to_tensor(t), "u": _to_tensor(u)}


def make_ic_points(x_min: float, x_max: float, n: int, ic_kind: str, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """n points at t=0, random x; target u = initial_condition(x) (the
    known IC used to generate the Phase 1 data)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n)
    t = np.zeros_like(x)
    u = ic_kind(x) if callable(ic_kind) else _ic_fn(x, ic_kind)  # Phase 7: callable IC for new systems
    return {"x": _to_tensor(x), "t": _to_tensor(t), "u": _to_tensor(u)}


def data_dict_to_tensors(data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {"x": _to_tensor(data["x"]), "t": _to_tensor(data["t"]), "u": _to_tensor(data["u"])}


def filter_by_time(data: Dict[str, np.ndarray], t_max: float) -> Dict[str, np.ndarray]:
    mask = data["t"] <= t_max
    return {k: (v[mask] if hasattr(v, "__len__") and len(v) == len(data["t"]) else v) for k, v in data.items()}
