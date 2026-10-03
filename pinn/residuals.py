"""
Phase 3, Part 3 & Part 8: physics residual interface.

A "hypothesis" is a name, a residual function, and the names of its
free physical parameters. The PINN trainer only ever calls
`residual_fn(derivs, params)` -- it has no idea whether it's testing
diffusion, advection, or anything else. Adding a new candidate PDE
(e.g. reaction-diffusion, u_t - D*u_xx - k*u) means adding one entry
here; nothing in pinn/trainer.py, pinn/network.py, or
pinn/derivatives.py needs to change.
"""

from typing import Callable, Dict

import torch

ResidualFn = Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], torch.Tensor]


def diffusion_residual(derivs: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """H1: u_t = D*u_xx  =>  residual = u_t - D*u_xx"""
    return derivs["u_t"] - params["D"] * derivs["u_xx"]


def advection_residual(derivs: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """H2: u_t = c*u_x  =>  residual = u_t - c*u_x"""
    return derivs["u_t"] - params["c"] * derivs["u_x"]


def reaction_diffusion_residual(derivs: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """u_t = D*u_xx - k*u  =>  residual = u_t - D*u_xx + k*u   (not used yet, included to
    demonstrate the interface supports more terms without touching the trainer)."""
    return derivs["u_t"] - params["D"] * derivs["u_xx"] + params["k"] * derivs["u"]


RESIDUAL_REGISTRY: Dict[str, Dict] = {
    "diffusion": {
        "fn": diffusion_residual,
        "param_names": ["D"],
        "equation_str": "u_t = D*u_xx",
    },
    "advection": {
        "fn": advection_residual,
        "param_names": ["c"],
        "equation_str": "u_t = c*u_x",
    },
    "reaction_diffusion": {
        "fn": reaction_diffusion_residual,
        "param_names": ["D", "k"],
        "equation_str": "u_t = D*u_xx - k*u",
    },
}


def make_params(param_init: Dict[str, float], trainable) -> Dict[str, torch.Tensor]:
    """Build the params dict handed to a residual function.

    `trainable` accepts either:
      - a single bool applied to every parameter (Phase 3 usage: forward.py
        passes False, inverse.py passes True), or
      - a dict {param_name: bool} for per-parameter control (Phase 4 usage:
        an LLM-generated hypothesis may mark some parameters trainable and
        others fixed). Any name missing from the dict defaults to True.

    Trainable parameters are nn.Parameter (add to the optimizer yourself);
    fixed parameters are plain (non-leaf-grad) tensors.
    """
    params = {}
    for name, value in param_init.items():
        t = torch.tensor(float(value), dtype=torch.float32)
        if isinstance(trainable, dict):
            is_trainable = trainable.get(name, True)
        else:
            is_trainable = trainable
        params[name] = torch.nn.Parameter(t, requires_grad=True) if is_trainable else t
    return params
