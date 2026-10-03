"""
Phase 3, Part 2: automatic differentiation for PINN derivatives.

Every derivative here (u_t, u_x, u_xx) comes from torch.autograd, not
finite differences -- this is the entire point of a PINN: the network
is queried at arbitrary (x,t), including points where there is no
observational data at all (the "collocation" points used for the
physics loss).
"""

from typing import Dict

import torch


def compute_derivatives(model, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    x, t: shape (N, 1), do NOT need requires_grad set by the caller --
    this function sets it internally.

    Returns dict with "u", "u_t", "u_x", "u_xx", "u_xxx", each shape (N, 1).

    create_graph=True on every grad() call is required so that (a)
    second/third derivatives can themselves be differentiated again if
    needed and (b) the physics loss built from these derivatives can
    still be backpropagated into the network's (and any physics
    parameter's) weights during training.

    u_xxx is included for Phase 4's generic equation compiler (which
    may receive LLM-proposed equations referencing third derivatives,
    e.g. dispersive terms) -- it costs one extra autograd.grad call and
    is unused by the diffusion/advection hypotheses tested so far.
    """
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0]
    u_xxx = torch.autograd.grad(
        u_xx, x, grad_outputs=torch.ones_like(u_xx), create_graph=True, retain_graph=True
    )[0]

    return {"u": u, "u_t": u_t, "u_x": u_x, "u_xx": u_xx, "u_xxx": u_xxx}
