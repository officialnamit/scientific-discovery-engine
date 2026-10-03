"""
Phase 6: generic numerical forward solver for ANY candidate equation.

Given a candidate's equation string + fitted parameters + the known
IC/BC, integrate u_t = f(u, u_x, u_xx, u_xxx) with method of lines
(central finite differences in x, explicit Euler in t, Dirichlet-zero
boundaries). The right-hand side f is extracted symbolically from the
same parsed equation the PINN used, so the solver is not specific to
any one PDE.

Why this exists: the PINN only enforces physics where it was trained
(t <= t_cutoff); asking the network to extrapolate far beyond that is
asking a neural net, not the law. Once an equation and its parameters
are selected, the scientifically meaningful prediction is to SOLVE the
selected law forward -- that is the "new prediction" step, and it
doubles as the conventional numerical-solver baseline.
"""

from typing import Callable, Dict

import numpy as np
import sympy

from hypotheses.equation_compiler import FIELD_SYMBOLS, parse_equation
from llm.schemas import Hypothesis


def rhs_function(hyp: Hypothesis, params: Dict[str, float]) -> Callable:
    param_names = [p.name for p in hyp.parameters]
    residual_expr, _, _, local_dict = parse_equation(hyp.equation, param_names)
    u_t = local_dict["u_t"]
    expanded = sympy.expand(residual_expr)
    coeff = expanded.coeff(u_t)
    rhs = sympy.expand(-(expanded - coeff * u_t) / coeff)
    rhs = rhs.subs({local_dict[k]: v for k, v in params.items()})
    syms = [local_dict[f] for f in FIELD_SYMBOLS if f != "u_t"]
    fn = sympy.lambdify(syms, rhs, "numpy")
    return lambda fields: np.broadcast_to(
        np.asarray(fn(*[fields[s.name] for s in syms]), dtype=float), fields["u"].shape)


def _effective_coefficients(hyp: Hypothesis, params: Dict[str, float]):
    """|coef of u_xx|, |coef of u_x|, max |d f/d u| bound for the linearised reaction part."""
    param_names = [p.name for p in hyp.parameters]
    residual_expr, _, _, local = parse_equation(hyp.equation, param_names)
    u_t = local["u_t"]
    ex = sympy.expand(residual_expr)
    rhs = sympy.expand(-(ex - ex.coeff(u_t) * u_t) / ex.coeff(u_t)).subs({local[k]: v for k, v in params.items()})
    d = abs(float(rhs.coeff(local["u_xx"]).subs({s: 0 for s in rhs.free_symbols}) or 0))
    c = abs(float(rhs.coeff(local["u_x"]).subs({s: 0 for s in rhs.free_symbols}) or 0))
    reac = rhs.subs({local["u_xx"]: 0, local["u_x"]: 0, local["u_xxx"]: 0})
    r = max(abs(float(sympy.diff(reac, local["u"]).subs(local["u"], uv))) for uv in (0.0, 0.5, 1.0)) if reac.has(local["u"]) else 0.0
    return d, c, r


def solve_forward(hyp: Hypothesis, params: Dict[str, float], ic_fn: Callable,
                  x: np.ndarray, t_end: float, n_out: int = 101, cfl: float = 0.2,
                  method: str = "euler") -> Dict:
    """method="euler" (default; what Phase 6 used, kept for reproducibility) or
    "rk4" (Phase 7). Forward Euler with central differences is unconditionally
    unstable for pure advection; RK4 is stable for it at these CFL numbers."""
    f = rhs_function(hyp, params)
    dx = x[1] - x[0]
    if method == "rk4":
        # step sized from the ACTUAL coefficients of u_xx, u_x, u in the RHS (per-term limits)
        d_eff, c_eff, r_eff = _effective_coefficients(hyp, params)
        limits = [1e-3]
        if d_eff > 0: limits.append(dx ** 2 / (2 * d_eff))
        if c_eff > 0: limits.append(dx / c_eff)
        if r_eff > 0: limits.append(1.0 / r_eff)
        dt = 0.8 * min(limits)
    else:
        # Phase 6 rule (kept unchanged for reproducibility): every |param| treated as a scale
        scale = max([abs(v) for v in params.values()] + [1e-3])
        dt = cfl * min(dx ** 2 / (2 * scale), dx / scale, 1e-3)
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    out_every = max(1, n_steps // (n_out - 1))

    u = ic_fn(x).astype(float)
    u[0] = u[-1] = 0.0
    times, snaps = [0.0], [u.copy()]
    def F(v):
        vx = np.gradient(v, dx)
        vxx = np.gradient(vx, dx)
        out = f({"u": v, "u_x": vx, "u_xx": vxx, "u_xxx": np.gradient(vxx, dx)}).copy()
        out[0] = out[-1] = 0.0
        return out

    for step in range(1, n_steps + 1):
        if method == "rk4":
            k1 = F(u); k2 = F(u + 0.5 * dt * k1); k3 = F(u + 0.5 * dt * k2); k4 = F(u + dt * k3)
            u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            u = u + dt * F(u)
        u[0] = u[-1] = 0.0
        if not np.all(np.isfinite(u)):
            return {"stable": False, "t": np.array(times), "U": np.array(snaps)}
        if step % out_every == 0 or step == n_steps:
            times.append(step * dt)
            snaps.append(u.copy())
    return {"stable": True, "t": np.array(times), "U": np.array(snaps)}
