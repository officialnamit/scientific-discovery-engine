| Dataset | Discovered Equation | D estimate | Rel. param error | Structure recovered |
|---|---|---|---|---|
| Clean | `u_t = 0.1009*u_xx` | 0.1009 | 0.88% | True |
| Sparse | `u_t = 0.1013*u_xx` | 0.1013 | 1.26% | True |
| Noisy (no smoothing) | `u_t = 0.5039*1 - 3.418*u + 4.147*u^2 + 0.05336*u*u_x + 0.1676*u*u_xx` | 0.0000 | 100.00% | False |
| Noisy (smoothed) | `u_t = 0.545*u + 0.1968*u_xx - 0.5059*u^2 + 0.06862*u*u_x - 0.1077*u*u_xx` | 0.1968 | 96.78% | False |
