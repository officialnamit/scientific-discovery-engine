| Hypothesis | Equation | Status | Data loss | Physics residual | BC loss | IC loss | In-dist RMSE | OOD RMSE |
|---|---|---|---|---|---|---|---|---|
| H1_diffusion | `u_t = D*u_xx` | supported by the available observations and physics constraints | 0.00013 | 0.00007 | 0.00000 | 0.00000 | 0.00070 | 0.00115 |
| H2_advection | `u_t = c*u_x` | rejected under the tested conditions | 0.03752 | 0.00519 | 0.00000 | 0.02500 | 0.20073 | 0.33418 |
