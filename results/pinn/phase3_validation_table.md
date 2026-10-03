| Hypothesis | Equation | Status | Data loss | Physics residual | BC loss | IC loss | In-dist RMSE | OOD RMSE |
|---|---|---|---|---|---|---|---|---|
| H1_diffusion | `u_t = D*u_xx` | supported by the available observations and physics constraints | 0.00013 | 0.00008 | 0.00000 | 0.00000 | 0.00078 | 0.00115 |
| H2_advection | `u_t = c*u_x` | rejected under the tested conditions | 0.03775 | 0.00499 | 0.00002 | 0.02515 | 0.20033 | 0.33145 |
