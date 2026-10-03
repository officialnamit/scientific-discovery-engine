Relative global balance violation (0 = perfectly conserving w.r.t. its own law)

| Run | Candidate | Equation | Gate | train | OOD | beyond |
|---|---|---|---|---|---|---|
| mock | H1 | `u_t = D*u_xx` | pass | 1.05e-03 | 3.66e-03 | 3.30e-02 |
| mock | H2 | `u_t + c*u_x = 0` | fail | 1.00e+00 | 1.01e+00 | 1.01e+00 |
| mock | H3 | `u_t = D*u_xx + r*u*(1 - u/K)` | pass | 1.01e-03 | 1.79e-03 | 1.48e-02 |
| mock | H4 | `u_t + c*u_x = D*u_xx` | pass | 1.92e-03 | 2.98e-02 | 1.33e-01 |
| mock | K3 | `u_t = r*u*(1-u/K)` | fail | 6.27e-03 | 8.04e-03 | 8.97e-02 |
| gemini | H1 | `u_t = D * u_xx` | pass | 2.11e-03 | 2.03e-02 | 8.77e-02 |
| gemini | H2 | `u_t = D * u_xx - k * u` | pass | 4.21e-03 | 3.64e-03 | 9.04e-02 |
| gemini | H3 | `u_t + c * u_x = D * u_xx - gamma * u` | pass | 2.39e-03 | 7.21e-03 | 1.69e-02 |
| gemini | H4 | `u_t = D * u_xx + r * u * (1 - u)` | pass | 9.65e-04 | 1.51e-02 | 9.04e-02 |
| gemini | K1 | `u_t + c*u_x = 0` | fail | 1.00e+00 | 1.01e+00 | 1.01e+00 |
| gemini | K3 | `u_t = r*u*(1-u/K)` | fail | 6.27e-03 | 8.04e-03 | 8.97e-02 |
| baseline | MLP (no physics) | `u_t = D*u_xx` | - | 1.49e-01 | 5.40e-02 | 3.18e-01 |
