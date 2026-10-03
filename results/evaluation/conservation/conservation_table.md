Relative global balance violation (0 = perfectly conserving w.r.t. its own law)

| Run | Candidate | Equation | Gate | train | OOD | beyond |
|---|---|---|---|---|---|---|
| mock | H1 | `u_t = D*u_xx` | pass | 7.63e-04 | 1.70e-02 | 5.60e-02 |
| mock | H2 | `u_t + c*u_x = 0` | fail | 1.00e+00 | 1.00e+00 | 1.00e+00 |
| mock | H3 | `u_t = D*u_xx + r*u*(1 - u/K)` | pass | 1.12e-03 | 5.14e-03 | 4.63e-02 |
| mock | H4 | `u_t + c*u_x = D*u_xx` | pass | 1.31e-03 | 3.60e-02 | 1.67e-01 |
| mock | K3 | `u_t = r*u*(1-u/K)` | fail | 7.45e-03 | 9.36e-03 | 1.09e-02 |
| gemini | H1 | `u_t = D * u_xx` | pass | 2.11e-03 | 2.03e-02 | 8.77e-02 |
| gemini | H2 | `u_t = D * u_xx - r * u` | pass | 4.21e-03 | 3.64e-03 | 9.04e-02 |
| gemini | H3 | `u_t + c * u_x = D * u_xx` | pass | 2.22e-03 | 1.59e-02 | 1.38e-01 |
| gemini | H4 | `u_t = D * u_xx + r * u * (1.0 - u / K)` | pass | 1.64e-03 | 2.99e-02 | 1.64e-01 |
| gemini | K1 | `u_t + c*u_x = 0` | fail | 1.00e+00 | 1.01e+00 | 1.01e+00 |
| gemini | K3 | `u_t = r*u*(1-u/K)` | fail | 6.27e-03 | 8.04e-03 | 8.97e-02 |
| baseline | MLP (no physics) | `u_t = D*u_xx` | - | 1.50e-01 | 5.34e-02 | 3.16e-01 |
