# Phase 4: Generated Hypotheses Report

## Problem description

```
We observe a scalar field u(x,t) on a 1D spatial domain x in [0,1] over a time window t in [0,1], measured via 400 sparse, noisy point samples (x_i, t_i, u_i) -- no dense grid or simulation is provided to you.

Observed facts (computed directly from the data):
- Near t=0.05, the field's peak absolute amplitude is approximately 0.983.
- Near t=0.95, the field's peak absolute amplitude is approximately 0.395 (i.e. the amplitude has decreased substantially over the observation window).
- Near the domain edges (x<0.03 or x>0.97), the field stays very close to zero at essentially all observed times (max |u| there is approximately 0.0711).
- The amplitude-weighted spatial centroid of |u| is approximately x~0.53 near t=0.05 and x~0.41 near t=0.95 (i.e. little evidence of bulk translation of the field across the domain -- compare this to the amplitude decay above).
- The spatial profile appears to smooth out over time: early-time samples show more fine-grained spatial structure (multiple local peaks) than late-time samples.

Propose plausible governing mechanisms for this field's evolution.
```

Provider: `MockLLMProvider`

## Candidates

| ID | Name | Equation | Valid | Composite score | PINN status |
|---|---|---|---|---|---|
| H1 | Diffusion | `u_t = D*u_xx` | True | 0.835 | supported by the available observations and physics constraints |
| H2 | Advection | `u_t + c*u_x = 0` | True | 0.785 | rejected under the tested conditions |
| H3 | Reaction-diffusion (logistic growth/decay) | `u_t = D*u_xx + r*u*(1 - u/K)` | True | 0.757 | supported by the available observations and physics constraints |
| H4 | Advection-diffusion | `u_t + c*u_x = D*u_xx` | True | 0.750 | supported by the available observations and physics constraints |

**Structural match to true diffusion equation:** H1 (Diffusion)
