# Demo: from noisy observations to a validated, predictive law

**System used:** the Phase 7 generalization experiment, `experiments/10_generalization.py`. It runs the full pipeline on
a system the project was *not* developed on, from noisy, sparse data.

**Hidden truth:** `u_t + c·u_x = D·u_xx` with c = 0.4 and D = 0.02. The pipeline never sees these values; they are
used only for the final oracle check.

**Live time:**

| Step | Time |
|---|---|
| Steps 1–2 | 4 s |
| Step 3 (re-train one PINN live, optional) | about 2 min |
| Steps 4–6 (selection from cached models) | 9 s |

All outputs below are the actual outputs, re-produced byte-identically in the Phase 8 audit.

---

## Step 1 — Observation data

```bash
python experiments/10_generalization.py --stage prepare
```

The input is 400 random (x, t, u) points with 5% Gaussian noise. The system computes a description from the data
alone, using one fixed template that never names a mechanism:

```
- Peak amplitude went from 0.838 to 0.356: it decreased substantially.
- The amplitude-weighted spatial centroid went from x~0.42 to x~0.74: it moved by +21% of the
  domain length (the profile translates across the domain).
- The spatial spread broadened (the profile spreads out).
```

**Talking point:** both drift and spreading are visible in the data. Which law explains them is not.

## Step 2 — Candidate hypotheses (same command)

The description drives retrieval from the 5-document corpus. Retrieved evidence, KG templates and (mock) LLM
proposals are merged and deduplicated, then the structural and physics filters are applied.

```
Survivors: H1 u_t = D*u_xx | H2 u_t + c*u_x = 0 | H3 u_t = D*u_xx + r*u*(1-u/K) | H4 u_t + c*u_x = D*u_xx
```

Contrast with pure data-driven discovery on the same noisy data:

```
SINDy (noisy, reconstructed): u_t = 1.821*u - 0.365*u_x - 5.796*u^2     <- spurious
```

**Be honest about the source:** the LLM candidates come from a **mock** provider, so this step demonstrates the
pipeline, not LLM reasoning.

## Step 3 — Parameter estimation with the inverse PINN (optional live re-train)

```bash
python experiments/10_generalization.py --stage pinn --only H4     # ~2 min on CPU; or skip and use the cache
```

Each candidate's equation is compiled into a PINN residual, and its parameters are trained jointly with the network.
Training uses only t ≤ 0.8: 248 points for training and 62 held out for validation. The initial guesses are
c = 0.1 and D = 0.2.

```
H4: supported   params={'c': 0.39897, 'D': 0.02008}   val_rmse=0.0099
```

## Step 4 — Candidate rejection and PINN validation

```bash
python experiments/10_generalization.py --stage select
```

The gate uses **held-out noisy observations only**. Thresholds were fixed in `config.yaml` before the run.

| Candidate | Fitted | Held-out RMSE | Verdict |
|---|---|---|---|
| H1 diffusion | D = 0.075 | 0.0958 | **rejected**: cannot produce the drift |
| H2 advection | c = 0.369 | 0.0635 | **rejected**: gets the drift, cannot spread or decay |
| H3 reaction-diffusion | D = 0.117, r = 1.22 | 0.0841 | **rejected** |
| H4 advection-diffusion | c = 0.399, D = 0.0201 | **0.0099** (noise level) | **supported** |

**Talking point:** H2 is the dangerous wrong answer because it explains most of what you see, and the physics plus
held-out data still reject it.

## Step 5 — Selected equation

```
Selected: H4   u_t + c*u_x = D*u_xx   (same result under the original Phase 6 rule and the revised Phase 7 rule)
```

## Step 6 — OOD and novel prediction

| Model | OOD RMSE, t > 0.8 vs. noisy obs | vs. clean field (oracle, evaluation only) |
|---|---|---|
| **H4 PINN (selected)** | **0.0096** | **0.0010** |
| Data-only MLP (same architecture, no physics) | 0.0119 | 0.0068 |
| H1, H2, H3 | 0.10–0.12 | 0.09–0.11 |

- **Novel prediction:** the selected law is solved forward from the initial condition with no observations, to
  t = 1.3, which is beyond the data. Its relative RMSE against the hidden truth is **0.58%**.
- **Oracle parameter error:** c 0.26%, D 0.40%.

## What to say, and what not to say

- **Say:** the candidate was *supported by the available observations, physics constraints and OOD validation*. The
  wrong mechanisms were rejected by the data and the physics, not by an LLM's opinion.
- **Do not say:** "the AI discovered a new law" or "the LLM found the equation". The LLM is a mock, the law is a known
  textbook law, and the selection is parsimony among equations consistent with the data, not a proof of uniqueness.

## Backup demo (diffusion, Phase 6, about 20 s from cache)

```bash
python experiments/08_physics_constrained_search.py --stage prepare
python experiments/08_physics_constrained_search.py --stage select
```

This shows the nested-model case. H3 and H4 fit as well as H1, but their extra terms are inactive (r, K, c),
so H1 `u_t = D·u_xx` is selected with D̂ = 0.10002.
