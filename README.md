# Generative Scientific Discovery Engine

Given sparse, noisy observations of a field `u(x,t)` -- and **not** the governing equation -- the system
retrieves scientific background (RAG), organizes it into a knowledge graph, generates candidate governing
equations, compiles each into a physics-informed neural network (PINN), estimates unknown parameters by
solving the inverse problem, rejects candidates that cannot satisfy data + physics, selects the most
parsimonious supported candidate, and uses it to predict outside the observed region.

> RAG provides evidence; the knowledge graph provides structured relationships; the hypothesis engine
> generates candidate explanations; PINNs test physical consistency; model selection prefers the simplest
> supported candidate. Retrieval relevance, LLM confidence and PINN consistency are each **not** proof.

## Current status (all numbers below are from actual runs)

| | |
|---|---|
| Benchmark systems | diffusion (Phases 1-6); **12 datasets across 4 mechanism families** (Phase 7); **advection-diffusion from noisy sparse data with the PINN pipeline** (Phase 7) |
| Tests | `python -m pytest tests/ -v` -> **93 passed, 0 failed, 0 skipped** |
| Phase 6 | 5 candidates; 2 rejected; nested candidates collapse to `u_t = D*u_xx`, selected with `D_hat=0.10002` |
| Phase 7 | 50 questions **50/50**; 25 multi-hop **23/25**; 20 adversarial **20/20** (3 by abstention); ablation: full system 12/12 vs SINDy-alone 8/12 vs generation-without-validation 3/12; second system (advection-diffusion, 5% noise, 400 pts): correct structure, c and D within 0.4% |
| LLM | **mock provider only** (canned, deterministic). OpenAI provider implemented but not run: no key, and `api.openai.com` returns HTTP 403 in the build sandbox. No claim of LLM-driven discovery is made. |

## Directory structure

```
config.yaml            all experiment parameters and pre-fixed decision thresholds
data/                  Phase 1 generator (FD solver), problem-description builder; data/generated/*.npz
equation_discovery/    Phase 2: derivative estimation, candidate library, STLSQ/SINDy
pinn/                  Phase 3: reusable MLP, autograd derivatives, residual registry, losses, trainer
validation/            Phase 3/6/7: scorer, physics residual, OOD, model selection, stability, forward solver (Euler/RK4)
llm/                   Phase 4: provider abstraction (mock, Gemini), hypothesis schema, prompts
hypotheses/            Phase 4/6: equation compiler, validation, ranking, PINN adapter, constraints, complexity, search
rag/                   Phase 5: ingestion, chunking, embeddings, vector store, retrieval, evidence, context
knowledge_graph/       Phase 5: schema, graph, builder, queries, serialization
evaluation/            Phase 7: benchmark datasets (4 families), describer, identification, engine,
                       50+25 benchmark, 20 adversarial cases, ablation, leakage audit
experiments/           02-10, one script per phase (01 = data/generate_diffusion.py; 09-10 = Phase 7)
tests/                 one test module per phase
results/               equations/ (P2), pinn/ (P3), hypotheses/ (P4), rag_kg/ (P5), phase6/ (P6), evaluation/ (P7)
```

## Reproduce everything

```bash
pip install -r requirements.txt
python data/generate_diffusion.py --config config.yaml           # Phase 1 data
python experiments/02_equation_discovery.py --config config.yaml   # Phase 2 (seconds)
python experiments/03_forward_pinn.py --config config.yaml         # Phase 3 (~4 min each)
python experiments/04_inverse_pinn.py --config config.yaml
python experiments/05_hypothesis_validation.py --config config.yaml
python experiments/06_hypothesis_generation.py --config config.yaml --skip-pinn
python experiments/07_scientific_rag_kg.py --config config.yaml
python experiments/08_physics_constrained_search.py --config config.yaml   # Phase 6 (~20 min: 5 PINNs)
python experiments/10_generalization.py --config config.yaml              # Phase 7 second system (~16 min: 4 PINNs)
python experiments/09_evaluation.py --config config.yaml                  # Phase 7 suites + report (~1 min)
python -m pytest tests/ -v
```

---

# Final summary (Phase 8)

Full report: **`results/final_report.md`**. Demo script: **`results/demo/demo_run.md`**. Slides outline:
**`results/final_presentation_outline.md`**.

## Installation and configuration

```bash
python -m pip install -r requirements.txt   # numpy, scipy, matplotlib, pyyaml, pysindy, pytest, torch (CPU), sympy, pydantic, google-genai, networkx
```

- **Configuration:** every experiment parameter and every decision threshold lives in `config.yaml`, with sections
  `diffusion`, `sampling`, `discovery`, `llm`, `pinn`, `phase6`, `phase7`. Thresholds were fixed before each phase's run.
- **LLM provider:** `llm.provider: "gemini"` (model `llm.model`, default `gemini-3.8-flash`) is the default and requires
  `GEMINI_API_KEY`; if the key is missing it falls back to the mock, with a warning. `"mock"` forces the offline provider.
- **Runtime:** CPU only. Each inverse PINN takes about 2–4 minutes. PINN experiments support `--stage` and `--only`
  and cache results per candidate.

## Running

```bash
python -m pytest tests/ -v                                               # 93 tests, ~1-2 min
python experiments/09_evaluation.py --config config.yaml                 # Phase 7 suites + report, ~1 min
python experiments/10_generalization.py --stage prepare && \
python experiments/10_generalization.py --stage select                   # demo from cached PINNs, ~15 s
```

All experiments are listed under "Reproduce everything" above.

**Results by phase:**

| Location | Contents |
|---|---|
| `results/equations/` | Phase 2 |
| `results/pinn/` | Phase 3 |
| `results/hypotheses/` | Phase 4 |
| `results/rag_kg/` | Phase 5, including `knowledge_graph.json` |
| `results/phase6/` | Phase 6, including trained models in `models/*.pt` |
| `results/evaluation/` | Phase 7: benchmark, adversarial, ablation, leakage, real-LLM status, `generalization/` |
| `data/generated/` | Datasets |

## Architecture

```
Scientific data / literature
        ↓
RAG + Knowledge Graph          retrieve documented mechanisms (with provenance); KG supplies equation templates + parameters
        ↓
Hypothesis generation          LLM layer (MOCK ONLY) -> schema-validated JSON hypotheses
        ↓
Candidate equations            LLM proposals ∪ KG templates, structurally deduplicated (sign/parameter-name invariant)
        ↓
Equation discovery             SINDy: independent data-driven channel + baseline (never decides)
        ↓
Physics constraints            syntax/variables/derivatives; evolution-equation check; parameter bounds (D>0, K>0)
        ↓
PINN forward/inverse           sympy-compiled residual; one reusable MLP; unknown parameters trained jointly
        ↓
Model selection                gate on HELD-OUT NOISY data -> term necessity -> fit-equivalence -> complexity -> BIC
        ↓
OOD validation                 held-out t > 0.8 observations, never trained on or selected on
        ↓
Novel prediction               selected law solved forward beyond the observed window (validation/forward_solver.py)
```

## Final experimental summary

| Evidence | Result | Source |
|---|---|---|
| Original diffusion discovery | 5 candidates searched; 2 rejected; `u_t = D·u_xx` selected (BIC agrees); nested H3/H4 have inactive extra terms | `results/phase6/` |
| Inverse PINN parameter recovery | D̂ = 0.099888 (Phase 3), 0.10002 (Phase 6); c, D within 0.26% / 0.40% on the second system | `results/pinn/`, `results/phase6/`, `results/evaluation/generalization/` |
| Advection-diffusion generalization | Diffusion, advection and reaction-diffusion rejected; advection-diffusion selected under both rules; SINDy gave a spurious equation | `results/evaluation/generalization/` |
| OOD / novel prediction | Diffusion: OOD 0.0018 vs. MLP 0.0712 (oracle); beyond-window RMSE 0.00025. Advection-diffusion: OOD 0.0010 vs. MLP 0.0068; beyond-window 0.58% | same |
| Adversarial testing | 20/20 (3 by abstention; 1 requires the revised rule) | `adversarial_20.json` |
| 50-question benchmark | 50/50 (top-1 retrieval only 1/6) | `questions_50.json` |
| 25 multi-hop benchmark | 23/25 (both failures from lexical top-1 retrieval) | `multihop_25.json` |
| Ablation, correct model on 12 datasets | Full system 12/12; Phase 6 rule 10/12; SINDy 8/12; generation without validation 3/12; without LLM 12/12; without KG and LLM 12/12 | `ablation.json` |
| Leakage audit | 0 decision functions read ground truth; 0 of 15 hidden values in the corpus; audit catches a planted cheater | `leakage_and_reproducibility.json` |
| Reproducibility | 93/93 tests pass; all experiments re-run byte-identical (runtime fields excepted); 5 PINNs re-trained from scratch, identical | `results/final_report.md` §13 |

## Claim audit

| Topic | Verdict | Evidence |
|---|---|---|
| LLM-driven discovery | **Not demonstrated** | Mock provider only (`results/evaluation/real_llm.json`: available = false) |
| RAG contribution | **Partial:** recall only | Top-1 1/6. Retrieved evidence + validation is 12/12, the same as the full system. Without validation, 3/12 |
| KG contribution | **Partial:** structured queries and provenance; no accuracy gain | Multi-hop KG chains 8/8; ablation without KG 12/12; contributed the K3 candidate |
| Equation uniqueness | **Not demonstrated** | Nested models fit equally well; selection is a stated parsimony rule |
| Uncertainty | **Partial** | Empirical finite-difference bootstrap only; it disagrees with the PINN on H3 |
| Noisy-data robustness | **Partial** | The PINN path works at 5% noise on two systems. SINDy fails; the regression engine abstains (A17–A19) |
| Generalization | **Partial** | 4 families on clean data (regression engine); 1 additional system with noisy data (PINN). 1D synthetic only |
| BC/IC inference | **Not demonstrated** | Supplied as known physics |
| Real-world scientific data | **Not demonstrated** | Synthetic only |
| New physical law | **Not demonstrated / not claimed** | Recovery of known synthetic laws |

## Requirement matrix (original brief)

*Implemented* means the code exists; *Demonstrated* means an experiment supports it.

| Requirement | Implementation | Evidence | Status | Limitation |
|---|---|---|---|---|
| Scientific RAG + KG for evidence | `rag/`, `knowledge_graph/` | 22 entities and 40 relationships, all sourced; KG multi-hop 8/8 | Demonstrated (as infrastructure) | Lexical retrieval; 5-document fixture corpus; no accuracy gain |
| Generative hypotheses from multi-source evidence | `llm/`, `hypotheses/pipeline.py`, `hypotheses/search.py` | Pool = LLM ∪ KG templates; KG added K3 | Partial | Mock LLM; Mode A ≈ Mode B |
| PINN forward | `pinn/`, `experiments/03_forward_pinn.py` | RMSE 0.000727 | Demonstrated | 1D only |
| PINN inverse / unknown parameters | `pinn/`, `experiments/04_inverse_pinn.py`, `08_…`, `10_…` | D, c within 0.4% on two systems | Demonstrated | Two synthetic systems |
| PDE/ODE discovery from data | `equation_discovery/` | Clean/sparse recovered; clean benchmark 8/12 | Partial | Fails on noisy data |
| Physics-constrained generative search | `hypotheses/search.py`, `hypotheses/constraints.py` | Phase 6; generalization | Demonstrated | Small candidate pools |
| Sparse/noisy-data learning | PINN path | 400 points at 5% noise, two systems | Partial | One noise level |
| BC/IC inference | — | — | Not implemented | Given as known physics |
| Uncertainty estimation | `validation/stability.py` | Bootstrap reported | Partial | Not Bayesian; disagrees on H3 |
| OOD physics validation | `hypotheses/search.py`, `validation/forward_solver.py` | Both systems; benchmark 12/12 within 5% | Demonstrated | Short OOD windows |
| Comparison with NN and numerical solver | MLP baseline; forward solver | MLP 40× / 6.6× worse on OOD | Demonstrated | Single architecture |
| ≥3 hypotheses, ≥1 discovered law, rejected hypothesis | Phases 4, 6, 7 | 5 and 4 candidate pools; 2–3 rejected per run | Demonstrated | Known laws |
| Inverse + forward PINN, sparse/noisy, OOD experiments | Phases 3, 6, 7 | See above | Demonstrated | — |
| 20 adversarial cases | `evaluation/adversarial.py` | 20/20 | Demonstrated | 3 by abstention |
| 50 questions / 25 multi-hop | `evaluation/benchmark.py` | 50/50, 23/25 | Demonstrated | Several categories easy by construction |
| Ablation (LLM/RAG → +KG → PINN → full) | `evaluation/ablation.py` | 7 arms | Demonstrated | Regression engine on clean data; mock LLM |
| Code, datasets, models, equations, KG, experiments, report, reproduction | This repository | `results/`, `tests/` | Demonstrated | — |

## The selection-rule change (Phase 6 → Phase 7)

Phase 6 selected the lowest-complexity candidate within 5% of the best held-out error. Phase 7 found that this fails
on clean two-mode diffusion data. Finite-difference u_xx gives each sine mode a slightly wrong eigenvalue, and
reaction-diffusion's extra `u` term corrects both modes exactly, so it fits the *discretization error* to about 1e-14.
Its reaction mechanism contributes only 0.27% of the dynamics: numerically supported, dynamically negligible.

The revised rule promotes Phase 6's own term-necessity check (5% threshold, fixed in Phase 6) into selection. It is
opt-in, so Phase 6 outputs are untouched. Every table reports both rules: 10/12 vs. 12/12 on the benchmark, and the
same answer on both PINN runs. Full explanation: `results/final_report.md` §12.

## Real LLM status

**Not run.** There is no `OPENAI_API_KEY`, and the build sandbox's proxy blocks `api.openai.com` (HTTP 403). Every LLM
candidate in every result comes from the deterministic mock provider. `experiments/09_evaluation.py --real-llm` exists
for when access is available, and a test enforces that no LLM results are recorded while it is unavailable.

The project now uses Gemini as its only real LLM provider (`llm/gemini_provider.py`). To produce real-LLM results, set
`GEMINI_API_KEY` and run `python experiments/09_evaluation.py --config config.yaml --real-llm`.

## Limitations

1. Mock LLM only.
2. Lexical retrieval over a 5-document corpus.
3. Synthetic 1D PDEs only.
4. BC/IC are given, not inferred.
5. Bootstrap-only uncertainty.
6. No uniqueness.
7. Thresholds are hand-set.
8. The fast regression engine has no noise robustness.

Details: `results/final_report.md` §14.

---

# Development log (Phases 1–7)

The sections below were written as each phase was completed and are kept as the project record. Numbers in them
are correct as of their phase. Statements superseded later are marked in place. The final, audited numbers are in
the summary above.

---

# Phase 1 — Synthetic Data

**Milestone:** synthetic 1D diffusion dataset (clean, sparse, sparse+noisy),
verified. This is the ground-truth generator that every later phase
(equation discovery, PINN, inverse problem) will consume.

Nothing here does discovery yet — this phase only produces and verifies
the observational data that the rest of the system is not allowed to see
the answer for.

## What's here

```
scientific-discovery-engine/
├── config.yaml                    # single source of truth for all parameters
├── requirements.txt
├── data/
│   ├── generate_diffusion.py      # FD solver + sparse/noisy sampling
│   └── generated/                 # created by running the script (gitignored in practice)
│       ├── full.npz
│       ├── sparse.npz
│       ├── noisy.npz
│       └── verification_plot.png
└── tests/
    └── test_diffusion_data.py     # sanity checks + verification plot
```

## The physics (hidden from the discovery system)

Ground truth: `u_t = D * u_xx`, with `D = 0.1`, on `x in [0,1]`,
`t in [0,1]`, Dirichlet-zero boundaries, and initial condition
`u(x,0) = sin(pi*x) + 0.5*sin(3*pi*x)` (a two-mode combination — see
"Why two modes, not one" below). Solved with an explicit
finite-difference scheme — this solver also doubles as the
numerical-solver baseline used later in the project.

### Why two modes, not one

The first version of this generator used a single sine mode,
`sin(pi*x)`. That choice turned out to be a hidden trap: `sin(pi*x)`
is an **eigenfunction** of `d^2/dx^2` on `[0,1]` with Dirichlet
boundaries, satisfying `u_xx = -pi^2 * u` exactly at every point. That
makes the fields `u` and `u_xx` perfectly collinear in the resulting
dataset — no regression method, however good, can structurally tell
`D*u_xx` apart from `-D*pi^2*u`, because in this special-case dataset
they are literally the same signal. This was caught empirically in
Phase 2 (equation discovery confidently — and wrongly — selected `u`
instead of `u_xx`).

The fix, used for every experiment from Phase 2 onward, is the
`two_mode_sin` initial condition, `sin(pi*x) + 0.5*sin(3*pi*x)`: a mix
of two Laplacian eigenmodes with *different* eigenvalues (`-pi^2` and
`-9pi^2`). It still satisfies the Dirichlet boundary conditions
exactly (both terms vanish at `x=0,1`), so nothing else about the
setup changes. The single-mode `"sin"` option is kept in
`data/generate_diffusion.py` and documented there as the degenerate
case, for reference and as a cautionary example — not used as the
default.

Three artifacts come out of this:

- **`full.npz`** — the entire clean field (61 x-points × 800 t-points).
  For plotting and grading only.
- **`sparse.npz`** — a noise-free random subsample of the full field
  (~10% of grid points). Not used yet, kept for later ablations that
  isolate the effect of noise vs. sparsity separately.
- **`noisy.npz`** — 400 randomly sampled points with Gaussian noise
  added (std = 5% of the clean field's std). **This is the only file
  the discovery system is allowed to see** — it contains `x`, `t`, `u`
  only, no `D`, no `u_clean` (those are present for evaluation code,
  not for the discovery pipeline itself).

## Install

```bash
cd scientific-discovery-engine
pip install -r requirements.txt
```

## Generate the dataset

```bash
python data/generate_diffusion.py --config config.yaml
```

Expected output:

```
Generated dataset in: .../data/generated
  full.npz   : full field, shape U=(800, 61) (n_t x n_x)
  sparse.npz : 4880 noise-free (x,t,u) points
  noisy.npz  : 400 noisy (x,t,u) points (noise_std=0.01159)
  D_true = 0.1 (hidden from discovery system)
```

## Verify it

```bash
python tests/test_diffusion_data.py --config config.yaml
```

This checks: files exist, shapes are consistent, boundaries are ~0,
the field actually decays over time (i.e. it's really diffusing, not
garbage), the sparse/noisy sets are genuinely smaller than the full
grid, and the injected noise magnitude matches what was configured.
It then saves `data/generated/verification_plot.png` with three
panels (full field heatmap, sparse points, noisy points) so you can
eyeball that the noisy scatter still visually resembles the same
decaying-peak pattern as the clean field.

All checks currently pass; the plot shows the characteristic double-hump
shape of `sin(pi*x) + 0.5*sin(3*pi*x)` decaying and smoothing into a
single broad hump over time, as expected for diffusion (the higher
spatial-frequency mode decays faster than the fundamental, since decay
rate scales with the square of the mode number).

## Next step at the time of Phase 1 (done in Phase 2)

Feed `noisy.npz` (x, t, u only) into a sparse-regression / SINDy-style
equation discovery routine and check whether it recovers
`u_t ≈ D u_xx` with `D ≈ 0.1`, without ever being told the equation.

---

# Phase 2 — Equation Discovery

**Milestone:** recover `u_t ≈ D·u_xx` from observational data alone,
using sparse regression (SINDy-style STLSQ), and honestly report where
it works and where it doesn't.

## What's new

```
equation_discovery/
├── derivatives.py       # grid reconstruction, denoising, finite differences
├── library.py           # modular candidate-term library (not diffusion-specific)
├── sindy.py              # STLSQ sparse regression (uses pysindy, falls back to manual)
└── discovery_engine.py   # orchestration, equation formatting, evaluation, hypothesis scoring
experiments/
└── 02_equation_discovery.py   # runs clean/sparse/noisy, saves results + plot
tests/
└── test_equation_discovery.py
results/equations/
├── phase2_discovery_report.json
├── phase2_results_table.md
└── phase2_coefficients.png
```

## An important mid-phase finding: the original Phase 1 data was undiscoverable

The Phase 1 default initial condition was a single sine mode,
`sin(pi*x)`. That's an **eigenfunction** of the second-derivative
operator on this domain: `u_xx = -pi^2 * u` exactly. That makes the
columns for `u` and `u_xx` in the candidate library perfectly
collinear — no amount of data, and no regression method, can tell
`D*u_xx` apart from `-D*pi^2*u` from this dataset, because they are
literally the same signal. This showed up immediately as STLSQ
confidently discovering `u_t = -0.987*u` — numerically a fine fit, but
structurally the wrong term.

**Fix:** `config.yaml`'s `diffusion.initial_condition` is now
`two_mode_sin` (`sin(pi*x) + 0.5*sin(3*pi*x)`), a mix of two Laplacian
eigenmodes with different eigenvalues. This still satisfies the
Dirichlet boundary conditions exactly and required no other change to
the Phase 1 pipeline. **`data/generated/*.npz` were regenerated** with
this initial condition (Phase 1's `tests/test_diffusion_data.py` was
re-run and still passes in full). The single-mode `"sin"` option is
kept in the code, documented as degenerate, for reference.

This is reported here rather than fixed silently because it's exactly
the kind of failure the assignment asks to surface: the method can look
like it's working (great data fit, tiny residual) while discovering
the wrong physics, for a purely structural, data-design reason.

## Method

1. **Clean data** (`full.npz`): already a regular grid — apply central
   finite differences directly via `np.gradient`.
2. **Sparse / noisy data** (`sparse.npz`, `noisy.npz`): scattered
   points, not on a grid. Reconstruct onto a regular grid via
   `scipy.interpolate.griddata` (linear, nearest-neighbor fallback at
   the edges), then finite-difference. The reconstruction grid
   resolution is matched to how many points are actually available —
   4880 sparse points support a 41×150 grid; 400 noisy points do not,
   and use a 13×25 grid instead (tuned empirically against sample
   density, **not** against the known answer).
3. For noisy data, an optional Gaussian smoothing pass denoises the
   reconstructed field before differentiation (numerical
   differentiation amplifies noise — see `derivatives.py` docstring
   for a fuller discussion of this and other limitations).
4. Build a fixed 8-term candidate library: `1, u, u_x, u_xx, u_xxx,
   u^2, u*u_x, u*u_xx` (`library.py` — fully generic, any field/term
   combination works, see the advection test below).
5. Fit `u_t ≈ Theta @ xi` via STLSQ (`sindy.py`), using `pysindy`'s
   `STLSQ` optimizer directly, with a self-contained fallback
   implementation of the same algorithm if pysindy is unavailable.
6. Compare the discovered equation against the (separately retained)
   hidden ground truth: `D_true = 0.1`, `true_term = "u_xx"`.

## Run it

```bash
# (re)generate Phase 1 data with the fixed initial condition, if needed
python data/generate_diffusion.py --config config.yaml
python tests/test_diffusion_data.py --config config.yaml

# Phase 2
python experiments/02_equation_discovery.py --config config.yaml
python -m pytest tests/test_equation_discovery.py -v
```

## Results

| Dataset | Discovered Equation | D estimate | Rel. param error | Structure recovered |
|---|---|---|---|---|
| Clean | `u_t = 0.1009*u_xx` | 0.1009 | 0.88% | **True** |
| Sparse | `u_t = 0.1013*u_xx` | 0.1013 | 1.26% | **True** |
| Noisy (no smoothing) | `u_t = 0.5039*1 - 3.418*u + 4.147*u^2 + 0.05336*u*u_x + 0.1676*u*u_xx` | 0.0000 | 100% | False |
| Noisy (smoothed) | `u_t = 0.545*u + 0.1968*u_xx - 0.5059*u^2 + 0.06862*u*u_x - 0.1077*u*u_xx` | 0.1968 | 96.8% | False |

**Clean and sparse data: the key demonstration works.** Full 8-term
STLSQ correctly zeroes out every term except `u_xx`, and estimates
`D` within ~1-1.3% of the true 0.1, with no knowledge of the equation.

**Noisy data (400 points, 5% noise): honest failure of the full
8-term library.** Neither raw nor smoothed reconstruction gets a
clean answer via unrestricted STLSQ — this is a real, reported limit
of derivative-based discovery under heavy sparsity + noise, not
something hidden.

**However**, restricting to single-term hypotheses (Step 9, below)
shows the *evidence* is actually there — it's the 8-term search that
struggles, not the underlying signal:

| Dataset | H1: `u_t = a·u_xx` (R², D_est) | H2: `u_t = b·u_x` (R², estimate) |
|---|---|---|
| Clean | R²=1.000, D=0.1009 | R²=-0.56 (worse than the mean) |
| Sparse | R²=0.994, D=0.1013 | R²=-0.68 |
| Noisy, no smoothing | R²=0.787, D=0.1068 (6.8% err) | R²=-1.30 |
| Noisy, smoothed | R²=0.813, D=0.1109 (10.9% err) | R²=-1.75 |

H2 (advection) has *negative* R² everywhere — worse than predicting
the mean — in every dataset including noisy ones. **This is the
"failed hypothesis correctly rejected" deliverable**: the data clearly
and consistently disfavor advection relative to diffusion, even when
the full unrestricted search struggles with noise.

Smoothing raises H1's R² slightly (0.787→0.813) but does NOT uniformly
improve the point estimate of D (6.8%→10.9% error) — smoothing trades
some variance for bias here. This is reported rather than cherry-picked.

See `results/equations/phase2_coefficients.png` for a bar chart of every
discovered coefficient across all four runs, and
`results/equations/phase2_discovery_report.json` for full numeric detail.

## Tests

```
tests/test_equation_discovery.py::test_clean_diffusion_recovery PASSED
tests/test_equation_discovery.py::test_generality_on_advection PASSED
tests/test_equation_discovery.py::test_stlsq_recovers_known_sparse_signal PASSED
tests/test_equation_discovery.py::test_library_shapes_and_labels PASSED
```

`test_generality_on_advection` is the proof of the "must not be
diffusion-specific" requirement: it builds a completely different,
in-memory dataset for `u_t = -c*u_x` (pure advection, no diffusion
term at all) and confirms the *same* `discovery_engine`/`library`/
`sindy` code correctly recovers `u_x` with coefficient `-c` (0.67%
error) instead of `u_xx` — nothing in `equation_discovery/` had to
change.

## Limitations (read before Phase 3)

- **Full-library STLSQ on the noisy dataset does not cleanly recover
  the equation.** 400 points with 5% noise, run through interpolation
  + finite differencing, is a genuinely hard regime for
  derivative-based discovery. This is a known, published weakness of
  SINDy-family methods (motivating, e.g., weak-form / integral
  formulations we have not implemented here for time reasons).
- **Reconstruction-grid resolution was hand-tuned** against sample
  density (roughly matching grid node count to sample count), not
  against a validation set. A more principled version would
  cross-validate this.
- **Third derivative (`u_xxx`) estimates are unreliable on noisy
  data** — three chained finite differences compound noise
  amplification substantially. It never gets selected here, correctly,
  but should not be trusted quantitatively in this experiment.
- **The competing-hypothesis scores (Step 9) are not yet a validation
  step** — they're regression diagnostics (R², coefficient), useful as
  evidence, but rejecting/accepting a hypothesis on physical grounds
  (residuals, conservation, dimensional consistency) is explicitly
  deferred to the PINN/validation stage in a later phase, per the
  assignment.

---

# Phase 3 — PINN Validation

**Milestone:** connect candidate equations from Phase 2 to a
Physics-Informed Neural Network, use it to actually test hypotheses
against data + physics simultaneously (not just regression
diagnostics), estimate an unknown parameter, and check
out-of-distribution generalization.

## What's new

```
pinn/
├── network.py       # reusable MLP: (x,t) -> u_theta(x,t) — same class for every hypothesis
├── derivatives.py    # torch.autograd u_t, u_x, u_xx (no finite differences)
├── residuals.py       # pluggable residual registry: diffusion, advection, reaction_diffusion
├── losses.py           # L_data, L_physics, L_bc, L_ic, weighted sum
├── trainer.py           # generic training loop (Adam -> L-BFGS), used for every run below
├── forward.py            # thin wrapper: physics params FIXED
├── inverse.py             # thin wrapper: physics params TRAINABLE
└── data_utils.py           # builds data/collocation/BC/IC tensors (shared helper, not in the original file list)
validation/
├── scorer.py         # numeric-thresholds-only accept/reject, see below
├── physics_validation.py  # PDE residual on a fresh grid (not training collocation points)
└── ood.py             # train/OOD time split + prediction-error helper
experiments/
├── 03_forward_pinn.py
├── 04_inverse_pinn.py
└── 05_hypothesis_validation.py
tests/
└── test_pinn.py
results/pinn/
├── phase3_forward_report.json, phase3_forward_plot.png
├── phase3_inverse_report.json, phase3_inverse_plot.png
└── phase3_hypothesis_validation_report.json, phase3_validation_table.md,
    phase3_ood_comparison.png, phase3_hypothesis_training_curves.png
```

## Architecture

One `PINN` class (`hidden_dims=[64,64,64,64]`, tanh activations,
Xavier init) maps `(x,t) -> u_theta(x,t)`. It is never subclassed or
duplicated per hypothesis — what changes between diffusion and
advection is only the **residual function** pulled from
`pinn/residuals.py`'s registry:

```python
RESIDUAL_REGISTRY = {
    "diffusion":  {"fn": lambda d,p: d["u_t"] - p["D"]*d["u_xx"], "equation_str": "u_t = D*u_xx"},
    "advection":  {"fn": lambda d,p: d["u_t"] - p["c"]*d["u_x"],  "equation_str": "u_t = c*u_x"},
    "reaction_diffusion": {"fn": lambda d,p: d["u_t"] - p["D"]*d["u_xx"] + p["k"]*d["u"], ...},  # included, unused
}
```

`u_t`, `u_x`, `u_xx` all come from `torch.autograd.grad` on the
network output (`pinn/derivatives.py`) — no finite differences appear
anywhere in the PINN code path.

## Loss formulation

```
L = lambda_data    * mean((u_theta(x_i,t_i) - u_i)^2)        [observed (x_i,t_i,u_i), noisy.npz]
  + lambda_physics * mean(residual(x,t)^2)                    [random collocation points, no u needed]
  + lambda_bc       * mean((u_theta(0,t)-0)^2 + (u_theta(1,t)-0)^2)
  + lambda_ic       * mean((u_theta(x,0) - [sin(pi x)+0.5 sin(3 pi x)])^2)
```

All four weights (`lambda_data/physics/bc/ic`) are `1.0` by default
and configurable in `config.yaml: pinn.weights`. Training is Adam
(3000 epochs, lr=1e-3) followed by a short L-BFGS polish (300 iters) —
standard PINN practice, since L-BFGS converges much further once Adam
gets close.

## Run it

```bash
python experiments/03_forward_pinn.py --config config.yaml
python experiments/04_inverse_pinn.py --config config.yaml
python experiments/05_hypothesis_validation.py --config config.yaml
python -m pytest tests/test_pinn.py -v
```

Each of the three experiment scripts takes 2-4 minutes on CPU.

## Forward PINN results (D known = 0.1)

Trained on the 400 noisy observations + known physics/BC/IC. Evaluated
against the clean reference field (never used in training):

| Metric | Value |
|---|---|
| Data loss | 0.000132 |
| Physics residual (fresh grid) | 0.0000266 |
| BC loss | 0.0000024 |
| IC loss | 0.0000008 |
| **Prediction RMSE vs. clean field** | **0.000727** |

The PINN's physics constraint effectively denoises the sparse, noisy
observations — 0.07% RMSE against the true field despite training data
having ~5% injected noise.

## Inverse PINN results (D unknown, initialized to 0.2)

| | Value |
|---|---|
| D_init | 0.2 |
| **D_hat** | **0.09989** |
| D_true | 0.1 |
| Absolute error | 0.000112 |
| **Relative error** | **0.112%** |

D converges monotonically from 0.2 toward 0.1 within the first ~1500
epochs (see `phase3_inverse_plot.png`), then holds steady through the
L-BFGS polish — a clean instance of the "unknown parameter estimation
using PINNs" requirement.

## Hypothesis validation: H1 (diffusion) vs. H2 (advection)

Both trained identically (same architecture, same weights, same noisy
observations restricted to `t<=0.7`, same BC, same IC), with their
physics parameter **trainable** in both cases — H2 was given every
chance to fit by adjusting `c` freely:

| Hypothesis | Equation | Status | Data loss | Physics residual | BC loss | IC loss | In-dist RMSE | OOD RMSE |
|---|---|---|---|---|---|---|---|---|
| H1_diffusion | `u_t = D*u_xx` | **supported by the available observations and physics constraints** | 0.00013 | 0.00008 | 0.000002 | 0.000001 | 0.00078 | 0.00115 |
| H2_advection | `u_t = c*u_x` | **rejected under the tested conditions** | 0.03775 | 0.00499 | 0.000016 | 0.02515 | 0.20033 | 0.33145 |

**H1 wins on every single metric, by 30-290x**, without ever being
told the correct equation — its D parameter (also trainable, also
initialized away from 0.1) converges to 0.0999. **H2's IC loss alone
is 25,000x worse than H1's** — a straight line/plane-wave solution
of the advection equation simply cannot reproduce a decaying two-mode
sine initial condition, and the optimizer cannot find a value of `c`
that reconciles this with the data. This is the "failed hypothesis
correctly rejected" deliverable, now demonstrated via an actual PINN
constraint rather than Phase 2's regression diagnostic.

`phase3_ood_comparison.png` shows this visually: H1's prediction is
visually indistinguishable from the clean reference on both sides of
the train/OOD split (white dashed line at t=0.7); H2 converges to a
completely different (roughly x-only) steady pattern that never
resembles the true decaying double-hump field, in or out of
distribution.

Status strings deliberately avoid overclaiming — "supported by the
available observations and physics constraints" / "rejected under the
tested conditions", never "proven" — per the assignment's requirement
that the validation engine, not an LLM, decides status from fixed
numeric thresholds (`config.yaml: pinn.validation.thresholds`), set
before looking at these results.


> **Audit note (added in Phase 6):** the accept/reject status in the Phase 3–5 experiments includes a
> `prediction_error` check measured against the *clean ground-truth field*. Training never used that field,
> but the decision rule did. Re-checking the numbers, no outcome depends on it: H2 also fails `data_loss`
> (0.038 > 0.01) and `ic_loss` (0.025 > 0.01), and H1 passes every other check. Phase 6 removes the
> issue: its gate and model selection use only held-out *noisy observations*, and the clean field is used
> afterwards purely to report benchmark error.

## OOD test

Training used only `t<=0.7` (278 of 400 noisy points); evaluation
compared against the clean reference on `t<=0.7` (34,160 points,
in-distribution) and `t>0.7` (14,640 points, OOD, never seen in
training). H1's OOD RMSE (0.00115) is actually *very close to* its
in-distribution RMSE (0.00078) — i.e., the physics constraint lets it
extrapolate almost as well as it interpolates, exactly what a
correctly-constrained PINN should do. H2's OOD RMSE (0.331) is even
worse than its already-poor in-distribution RMSE (0.200), i.e. it gets
worse, not just "already bad", when extrapolated.

## Tests

```
tests/test_pinn.py::test_network_output_shape PASSED
tests/test_pinn.py::test_autograd_derivatives_match_analytic PASSED
tests/test_pinn.py::test_residual_registry PASSED
tests/test_pinn.py::test_trainer_smoke PASSED
tests/test_pinn.py::test_phase3_reported_results PASSED
```

`test_autograd_derivatives_match_analytic` checks `pinn/derivatives.py`
against a closed-form function with known derivatives (isolating the
autograd machinery from training convergence). `test_trainer_smoke` is
a fast (100-epoch) synthetic run confirming the training loop itself
works. `test_phase3_reported_results` re-checks the actual saved
experiment numbers above against fixed tolerances.

## Limitations

- **Single physical system** *(as of Phase 3; superseded — Phase 7 tests 4 mechanism families and a second system with the PINN pipeline)*. Only diffusion (correct) vs. advection
  (wrong) have been tested end-to-end. `reaction_diffusion` is
  registered in `pinn/residuals.py` to demonstrate the interface
  generalizes, but has not been run.
- **Validation thresholds are hand-set**, not calibrated against a
  labeled validation set of known-good/known-bad hypotheses (that's
  effectively what the Phase 4 benchmark + adversarial cases will
  provide).
- **The OOD test is a time-extrapolation split only** (a different
  initial condition, as the spec allowed as an alternative, hasn't
  been tried).
- **Data loss uses the noisy observations directly** (not a
  Phase-2-style denoised reconstruction) — the PINN's physics term is
  doing the effective denoising here, which is realistic but means
  forward/inverse results would degrade further under even higher
  noise than tested.
- **BC points are drawn across the full `t` range even in the OOD
  experiment.** This is a deliberate, stated choice (a boundary
  condition is domain knowledge, not a future-time measurement) but is
  worth flagging as a modeling decision, not a data leak that was
  overlooked.

---

# Phase 4 — Generative Hypothesis Engine

**Milestone:** a problem description goes in, multiple structurally
different candidate equations come out in a validated schema, and
(optionally) each one is run through the *existing, unmodified* Phase
3 PINN validator — closing the loop
`observations -> hypotheses -> equations -> PINN -> accept/reject`.

## Why this phase exists

Phases 1-3 could only test equations a human typed into
`pinn/residuals.py` by hand. Phase 4 adds a generator that proposes
*which* equations are worth testing in the first place, from a
description of what was observed — without ever being told the
answer.

## LLM = hypothesis generator. PINN = physics-based validator.

This split is enforced architecturally, not just by convention:

- The LLM (`llm/`) can only produce a `Hypothesis` object (see schema
  below). It cannot mark anything "correct" — there is no field in the
  schema for that, only a self-reported `confidence` (explicitly
  documented as subjective, not scientific validity).
- `hypotheses/ranking.py`'s composite score decides **investigation
  order only**. Its docstring states outright: a top-ranked hypothesis
  can still be rejected by validation, and a bottom-ranked one could
  in principle be supported.
- The **only** place that produces an accept/reject verdict is
  `validation/scorer.py` (built in Phase 3, reused unchanged here),
  comparing fixed numeric thresholds against measured PINN losses.
  Status strings are "supported by the available observations and
  physics constraints" / "rejected under the tested conditions" —
  never "proven" or "correct".

## What's new

```
llm/
├── base.py            # LLMProvider ABC: generate(prompt, system, temperature) -> str
├── mock_provider.py     # deterministic, offline, no API key -- used by all tests
├── openai_provider.py    # real provider; reads OPENAI_API_KEY/OPENAI_MODEL from env, lazy import
├── schemas.py             # Hypothesis / ParameterSpec / HypothesisSet (Pydantic)
├── prompts.py              # SYSTEM_PROMPT + build_hypothesis_prompt() -- not embedded in experiments/
└── __init__.py              # get_provider(cfg) factory, falls back to mock with a warning
hypotheses/
├── equation_compiler.py   # sympy: equation string -> torch residual_fn (see below)
├── validation.py            # syntax/variable/parameter/derivative/dimensional checks + dedup
├── ranking.py                 # preliminary triage score (NOT a validity judgment, see above)
├── adapter.py                  # hypothesis_to_pinn_config(hypothesis) -- the ONLY Phase4->Phase3 bridge
└── pipeline.py                  # generate_hypotheses(): prompt -> LLM -> schema -> validate -> dedup -> rank
experiments/06_hypothesis_generation.py
tests/test_hypothesis_generation.py
results/hypotheses/
├── phase4_generated_hypotheses.json
└── phase4_hypothesis_report.md
```

`pinn/derivatives.py` gained one addition: `u_xxx` is now computed
(one extra `autograd.grad` call), so the generic compiler below can
support third-derivative terms if a future hypothesis needs them.
`pinn/residuals.py`'s `make_params()` was extended to accept a
*per-parameter* trainable dict (previously only a single bool for
all parameters) — a backward-compatible generalization needed because
an LLM-generated hypothesis might declare some parameters trainable
and others fixed. Nothing else in `pinn/` or `validation/` changed.

## The key design choice: a generic equation compiler, not a bigger registry

Phase 3's `RESIDUAL_REGISTRY` requires hand-writing a Python lambda for
every equation. That doesn't scale to LLM-proposed equations. Instead,
`hypotheses/equation_compiler.py` **parses the hypothesis's own
equation string with `sympy`** and compiles it directly into the exact
`residual_fn(derivs, params)` shape `pinn/trainer.py` already expects:

```python
compile_residual("u_t = D*u_xx + r*u*(1-u/K)", ["D", "r", "K"])
# -> residual_fn, used_fields=['u','u_t','u_xx'], used_params=['D','r','K'], expr
```

`used_fields` is auto-detected from the parsed expression — this is
also how derivative validation works (see below): the equation itself
is the source of truth, not a hand-typed list. This covers every
example in the assignment (diffusion, advection, advection-diffusion,
reaction-diffusion) **with zero code changes** — only a new
`Hypothesis` object is needed for a new equation. Verified directly:

| Equation | Auto-detected fields | Auto-detected params |
|---|---|---|
| `u_t = D*u_xx` | `u_t, u_xx` | `D` |
| `u_t + c*u_x = 0` | `u_t, u_x` | `c` |
| `u_t + c*u_x = D*u_xx` | `u_t, u_x, u_xx` | `c, D` |
| `u_t = D*u_xx + r*u*(1-u/K)` | `u, u_t, u_xx` | `D, r, K` |

Limitation, stated honestly: this handles algebraic combinations of
`{u, u_t, u_x, u_xx, u_xxx}` and named parameters — no integral/
non-local terms, no 4th+ derivatives, no systems of multiple dependent
variables. That covers everything this project has needed so far.

## Hypothesis schema

(`llm/schemas.py`, Pydantic.) Every field the Phase 4 spec's example
showed is present, plus two Phase-3-facing additions: a required `=`
in `equation` (checked at construction time) and a `confidence` range
check `[0,1]`. Fields are typed and required where it matters (`id`,
`name`, `equation` are mandatory; `parameters`, `assumptions`,
`testable_predictions` default to empty lists rather than `None` so
downstream code never null-checks them).

## Example generated hypotheses (mock provider)

Problem description handed to the generator (built from **actually
computed statistics** of `noisy.npz` — never the equation or `D`):

> *"...peak absolute amplitude is approximately 0.983 near t=0.05... \
approximately 0.395 near t=0.95 (amplitude has decreased \
substantially)... near the domain edges, the field stays very close \
to zero... amplitude-weighted spatial centroid is x~0.53 near t=0.05 \
and x~0.41 near t=0.95 (little evidence of bulk translation)... \
spatial profile appears to smooth out over time..."*

| ID | Name | Equation | Confidence (self-reported) |
|---|---|---|---|
| H1 | Diffusion | `u_t = D*u_xx` | 0.55 |
| H2 | Advection | `u_t + c*u_x = 0` | 0.30 |
| H3 | Reaction-diffusion (logistic) | `u_t = D*u_xx + r*u*(1-u/K)` | 0.35 |
| H4 | Advection-diffusion | `u_t + c*u_x = D*u_xx` | 0.25 |

All 4 are structurally distinct mechanisms (not 4 coefficients of the
same equation), matching the "candidate diversity" requirement.
**IMPORTANT HONESTY NOTE on the mock provider**: `MockLLMProvider`
returns this same canned, generic set of 4 transport mechanisms
regardless of prompt content — it is a deterministic stand-in for
testing the rest of the pipeline (schema validation, dedup, ranking,
the PINN adapter) without a network or API key, **not** a
demonstration of real LLM reasoning. Set `llm.provider: "openai"` and
`OPENAI_API_KEY` in your environment to get genuine LLM-generated
hypotheses from the same prompt; `llm/openai_provider.py` is fully
implemented but was not exercised in this environment (no outbound
network access to `api.openai.com` here, and no key configured).

## Example JSON schema output (one hypothesis, abbreviated)

```json
{
  "id": "H1",
  "name": "Diffusion",
  "domain": "transport",
  "equation": "u_t = D*u_xx",
  "latex": "u_t = D u_{xx}",
  "dependent_variables": ["u"],
  "independent_variables": ["x", "t"],
  "parameters": [
    {"name": "D", "description": "diffusion coefficient", "trainable": true,
     "initial_value": 0.2, "units": "length^2/time"}
  ],
  "required_derivatives": ["u_t", "u_xx"],
  "required_initial_conditions": true,
  "required_boundary_conditions": true,
  "assumptions": ["isotropic medium", "no advective transport", "no source/sink terms"],
  "mechanism": "diffusive transport",
  "rationale": "A field that spreads out and decays smoothly over time...",
  "testable_predictions": ["spatial variance grows linearly in time", "..."],
  "confidence": 0.55
}
```

## Deterministic validation (before any PINN time is spent)

Run automatically inside `generate_hypotheses()`, with real results
from the run above:

1. **Syntax** — can `sympy` parse the equation? (all 4 passed)
2. **Variables** — is `u` in `dependent_variables`, are `x,t` in
   `independent_variables`? (all 4 passed)
3. **Parameters** — does every parameter the equation uses appear in
   `hypothesis.parameters`, and vice versa? (all 4 passed, 0 warnings)
4. **Derivatives** — do the LLM's *stated* `required_derivatives`
   match what `sympy` actually finds in the equation? (all 4 matched
   exactly — 0 mismatch warnings)
5. **Dimensional metadata** — fraction of parameters with a `units`
   string. All 4 candidates had units for every parameter in this run
   (`dimensional_completeness=1.0`); this is reported as a fraction,
   never claimed as a complete dimensional-consistency proof.
6. **Duplicates** — `hypotheses/validation.py:canonical_signature()`
   substitutes every parameter with a generic placeholder (so `D` vs
   `k` doesn't evade detection) and is sign-invariant (`u_t=D*u_xx`
   and `D*u_xx=u_t` collide on purpose). 0 duplicates in this run
   (verified separately in `tests/test_hypothesis_generation.py` with
   a deliberately-renamed duplicate).

## Connecting to Phase 3 — and a real finding from doing it

`hypotheses/adapter.py:hypothesis_to_pinn_config()` is the only new
code that touches both layers; it compiles the equation and returns a
dict with the exact shape of a hand-written `RESIDUAL_REGISTRY` entry,
so `pinn/trainer.py:build_and_train()` runs it completely unmodified.
All 4 candidates were run through the *same* training setup as
Phase 3 (noisy observations, known BC/IC, 3000 Adam + 300 L-BFGS
epochs):

| ID | Equation | PINN status | Fitted parameters |
|---|---|---|---|
| H1 | `u_t = D*u_xx` | **supported** | D=0.09989 |
| H2 | `u_t + c*u_x = 0` | **rejected** | c=0.01641 |
| H3 | `u_t = D*u_xx + r*u(1-u/K)` | **supported** | D=0.10063, r=0.01478, K=1.215 |
| H4 | `u_t + c*u_x = D*u_xx` | **supported** | c=-0.00036, D=0.09999 |

**This surfaced a genuine, important limitation, not a bug:** H3 and
H4 both *structurally contain* diffusion as a special case
(reaction-diffusion with `r=0`; advection-diffusion with `c=0`). Their
extra parameters converge almost exactly to the "turn this term off"
value (`r≈0.015`, `c≈-0.0004`), so they fit the data and satisfy their
own (now-trivial) physics constraint just as well as H1 does, and all
three pass the same fixed thresholds. **A threshold-based PINN
validator alone cannot penalize harmless extra complexity that nests
the true equation as a special case** — only H2 (which cannot reduce
to diffusion through any parameter limit) was cleanly rejected (30-250x
worse on every metric). This is exactly why `hypotheses/ranking.py`
includes a simplicity term (fewer free parameters scores higher):
among the three "supported" hypotheses, H1 already ranks first
(composite score 0.835 vs. 0.757 and 0.750) purely from being the
simplest, before any PINN was even run. The validation layer answers
"is this consistent with the data and physics?"; distinguishing
"minimal correct model" from "unnecessarily general model that also
happens to be consistent" still needs a parsimony criterion on top —
exactly the kind of result this project is supposed to surface
honestly rather than paper over.

## Run it

```bash
# generation + validation + ranking only (fast, no PINN, no network)
python experiments/06_hypothesis_generation.py --config config.yaml --skip-pinn

# full run: also trains a PINN for every surviving candidate (~3-4 min each)
python experiments/06_hypothesis_generation.py --config config.yaml

# run PINN validation for just one hypothesis at a time (results merge into
# the same report file) -- useful if you're time- or compute-constrained
python experiments/06_hypothesis_generation.py --config config.yaml --only H1

python -m pytest tests/test_hypothesis_generation.py -v
```

## API configuration / mock vs. real

```yaml
llm:
  provider: "mock"       # or "openai"
  model: "gpt-4o-mini"    # only used when provider=="openai"
  temperature: 0.7
  n_hypotheses: 4
```

```bash
export OPENAI_API_KEY=sk-...        # required for provider="openai"
export OPENAI_MODEL=gpt-4o          # optional, overrides config.yaml's llm.model
```

No key is ever read from a file or hardcoded. If `provider: "openai"`
is set but `OPENAI_API_KEY` is missing, `llm.get_provider()` logs a
warning and transparently falls back to `MockLLMProvider` rather than
crashing — tests and offline runs are unaffected either way.

## Tests

```
tests/test_hypothesis_generation.py (19 assertions across these functions):
  test_hypothesis_schema_validation PASSED
  test_schema_rejects_bad_confidence PASSED
  test_schema_rejects_equation_without_equals PASSED
  test_required_fields_enforced PASSED
  test_equation_parsing_diffusion PASSED
  test_equation_parsing_advection_diffusion PASSED
  test_equation_parsing_reaction_diffusion PASSED
  test_equation_parsing_rejects_undeclared_symbol PASSED
  test_invalid_hypothesis_rejected_by_deterministic_validation PASSED
  test_derivative_mismatch_warning PASSED
  test_unused_parameter_warning PASSED
  test_dimensional_completeness_partial PASSED
  test_duplicate_removal PASSED
  test_canonical_signature_sign_invariant PASSED
  test_hypothesis_to_pinn_config PASSED
  test_mock_provider_generates_valid_json PASSED
  test_end_to_end_pipeline_with_mock_provider PASSED
  test_backward_compatibility_phase1_3 PASSED   <- re-runs 8 Phase 1-3 test functions directly
```

Full regression (`python -m pytest tests/ -v`, excluding the
standalone `test_diffusion_data.py` script which has no pytest-style
functions): **27 passed**, 0 failed, 0 skipped *(count at the end of Phase 4; current total in the status table at the top)*.

## Limitations

- **The live demonstration used the mock provider.** The real
  `OpenAIProvider` is fully implemented and unit-testable (import,
  config, lazy client construction all verified), but this sandboxed
  environment has no outbound network access to `api.openai.com` and
  no API key configured. The mock's 4 canned hypotheses happen to
  include the correct equation (by construction, since they're meant
  to cover the project's known equation families) — this demonstrates
  the *pipeline* (schema -> validation -> dedup -> ranking -> PINN
  adapter) works correctly, not that an LLM "discovered" diffusion
  from reasoning. Re-run with a real key for the genuine version of
  this claim.
- **Threshold-based validation can't penalize harmless extra
  complexity**, as the H3/H4 result above shows directly. The
  simplicity term in ranking is a partial mitigation, not a complete
  solution — Phase 5's benchmark should test this more systematically.
- **Dimensional/unit checking is metadata bookkeeping, not real
  dimensional analysis** (no unit-arithmetic consistency check across
  an equation's terms is performed) — stated explicitly in the schema
  and validation code, not papered over.
- **The equation compiler covers one scalar field in 1D** (`u(x,t)`
  only) — no systems of equations, no multiple dependent variables, no
  non-local/integral terms, no derivatives above third order.
- **Prompt-injection / adversarial equations are not specifically
  hardened against** beyond the symbol-scope restriction in
  `equation_compiler.py` (any symbol not in `{u,u_t,u_x,u_xx,u_xxx}`
  plus declared parameters raises a parse error rather than silently
  being treated as a new free variable) — a determined adversarial
  prompt is a later evaluation phase's "20 adversarial physics cases"
  territory, not fully addressed here.

---

# Phase 5 — Scientific RAG + Knowledge Graph

**Milestone:** a small, honest, local scientific corpus can be
ingested, chunked, embedded, and retrieved; the retrieved evidence is
organized into a domain-agnostic knowledge graph with full provenance;
both feed a unified `ScientificContext` that Phase 4's hypothesis
engine can optionally consume — all without ever leaking this
project's specific hidden equation or parameter value into the
discovery process.

## Why RAG, why a knowledge graph, and how they differ

**RAG answers "what has been documented about this kind of
phenomenon?"** — it retrieves relevant passages of text from a corpus,
ranked by similarity to a query. It is good at surfacing relevant
background, bad at expressing *structured* relationships (which
variables appear in which equations, which mechanisms are documented
by which papers).

**The knowledge graph answers "how do these documented concepts relate
to each other?"** — entities (mechanisms, variables, parameters,
equations, documents) and typed relationships (`GOVERNS`,
`HAS_VARIABLE`, `HAS_PARAMETER`, `SUPPORTED_BY`, ...) connected with
explicit provenance, so a question like "what parameters appear in the
documented equations for diffusion?" has a structured, traceable
answer instead of requiring another fuzzy text search.

They are complementary, not redundant: RAG finds candidate evidence;
the graph organizes confirmed structure *within* that evidence. Neither
decides what's true — see "Responsibility boundaries" below.

## What's new

```
rag/
├── documents.py       # Document / DocumentMetadata / Chunk schemas (provenance-preserving)
├── chunker.py           # header-based scientific chunking (title/section/subsection), paragraph fallback
├── embeddings.py          # EmbeddingProvider ABC: LocalEmbeddingProvider (deterministic, offline) + APIEmbeddingProvider (stub)
├── vector_store.py         # VectorStore ABC: InMemoryVectorStore + PersistentVectorStore (JSON-backed)
├── retriever.py              # top-k + domain/section/metadata filtering -> Evidence objects
├── evidence.py                 # Evidence schema (confidence = retrieval score ONLY, never truth)
├── metadata.py                   # filter-matching helpers
├── corpus_loader.py                # loads rag/corpus/*.md fixture files (YAML frontmatter + markdown body)
├── corpus/                           # 5 original, hand-written "benchmark fixture" documents (see below)
├── context.py                          # build_scientific_context(): RAG + KG -> ScientificContext
└── pipeline.py                           # RAGPipeline: ingest() -> chunk+embed+index, get_retriever()
knowledge_graph/
├── schema.py        # domain-agnostic EntityType/RelationType enums, Entity/Relationship (with provenance)
├── graph.py           # KnowledgeGraph: dedup-by-construction, stable hash-based relationship IDs
├── builder.py           # deterministic graph construction from documents' own declared_* facts (no LLM)
├── query.py                # equations_for_mechanism(), variables_for_mechanism(), evidence_connecting(), ...
└── serialization.py          # graph_to_dict/from_dict, save_graph/load_graph
data/problem_description.py    # moved here from experiments/06 (now shared with experiments/07, see below)
experiments/07_scientific_rag_kg.py
tests/{test_rag.py, test_knowledge_graph.py}
results/rag_kg/
├── retrieval_results.json, knowledge_graph.json, generated_hypotheses.json, phase5_report.md
```

`llm/prompts.py` gained `build_hypothesis_prompt_with_context()` (Mode
B); `hypotheses/pipeline.py`'s `generate_hypotheses()` gained one new
optional parameter, `context: Optional[Dict] = None` — passing nothing
(the default) reproduces Phase 4's exact behavior (Mode A), confirmed
by Phase 4's entire test suite still passing unchanged. `llm/` has
**no import dependency on `rag/`** — the prompt builder takes a plain
dict, not a `ScientificContext` import — so Phase 4 remains fully
usable without Phase 5 installed, per the architectural constraint.

## Local reproducible corpus

`rag/corpus/*.md`: 5 documents (`diffusion`, `advection`, `reaction`,
`reaction_diffusion`, `advection_diffusion`), each with YAML
frontmatter (`document_id`, `title`, `source: "benchmark fixture"`,
`domain`, `declared_mechanism/variables/parameters/equations/
assumptions`) and an original markdown body. **These are not
reproductions of any real publication** — they are original summaries
of well-known, generic textbook physics (Fick's law, the linear
advection equation, logistic growth, their combinations), written for
this project and explicitly labeled `is_benchmark_fixture: true`, with
`source: "benchmark fixture"` rather than a fabricated citation. No
author, year, or venue is invented — `authors: []` and `year: null`
where genuinely unknown, per the "represent unknown as null, never
fabricate" requirement.

## Preventing answer leakage — the key design decision of this phase

The corpus's diffusion document states the **generic textbook form**
`u_t = D*u_xx` as "Fick's second law" — exactly as any physics
textbook would. **It never states this project's fitted value `D≈0.1`,
and it never asserts "this specific dataset follows diffusion."** This
is a deliberate distinction, not an oversight:

- A real scientist already knows the general candidate equations for
  common transport mechanisms (textbook knowledge) before investigating
  a new dataset. Retrieving that same textbook knowledge is not
  answer leakage — it's exactly the kind of prior scientific knowledge
  the hypothesis engine is *supposed* to draw on.
- What **would** be leakage is stating *this specific dataset's*
  governing relationship or parameter value directly in the retrieved
  context, bypassing the need for PINN validation and parameter
  estimation entirely. No corpus document does this; every
  `declared_equations` entry is the generic textbook form, and every
  document's `declared_parameters` list names the parameter (`D`, `c`,
  `r`, `K`) without ever stating its value for this experiment.
- `rag/context.py`'s module docstring states this reasoning explicitly,
  and `experiments/07_scientific_rag_kg.py` runs an automated check
  (`Step 0`, not just an unverified claim) confirming the literal
  string `"0.1"` (this experiment's fitted `D_true`) does not appear
  anywhere in the corpus text — real output: `"Answer-leakage check:
  D_true (0.1) does not appear in any corpus document's text. OK."`

The hypothesis engine (and, downstream, PINN validation) still has to
decide *which* documented mechanism (if any) applies to the specific
noisy observations and estimate its parameters from scratch — nothing
about this is made easier by retrieval beyond the same background
knowledge a human scientist would already have.

## Document ingestion, chunking, retrieval — in practice

```
Ingested 5 corpus documents -> 25 chunks.
```

Chunking splits on markdown headers (`rag/chunker.py`), so each chunk
corresponds to the document's own title/section/subsection structure
("Overview", "Governing equation", "Typical observational signatures",
"Typical assumptions") rather than an arbitrary character count.

`rag/embeddings.py`'s `LocalEmbeddingProvider` is a deterministic
bag-of-words **hashing** embedding (token → `hashlib.md5` → fixed-size
vector index, L2-normalized) — stated honestly as lexical/keyword
similarity, not a learned semantic embedding model. It is enough to
retrieve meaningfully for this small, keyword-distinct fixture corpus:
querying with the exact evidence-based problem description from Phase
4 (amplitude decay, smoothing, no bulk translation — never naming a
mechanism) correctly surfaces the diffusion document's "Governing
equation" and "Typical observational signatures" sections as the two
top hits, with the advection document close behind (score 0.547) and
reaction further down (0.474) — a real, unforced retrieval result, not
cherry-picked:

| Evidence | Document | Section | Score |
|---|---|---|---|
| evidence_doc_advection_chunk_003 | doc_advection | Governing equation | 0.547 |
| evidence_doc_advection_chunk_004 | doc_advection | Typical observational signatures | 0.533 |
| evidence_doc_diffusion_chunk_003 | doc_diffusion | Governing equation | 0.528 |
| evidence_doc_diffusion_chunk_004 | doc_diffusion | Typical observational signatures | 0.520 |
| evidence_doc_reaction_chunk_003 | doc_reaction | Governing equation | 0.474 |

(Advection narrowly outscores diffusion here under this simple
lexical/keyword embedding — a good concrete illustration of why
`Evidence.confidence` must never be read as scientific truth: both
documents use very similar vocabulary to describe their respective
"typical observational signatures", and the embedding cannot tell
which one actually fits *this* data better. Only PINN validation can.)

## Knowledge graph — in practice

```python
build_graph_from_documents(docs, chunks)
# -> 22 entities (5 Document, 5 Mechanism, 3 Variable, 4 Parameter, 5 Equation)
# -> 40 relationships, ALL sourced (0 unsourced)
```

Entities are deduplicated by deterministic ID, so shared concepts merge
correctly across documents: the `Variable` entity for `u` and the
`Parameter` entity for `D` are each a *single* node, even though `D`
is declared by three different documents (`diffusion`,
`reaction_diffusion`, `advection_diffusion`) — verified directly in
`tests/test_knowledge_graph.py:test_shared_variable_and_parameter_entities_merge_across_documents`,
which confirms the single `parameter_D` node accumulates sourced
relationships from all three. Relationship IDs are a stable hash of
`(subject, predicate, object)`, so adding the same fact from a second
document **merges sources** rather than creating a duplicate edge —
this is the "duplicate handling" requirement, demonstrated, not just
described.

Example query results (`knowledge_graph/query.py`, real output):
```
equations_for_mechanism(graph, "diffusion")   -> ["u_t = D*u_xx"]
variables_for_mechanism(graph, "diffusion")    -> ["t", "u", "x"]
parameters_for_mechanism(graph, "diffusion")    -> ["D"]
documents_supporting(graph, mechanism_advection) -> ["Advective (Directional) Transport -- Benchmark Fixture"]
```

## The unified scientific context

`rag/context.py:build_scientific_context()` retrieves evidence, then
queries the graph for every mechanism that evidence mentions, and
returns one `ScientificContext` object:
```
mechanisms:      ['advection', 'diffusion', 'reaction']
known_equations: ['u_t + c*u_x = 0', 'u_t = D*u_xx', 'u_t = r*u*(1-u/K)']
parameters:      ['D', 'K', 'c', 'r']
sources:         ['benchmark fixture']
```
This is the object passed (as a plain dict, `context.model_dump()`) to
Phase 4's `generate_hypotheses(..., context=...)`.

## Connecting to Phase 4 and Phase 3 (and an honest ablation-readiness finding)

`experiments/07_scientific_rag_kg.py` runs **both** modes on the same
problem and reports a structured comparison:

| | Mode A (evidence-free) | Mode B (evidence-grounded) |
|---|---|---|
| Raw candidates | 4 | 4 |
| Structurally valid | 4 | 4 |
| Duplicates removed | 0 | 0 |
| Successfully compiled for Phase 3 | 4 | 4 |

**Mode A and Mode B produced the identical 4 candidate equations in
this run.** This is reported, not hidden: `MockLLMProvider` returns a
fixed canned response regardless of prompt content (by design, for
deterministic offline testing — see Phase 4's README section), so it
does not yet condition on the retrieved context. **The architecture is
fully wired and tested** (`ScientificContext` → context-aware prompt →
`provider.generate()` → identical downstream schema/validation/ranking
pipeline — verified in
`tests/test_rag.py::test_integration_full_chain_to_phase3_adapter`,
which runs the complete chain from raw documents through to Phase
3-compatible compiled hypotheses), but **demonstrating RAG/KG actually
changing generated hypotheses requires a real provider**
(`llm.provider: "openai"` + `OPENAI_API_KEY`), not exercised live in
this sandboxed environment (same network/key limitation as Phase 4).
This is exactly the "ablation-ready, not yet demonstrated" scope the
Phase 5 spec asked for.

The top-ranked Mode-B candidate (H1, diffusion) was run through the
**unmodified** Phase 3 PINN validator via the **same**
`hypotheses/adapter.py` from Phase 4 (no second PINN implementation):

```
status: supported by the available observations and physics constraints
data_loss=0.000132  physics_residual=0.000033  prediction_RMSE=0.000880  D_hat=0.09989
```

— consistent with Phase 3's and Phase 4's own results for the same
equation, as expected, since nothing about the PINN layer changed.

## Responsibility boundaries (unchanged, now with one more link in the chain)

```
RAG            -> provides evidence (text, with a similarity score, never "truth")
Knowledge Graph -> provides structured relationships (with provenance, never "proof")
Hypothesis Engine -> generates candidate equations (with self-reported confidence, never "correctness")
Equation Compiler  -> checks structural/syntactic validity only
PINN               -> tests physical consistency against data (the ONLY accept/reject authority)
```

No retrieval score, no graph edge, and no LLM confidence value is ever
substituted for a PINN validation status anywhere in this codebase.

## Run it

```bash
# fast: ingestion, chunking, retrieval, graph, context, Mode A/B generation -- no PINN
python experiments/07_scientific_rag_kg.py --config config.yaml --skip-pinn

# full: also runs the top-ranked Mode-B candidate through Phase 3 PINN validation (~4 min)
python experiments/07_scientific_rag_kg.py --config config.yaml

# validate specific hypothesis id(s) instead of just the top-ranked one
python experiments/07_scientific_rag_kg.py --config config.yaml --only H1,H2

python -m pytest tests/test_rag.py tests/test_knowledge_graph.py -v
```

## Tests

`tests/test_rag.py` (14 functions): document ingestion, metadata
preservation (including honestly-null `year`), header-based + fallback
chunking, deterministic local embeddings, deterministic retrieval,
top-k behavior, domain filtering, section filtering, provenance
preservation, empty-query/empty-index edge cases, vector-store JSON
persistence round-trip, and the full integration chain through to
Phase 3-compiled hypotheses.

`tests/test_knowledge_graph.py` (13 functions): entity creation
(including no-duplicate-on-same-id), relationship creation, a
relationship to a non-existent entity raising `ValueError`, duplicate
relationship merging (same fact from two documents → one edge, two
sources), same-source-not-double-counted, provenance-type tagging,
domain-agnostic graph construction over the real corpus, cross-document
entity merging (the `D` parameter test above), all three
`query.py` lookup functions, and a full serialization round-trip.

**Full regression**: `python -m pytest tests/ -v` (excluding the
standalone `test_diffusion_data.py` script) → **54 passed, 0 failed** *(count at the end of Phase 5)* —
every Phase 1–4 test still passes unchanged alongside the 27 new Phase
5 tests.

## Limitations

- **Retrieval is lexical (bag-of-words hashing), not semantic.** Stated
  throughout (`rag/embeddings.py`, `Evidence.confidence` docs, and the
  advection-narrowly-beats-diffusion example above). `APIEmbeddingProvider`
  exists for real semantic embeddings but was not exercised here (no
  network/key in this sandbox, same as `OpenAIProvider`).
  `PersistentVectorStore` is a JSON file, not a real vector database —
  adequate for this corpus size, documented as a scaling limitation.
- **The live Mode A vs. Mode B comparison shows identical output**,
  because `MockLLMProvider` doesn't read its prompt. This demonstrates
  the architecture is ablation-ready, not that RAG/KG currently change
  results — a real provider is required for that, and is fully wired
  to accept one (`llm.provider: "openai"`).
- **Graph construction is rule-based over each fixture document's own
  declared metadata, not LLM extraction over free text.** This keeps
  everything deterministic and avoids fabricating relationships, but
  means graph coverage is limited to what a document's author (here:
  this project) explicitly declared. The schema and `provenance_type`
  field are ready for an LLM-based extractor over real literature
  later, with its output correctly marked `"extracted"`/`"inferred"`
  rather than `"documented"`.
- **No `Observation`/`Experiment` entities are populated** by the
  current builder (the fixture corpus is mechanism-description
  documents, not experimental records) — `mechanisms_for_observation()`
  exists in `knowledge_graph/query.py` and is tested to return an
  empty list honestly, ready for a corpus that includes actual
  experimental observation documents.
- **PDF ingestion is not implemented** (`rag/documents.py` and
  `corpus_loader.py` handle markdown/text only) — noted as a "where
  practical" scope reduction, not a silent omission.
- **The corpus is small (5 documents, 25 chunks) by design** — large
  enough to demonstrate every pipeline stage honestly, far too small to
  demonstrate retrieval quality at scale.



---

# Phase 6 — Physics-Constrained Generative Search

**Milestone:** move from "generate plausible hypotheses" to
**generate → constrain → test → compare → select → predict**, reusing every earlier phase. There is no
second hypothesis pipeline and no second PINN.

## Pipeline (actual implementation)

```
problem description (computed data statistics only)          data/problem_description.py
  -> RAG retrieval + KG -> ScientificContext                  rag/, knowledge_graph/  (Phase 5)
  -> candidate pool = LLM proposals ∪ KG equation templates   hypotheses/search.py (+ Phase 4 pipeline)
  -> structural dedup + validation                            hypotheses/validation.py (Phase 4)
  -> physics pre-checks (well-posedness, bounds)              hypotheses/constraints.py
  -> inverse PINN per candidate, train split only, t<=0.8     pinn/trainer via hypotheses/adapter.py (Phase 3/4)
  -> post-fit physical bounds (e.g. D>0)                      hypotheses/constraints.py
  -> gate on HELD-OUT NOISY observations                      hypotheses/search.py
  -> term necessity + complexity + BIC -> selection           validation/model_selection.py, hypotheses/complexity.py
  -> empirical stability (bootstrap)                          validation/stability.py
  -> OOD prediction (t>0.8, never trained/selected on)        hypotheses/search.py
  -> novel prediction: solve the selected law to t=1.5        validation/forward_solver.py
```

**How the KG feeds the search:** the knowledge graph's documented equation forms for the retrieved
mechanisms become candidates alongside the LLM's proposals. In this run the KG templates for advection
and diffusion were recognized as structural duplicates of LLM candidates H2/H1. The KG also contributed
one genuinely new candidate, **K3: pure reaction `u_t = r*u*(1-u/K)`**, which the LLM had not proposed.

**No oracle in any decision.** Observations are split into train (t≤0.8), an in-distribution validation
holdout (20%, random), and OOD (t>0.8). The gate and model selection use only the noisy holdout. `D_true`
and the clean field are loaded in one function (`oracle_evaluation`), which runs *after* selection and only
reports benchmark error.

## Physical constraints

Bounds are declared **per parameter** on the hypothesis (`ParameterSpec.lower_bound/upper_bound`, a
backward-compatible schema addition). `config.yaml` supplies a name-keyed fallback: `D>0`, because D≤0 is
the ill-posed backward heat equation, and `K>0`. Velocity `c` and growth rate `r` are deliberately
unconstrained. Pre-PINN checks reject non-evolution equations and inconsistent bounds. They also warn,
without rejecting, on two real modeling issues:
- Advection with Dirichlet data at both ends is over-determined: a hyperbolic equation admits only an inflow boundary.
- An equation with no spatial derivative cannot couple to boundary conditions at all.

## Model selection (the nested-model problem from Phase 4)

Three questions, kept separate:
1. *Can it explain the data?* Answered by the gate.
2. *Does the data require each term?* Answered by **term necessity**: the RMS share of `u_t` contributed
   by all terms containing each parameter, evaluated on the trained PINN. A share below 5% marks the
   parameter "inactive", and the equation with those terms removed is matched against the other candidates.
3. *Which is the most parsimonious supported explanation?* Rule, fixed in config before the run: keep
   candidates whose holdout RMSE is within 5% of the best, then take the lowest complexity, then the lowest
   BIC. Complexity counts terms, free parameters, derivative order, nonlinear terms, and expression size.

## Results (actual run)

| ID | Origin | Equation | Estimated params | Holdout RMSE | Status | Inactive | Complexity |
|---|---|---|---|---|---|---|---|
| H1 | LLM | `u_t = D*u_xx` | D=0.10002 | 0.01214 | supported | – | 4.1 |
| H2 | LLM | `u_t + c*u_x = 0` | c=0.0089 | 0.2269 | **rejected** (holdout RMSE, IC loss) | – | 3.2 |
| H3 | LLM | `u_t = D*u_xx + r*u*(1-u/K)` | D=0.101, r=0.023, K=1.18 | 0.01217 | supported | r, K → reduces to H1 | 10.7 |
| H4 | LLM | `u_t + c*u_x = D*u_xx` | c=−0.00024, D=0.09995 | 0.01214 | supported | c → reduces to H1 | 6.3 |
| K3 | KG | `u_t = r*u*(1-u/K)` | r=−1.44, K=6.67 | 0.1176 | **rejected** (holdout RMSE) | – | 6.5 |

**Selected: H1, `u_t = D*u_xx`, D̂ = 0.10002.** The equivalence set is {H1, H4, H3}, and BIC independently
selects H1. Note that **H4 has the lowest raw holdout RMSE** (0.012135 vs 0.012143). A pure best-fit rule
would have chosen the more complex model, and the 0.06% gap is far below what 5% measurement noise
can resolve. The holdout RMSE for all three supported candidates sits at the measurement-noise floor (~0.012).

**OOD (t>0.8, held-out noisy observations; the oracle column is evaluation-only):**

| Model | OOD RMSE vs noisy obs | OOD RMSE vs clean field (oracle) |
|---|---|---|
| H1 PINN (selected) | 0.0114 | 0.0018 |
| H4 PINN | 0.0118 | 0.0031 |
| H3 PINN | 0.0121 | 0.0038 |
| K3 PINN | 0.1026 | 0.0985 |
| H2 PINN | 0.3249 | 0.3337 |
| **Data-only MLP (conventional NN baseline)** | 0.0727 | 0.0712 |
| Forward solve of selected law (numerical solver) | 0.0113 | 0.0003 |

**Novel prediction:** the selected law, solved forward from the known IC with D̂ and using no observations
at all, predicts the window t∈(1.0, 1.5] where no data exists. Its oracle RMSE there is **0.00025**. The
data-only MLP with identical architecture is 5× worse even in-distribution (0.061 vs 0.012) and drifts visibly out of
distribution (`results/phase6/phase6_summary.png`).

## Empirical stability, and a disagreement we report rather than resolve

A bootstrap refits each candidate's terms on 20 random 80% subsets of the training data. It uses ordinary
least squares on Phase 2's finite-difference derivatives, which is cheap but **not** PINN retraining and
**not** Bayesian.
- **Agreement:** the `u_xx` coefficient is sign-stable in every diffusion-containing candidate. H4's `u_x`
  coefficient flips sign in 20% of subsets, consistent with the PINN finding `c` inactive.
- **Disagreement:** for H3, the bootstrap finds sign-stable `u` and `u²` coefficients, while the PINN finds the
  reaction mechanism contributes under 1%. The same spurious `u`/`u²` terms appear in Phase 2's noisy SINDy
  failure. The bootstrap's `u_xx` coefficient (~0.11) is also biased high relative to the PINN's 0.100.
  Both point to a finite-difference reconstruction artifact. This is an interpretation, not a proof, and the
  report flags it automatically.

## Files

Created:
- `hypotheses/search.py`, `hypotheses/constraints.py`, `hypotheses/complexity.py`
- `validation/model_selection.py`, `validation/stability.py`, `validation/forward_solver.py`
- `experiments/08_physics_constrained_search.py`, `tests/test_physics_constrained_search.py` (18 tests)
- `results/phase6/`: `generated_candidates.json`, `filtered_candidates.json`, `pinn_validation.json`,
  `model_selection.json`, `ood_predictions.json`, `phase6_report.md`, `phase6_summary.png`,
  `candidates/*.json`, `models/*.pt` (trained PINN weights)

Modified (backward-compatible):
- `llm/schemas.py`: optional parameter bounds
- `llm/mock_provider.py`: declares `D>0` and `K>0`
- `config.yaml`: `phase6` section
- `tests/test_diffusion_data.py`: now pytest-collectable. It previously contributed **zero** tests to
  `pytest tests/`, which was found in the audit.

## Run

```bash
python experiments/08_physics_constrained_search.py --config config.yaml           # everything (~20 min CPU)
# or staged (each inverse PINN is ~4 min; results are cached per candidate):
python experiments/08_physics_constrained_search.py --stage prepare
python experiments/08_physics_constrained_search.py --stage pinn --only H1          # repeat per candidate
python experiments/08_physics_constrained_search.py --stage select
python -m pytest tests/test_physics_constrained_search.py -v
```

## Limitations (what Phase 6 does NOT show)

- **Not proof, not uniqueness.** Three structurally different equations explain the data equally well. H1 is
  chosen by a stated parsimony rule, and the nested candidates collapse to it when their inactive terms are
  dropped. The correct statement is: *H1 is the simplest candidate supported by the available observations,
  physics constraints and OOD validation.*
- **Mock LLM.** The LLM candidates are canned, and the pool happens to contain the right family by
  construction. The search, filtering and selection machinery is genuine; LLM-driven proposal is not
  demonstrated.
- **Synthetic, single system:** one PDE family, one noise level (5%), 400 points, a 5-document hand-written
  corpus, and lexical (hashing) embeddings.
- **Stability is a cheap FD-regression bootstrap** and partly disagrees with the PINN analysis (see above).
  There is no per-candidate PINN retraining on subsets and no Bayesian uncertainty.
- **IC/BC are supplied as known physics**, not inferred.
- **Thresholds are hand-set,** fixed before the run, and not calibrated on an independent benchmark.
- **Not yet built at the end of Phase 6** *(since built in Phase 7)***:** the 50/25/20 benchmark and the LLM vs RAG vs KG ablation. The architecture supports them
  (Mode A/B in Phase 5), but no experiments have been run.

---

# Phase 7 — Evaluation, Adversarial Testing & Generalization

**Milestone:** make the system experimentally defensible. Test it on mechanisms other than diffusion,
attack it with cases built to make it accept unsupported physics, measure each component's contribution,
and audit it for leakage. Full generated report: `results/evaluation/phase7_report.md`.

## What was built (evaluation/ package, no new architecture)

- `datasets.py`: 12 ground-truth datasets covering 4 families (diffusion, advection, reaction-diffusion,
  advection-diffusion), 3 parameter settings each. Exact analytic solutions are used where they exist. Reaction-diffusion
  uses the generic solver with RK4 and is grid-converged to within 0.1%. Cached, with a bit-identical reload.
- `describe.py`: turns any dataset into the text query that RAG receives, using one fixed template whose wording
  is chosen only by thresholds on measured statistics. A test checks that it never names a mechanism.
- `identify.py`: mechanism identification by term set, symbolic parameter recovery, and an identifiability
  (collinearity) diagnostic.
- `engine.py`: the Phase 6 pipeline with the PINN replaced by OLS on finite-difference derivatives, so 95 items
  run in about 1 minute. The PINN version of the same pipeline is run on the second system (below).
- `benchmark.py`, `adversarial.py`, `ablation.py`, `leakage.py`.
- Experiments: `09_evaluation.py` runs all suites and writes the report; `10_generalization.py` runs the PINN pipeline
  on the second system.

Minimal changes to existing code, all backward compatible and regression-checked. Phase 6 results reproduce
byte-for-byte and the Phase 3 inverse PINN still gives D̂=0.099888:
- `forward_solver`: an optional RK4 method. Euler with central differences is unconditionally unstable for advection.
- `pinn/data_utils.py`: the IC sampler accepts a function.
- `select_model`: an opt-in `prefer_reduced` flag, explained below.

## Results

**50 scientific questions: 50/50.** Structure identification 12/12, parameter estimation 12/12 (within 5%),
KG parameters 5/5, equation derivatives 5/5, KG provenance 5/5, complexity 5/5, and retrieval recall 6/6.
*Caveat:* the KG, compiler and complexity questions check internal consistency and are easy by construction.
Retrieval recall over a 5-document corpus is weak evidence, and **top-1 retrieval accuracy is only 1/6**.

**25 multi-hop problems: 23/25.**
| Category | Hops | Score |
|---|---|---|
| data → description → RAG → KG → validation → selection → beyond-window prediction | 6 | 12/12 |
| text → retrieval → KG → compiler | 4 | 3/5 |
| parameter → KG → equations → derivatives | 3 | 4/4 |
| nesting tests | 3 | 4/4 |

Both failures come from top-1 lexical retrieval returning "advection" for descriptions of combined mechanisms.

**20 adversarial cases: 20/20.** The cases are:
- invalid equations: undeclared symbols, no time derivative, nonlinear in u_t;
- invalid bounds and negative diffusivity;
- **time-reversed data whose best fit genuinely needs D<0** (rejected by the D>0 bound);
- disguised duplicates and malicious LLM output;
- the Phase 1 eigenfunction trap (flagged as non-identifiable);
- nested and Occam traps:
  - **reaction-diffusion data, where the simpler diffusion candidate is rejected at 85% error** rather than chosen for
    simplicity;
  - **advection-diffusion data, where SINDy alone drops the real u_xx term** but the full system recovers it;
- retrieval that ranks the wrong mechanism first;
- noise and sparsity;
- corpus answer leakage and prompt injection.

Qualifications:
- **A17–A19 (1%/10% noise, 10% sparse) pass by abstention.** No candidate clears the gate. This is safe but is not
  discovery, because the regression engine has no noise robustness.
- **A10 passes only under the revised rule** described below.

**Ablation (12 datasets):**

| Arm | Correct model | Wrong candidates rejected | Params within 5% | OOD within 5% |
|---|---|---|---|---|
| Equation discovery alone (SINDy) | 8/12 (adv-diff 0/3) | – | 8/12 | 6/12 |
| RAG + hypothesis generation, no validation | 3/12 (always "diffusion") | 0% | 3/12 | 3/12 |
| RAG + KG + hypothesis generation, no validation | 3/12 (always "diffusion") | 0% | 3/12 | 3/12 |
| **Full system** | **12/12** | **100%** | **12/12** | **12/12** |
| Full system, original Phase 6 rule | 10/12 | 100% | 10/12 | 12/12 |
| Full system without the LLM | 12/12 | 100% | 12/12 | 12/12 |
| Retrieved-evidence equations + validation (no KG, no LLM) | 12/12 | 100% | 12/12 | 12/12 |

The original-rule arm reaches 12/12 on OOD while identifying only 10/12 models correctly. Its two wrong picks
(reaction-diffusion with r≈0) are numerically equivalent to the truth. **Accurate prediction alone does not establish
the right law,** which is why identity and prediction are scored separately.

**What each component contributes on this benchmark:**
- **Physics validation and selection** produce correct identification. Without them, generation repeats its
  top-ranked prior.
- **RAG** supplies a candidate pool that contains the truth (100% recall). With 5 documents this is easy.
- **The KG and the mock LLM add no measurable discovery accuracy.** Retrieved evidence already carries each document's
  declared equations. The KG's demonstrated value is structured multi-hop querying and provenance, not accuracy.
- **SINDy** is fast and accurate when every true term is large. Its threshold silently drops small real terms.

## Generalization: second physical system with the full PINN pipeline

The hidden truth is `u_t + c*u_x = D*u_xx` with c=0.4 and D=0.02 on x∈[0,1.5], a Gaussian IC, and 400 random points with 5% noise.
The splits, gate, selection and constraints are identical to Phase 6, and no new architecture was added.

| Candidate | Fitted | Held-out RMSE | Status | OOD RMSE vs clean field (oracle) |
|---|---|---|---|---|
| H1 diffusion | D=0.075 | 0.096 | rejected | 0.111 |
| H2 advection | c=0.369 | 0.064 | rejected | 0.113 |
| H3 reaction-diffusion | D=0.117, r=1.22, K=1.44 | 0.084 | rejected | 0.095 |
| **H4 advection-diffusion** | **c=0.39897, D=0.02008** | **0.0099** | **selected** (both rules) | **0.0010** |
| Data-only MLP | – | 0.0124 | – | 0.0068 |

The parameters are within 0.26% (c) and 0.40% (D) of truth. Solving the selected law forward to t=1.3, beyond the data,
gives a relative error of 0.58%. SINDy on the same noisy data returned a spurious `u`, `u²` equation.

## A disclosed rule change

Phase 7 found that the Phase 6 selection rule fails on noise-free two-mode diffusion data. Finite-difference
derivatives give each sine mode a slightly wrong eigenvalue, and `a*u_xx + b*u` can correct both modes exactly. As a
result, reaction-diffusion fits the *discretization error* to about 1e-14 and wins "within 5% of best error". Yet its
reaction mechanism contributes only **0.27%** of the dynamics.

Phase 6 already computed term necessity but used it only to explain selections. Phase 7 promotes it, using the same
pre-fixed 5% threshold: *if the selected candidate's extra terms are inactive and the reduced equation is itself
supported, select the reduced equation.* The flag is opt-in (`prefer_reduced`), so Phase 6 results are unchanged.
Every Phase 7 table reports both rules. The revised rule changes 2 of 12 benchmark outcomes, never strips a real
reaction term (3/3 reaction-diffusion still selected), and agrees with the original rule on the PINN generalization run.

## Leakage and reproducibility

- **Static audit:** none of the 6 decision functions (pool building, validation, splits, selection) reference
  ground-truth identifiers. A test confirms the audit flags a deliberately cheating selector, so the check is not vacuous.
- **Corpus scan:** none of the 15 hidden benchmark values appears in any corpus document. An injected document stating
  the answer is flagged, and it adds 0 KG relationships.
- **Ground truth enters only scoring functions,** which run after selection. `validate_and_select` has no dataset argument.
- **Determinism:** re-running the 50-question suite gives identical answers. Seeds are fixed in `config.yaml`.

## Real LLM

**Not run.** There is no `OPENAI_API_KEY`, and the sandbox proxy blocks `api.openai.com` (HTTP 403).
`experiments/09_evaluation.py --real-llm` runs the Mode A vs Mode B comparison on 3 datasets when a key and network are
available. `results/evaluation/real_llm.json` records `available: false, results: null`, and a test enforces that
no results are present when the LLM was unavailable.

## Requirement coverage (original brief)

| Requirement | Status | Evidence |
|---|---|---|
| Scientific RAG + KG | ✅ | Phase 5; KG multi-hop 8/8; provenance 5/5 |
| Generative hypotheses from multi-source evidence | ⚠️ pipeline yes, real LLM no | Mock LLM plus KG templates plus retrieved equations |
| Forward / inverse PINN | ✅ | Phase 3 (forward RMSE 0.0007, D̂ 0.1% error); Phase 7 (c, D within 0.4%) |
| PDE discovery from data | ✅ / ⚠️ | SINDy 8/12 clean; fails on noisy data, which is documented |
| Unknown parameter estimation | ✅ | Phases 3, 6, 7 |
| Sparse / noisy-data learning | ✅ (PINN) / ❌ (regression engine) | PINN: 400 pts at 5% noise on two systems; regression engine abstains |
| BC/IC inference | ❌ | Supplied as known physics |
| Physics-constrained generative search | ✅ | Phase 6, Phase 7 |
| Uncertainty | ⚠️ | Empirical bootstrap only (Phase 6) |
| OOD validation | ✅ | Phase 6, Phase 7 |
| Comparison vs conventional NN and numerical solver | ✅ | Data-only MLP; generic forward solver |
| ≥3 hypotheses, discovered equation, inverse + forward PINN, sparse/noisy, OOD, rejected hypothesis | ✅ | Phases 3–7 |
| 20 adversarial cases | ✅ | 20/20 (3 by abstention) |
| 50 questions / 25 multi-hop | ✅ | 50/50, 23/25 |
| Ablation | ✅ | 7 arms |
| Code, datasets, models, equations, KG, experiments, report, reproduction | ✅ | This repository |

## Demonstrated vs. not demonstrated

**Demonstrated:**
- rejection of structurally invalid, physically invalid, and data-inconsistent candidates;
- parsimonious selection among nested models;
- correct identification across 4 mechanism families, and from noisy sparse data on a second system;
- parameter recovery within 0.5%;
- prediction beyond the observed window;
- clear wins over SINDy alone and over generation without validation;
- deterministic, leakage-audited runs.

**Not demonstrated:**
- LLM-driven hypothesis generation;
- any accuracy benefit from retrieval ranking or the KG;
- noise robustness of the cheap engine;
- uniqueness of any selected law;
- PDEs beyond 1D scalar transport-reaction;
- real experimental data.
