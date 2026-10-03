# Generative Scientific Discovery Engine — Final Report

*Every number in this report is read from a result file in this repository, and every one was re-produced in the
Phase 8 audit (section 13). Each claim names its file.*

---

## 1. Problem statement

Given sparse, noisy observations of a physical field, and **not** the equation that governs it, can a system:
- propose candidate physical laws,
- test them against both the data and fundamental physics,
- reject the unsupported ones,
- estimate unknown parameters, and
- make a validated prediction outside the observed region?

The brief explicitly excludes a chatbot, a plain RAG system, and a PINN trained on a PDE it is handed.

## 2. Objective

Build and *experimentally evaluate* an end-to-end pipeline:

```
observations → evidence → hypotheses → equations → PINN → validation → selection → prediction
```

The evaluation must be honest about which stages actually contribute, and must never let the hidden answer
influence any decision.

## 3. System architecture

```
Scientific data / literature
        ↓
RAG + Knowledge Graph            rag/, knowledge_graph/
        ↓
Hypothesis generation            llm/ (mock provider), hypotheses/pipeline.py
        ↓
Candidate equations              hypotheses/search.py: LLM proposals ∪ KG-documented templates, deduplicated
        ↓
Equation discovery               equation_discovery/ (SINDy): an independent data-driven channel and a baseline
        ↓
Physics constraints              hypotheses/validation.py, hypotheses/constraints.py
        ↓
PINN forward/inverse validation  pinn/ via hypotheses/adapter.py and hypotheses/equation_compiler.py
        ↓
Model selection                  validation/model_selection.py, hypotheses/complexity.py
        ↓
OOD validation                   held-out t > 0.8 observations (hypotheses/search.py)
        ↓
Novel prediction                 validation/forward_solver.py: the selected law solved beyond the data window
```

| Component | Role in the actual implementation |
|---|---|
| RAG | Retrieves documented background (chunks with provenance) for a text description computed from the data. Retrieval scores are lexical similarity, never truth. |
| Knowledge graph | Typed entities and relationships (Mechanism GOVERNS Equation, HAS_PARAMETER, SUPPORTED_BY Document), every edge sourced. Supplies generic equation templates and parameter lists as candidates. |
| Hypothesis generation | A provider-agnostic LLM layer that returns schema-validated JSON hypotheses. **Only the deterministic mock provider was ever run.** |
| Equation compiler | sympy parses any candidate equation string into a PINN residual. No equation is hand-coded. |
| Equation discovery (SINDy) | Sparse regression on finite-difference derivatives. Used as a baseline and as an independent channel; it never decides. |
| Physics constraints | Rejects non-evolution equations and inconsistent bounds before training, and checks fitted parameters after training (e.g. D > 0). Warns on ill-posed boundary data. |
| PINN | One reusable MLP. The physics residual comes from the compiled candidate; autograd provides derivatives. In the inverse mode the unknown parameters are trained jointly with the network. |
| Model selection | Gate on **held-out noisy observations** (never the clean field), then term necessity, the fit-equivalence band, complexity and BIC. |
| OOD / prediction | Error on held-out t > 0.8 observations; the selected law is then solved forward with no observations into a window that has no data. |

## 4. Methodology by phase

| Phase | What was built | Key output |
|---|---|---|
| 1 | Synthetic diffusion data (finite-difference solver); 400 noisy points (noise std 5% of the field std); two-mode initial condition | `data/generated/` |
| 2 | Derivative estimation, candidate library, STLSQ | `results/equations/` |
| 3 | Reusable forward/inverse PINN, residual registry, OOD split | `results/pinn/` |
| 4 | LLM abstraction, hypothesis schema, symbolic equation compiler, ranking, PINN adapter | `results/hypotheses/` |
| 5 | 5-document fixture corpus, chunking, hashing embeddings, retriever, KG, ScientificContext | `results/rag_kg/` |
| 6 | Physics-constrained search, constraints, complexity, oracle-free selection, empirical stability, forward solver, MLP baseline | `results/phase6/` |
| 7 | 12-dataset 4-family benchmark, 50 + 25 questions, 20 adversarial cases, 7-arm ablation, PINN generalization, leakage audit | `results/evaluation/` |
| 8 | Full re-run of every experiment, claim audit, documentation | this report |

**Single-mode trap (found in Phase 2).** A single-sine initial condition is an eigenfunction of the second-derivative
operator, so `u_xx = −π²u` exactly. In that case `D·u_xx` and `−Dπ²·u` cannot be distinguished from the data.
Phase 2 discovered this when SINDy confidently returned `u_t = −0.987u`. All later work uses a two-mode initial
condition. The trap is kept as adversarial case A09.

## 5. Equation discovery (Phase 2, `results/equations/phase2_results_table.md`)

| Data | SINDy result | D estimate | Structure recovered |
|---|---|---|---|
| Clean | `u_t = 0.1009 u_xx` | 0.1009 (0.88%) | yes |
| Sparse, noise-free | `u_t = 0.1013 u_xx` | 0.1013 (1.26%) | yes |
| 400 noisy points (5%), raw or smoothed | 5-term spurious equation | — | **no** |

Full-library SINDy **fails on the noisy data**, and this is reported as a failure. In the Phase 7 benchmark, on clean
data across four families, SINDy alone scores 8/12. Its threshold drops small but real terms: all three
advection-diffusion cases lose their u_xx term.

## 6. PINN forward and inverse validation (Phase 3, `results/pinn/`)

| Experiment | Result |
|---|---|
| Forward PINN (D known, 400 noisy points) | RMSE vs. clean field **0.000727** |
| Inverse PINN (D unknown, initialized at 0.2) | **D̂ = 0.099888 (0.112% error)** |
| H1 diffusion vs. H2 advection, trained on t ≤ 0.7 | H1 OOD RMSE 0.00115; H2 0.331 (H2 rejected) |

*Audit note:* the Phase 3–5 status rule also checks error against the clean field. No outcome depends on that check:
H2 fails its data-fit and IC checks too. Phases 6–7 remove the oracle from all decisions.

## 7. RAG + knowledge graph (Phases 5 and 7)

- **Corpus and graph:** 5 hand-written benchmark-fixture documents (no fabricated citations), 25 chunks, and a KG with
  22 entities and 40 relationships, all of them sourced (`results/rag_kg/`).
- **Leakage check:** none of the 15 hidden benchmark values appears in the corpus
  (`results/evaluation/leakage_and_reproducibility.json`).
- **Measured retrieval quality is poor at top-1.** The lexical hashing embedding ranks "advection" first for every
  benchmark dataset, so top-1 accuracy is 1/6 (`results/evaluation/questions_50.json`). Recall in the top-5 chunks
  is 6/6, but with 5 documents that is weak evidence.
- **The KG works for structured multi-hop queries:** parameter-to-equation-to-derivative chains 4/4 and nesting
  tests 4/4 (`multihop_25.json`).

## 8. Physics-constrained hypothesis search (Phase 6, `results/phase6/`)

Candidates: H1–H4 from the mock LLM, plus K3 (pure reaction), which **came from the KG and not from the LLM**. The
KG's diffusion and advection templates were recognized as duplicates of H1 and H2. Training used t ≤ 0.8: 248 training
points, a held-out split of 62 used for the gate and for selection, and 90 OOD points.

| ID | Equation | Fitted | Held-out RMSE | Status | Inactive terms | Complexity |
|---|---|---|---|---|---|---|
| H1 | u_t = D u_xx | D = 0.10002 | 0.012143 | supported | – | 4.1 |
| H2 | u_t + c u_x = 0 | c = 0.009 | 0.2269 | rejected (held-out, IC) | – | 3.2 |
| H3 | u_t = D u_xx + r u(1 − u/K) | D = 0.101, r = 0.023 | 0.012172 | supported | r, K | 10.7 |
| H4 | u_t + c u_x = D u_xx | c = −0.0002, D = 0.09995 | **0.012135** | supported | c | 6.3 |
| K3 | u_t = r u(1 − u/K) | r = −1.44 | 0.1176 | rejected | – | 6.5 |

**Selected: H1.** The equivalence set was {H1, H4, H3}, and BIC independently picks H1. The noise floor is about 0.012.

- **H4 had the lowest raw error.** A best-fit rule would have chosen the more complex model; the 0.06% gap is far
  below what the noise can resolve.
- **OOD (t > 0.8):** H1 has RMSE 0.0114 against noisy observations and 0.0018 against the clean field. The data-only
  MLP scores 0.0727 and 0.0712.
- **Novel prediction:** the selected law solved over t ∈ (1.0, 1.5] has an oracle RMSE of 0.00025.
- **Stability methods disagree on H3.** The finite-difference bootstrap finds stable u and u² coefficients, while the
  PINN finds that reaction contributes under 1%. This is reported, not resolved.

## 9. Generalization experiment (Phase 7, `results/evaluation/generalization/`)

The hidden law is `u_t + c u_x = D u_xx` with c = 0.4 and D = 0.02, using 400 random points with 5% noise and the
same pipeline and thresholds. SINDy on the same data returned a spurious `u_t = 1.821u − 0.365u_x − 5.796u²`.

| Candidate | Fitted | Held-out RMSE | Status | OOD RMSE vs. clean field |
|---|---|---|---|---|
| H1 diffusion | D = 0.075 | 0.0958 | rejected | 0.111 |
| H2 advection | c = 0.369 | 0.0635 | rejected | 0.113 |
| H3 reaction-diffusion | D = 0.117, r = 1.22, K = 1.44 | 0.0841 | rejected | 0.095 |
| **H4 advection-diffusion** | **c = 0.39897, D = 0.02008** | **0.0099** | **selected (both rules)** | **0.0010** |
| Data-only MLP | – | 0.0124 | – | 0.0068 |

- **Parameter error:** c 0.26%, D 0.40%.
- **Novel prediction:** over t ∈ (1.0, 1.3], relative RMSE is 0.58%.
- **MLP comparison:** the MLP is closer here than in Phase 6 (6.6× worse on OOD, not 40×), because the OOD window is
  short and a single bump is easy to extrapolate briefly.

## 10. Adversarial evaluation (`results/evaluation/adversarial_20.json`): 20/20

Covered: invalid structure (A01–A03, A07), invalid parameters (A04–A06), malicious LLM output (A08), degeneracy
(A09–A10), nested models and Occam traps (A11, A13), SINDy's dropped term (A12), wrong candidates (A14), misleading
retrieval (A15), selection traps (A16), noise and sparsity (A17–A19), and leakage plus prompt injection (A20).

Highlights:
- **A06:** time-reversed data genuinely fits D = −0.100, and the D > 0 constraint rejects it.
- **A13:** the simpler diffusion candidate is rejected (85% error) on reaction-diffusion data, so parsimony never
  overrides fit.
- **A12:** advection-diffusion is recovered where SINDy reports advection.

Qualifications:
- **A17–A19 pass by abstention.** Nothing is selected, which is safe but is not discovery.
- **A10 passes only under the revised rule** (section 12).

## 11. Ablation (`results/evaluation/ablation.json`, 12 datasets in 4 families)

| Arm | Correct model | Wrong candidates rejected | OOD within 5% |
|---|---|---|---|
| Equation discovery alone (SINDy) | 8/12 | – | 6/12 |
| RAG + generation, no validation | 3/12 (always "diffusion") | 0% | 3/12 |
| RAG + KG + generation, no validation | 3/12 | 0% | 3/12 |
| **Full system** | **12/12** | **100%** | **12/12** |
| Full system, original Phase 6 rule | 10/12 | 100% | 12/12 |
| Full system without the LLM | 12/12 | 100% | 12/12 |
| Retrieved evidence + validation (no KG, no LLM) | 12/12 | 100% | 12/12 |

**Interpretation:**
- **Physics validation and selection are what produce correct identification.**
- **The mock LLM and the KG add no measurable discovery accuracy on this benchmark.**
- **The original-rule arm shows prediction is not proof.** It predicts within 5% on 12/12 datasets while choosing the
  wrong law twice, because its wrong picks are numerically equivalent to the truth.
- **Engine caveat:** these arms use OLS on finite-difference derivatives of *clean* grids. The PINN path is evaluated
  in sections 8–9.

## 12. Benchmark results and the selection-rule change

- **50 scientific questions: 50/50.** Structure 12/12, parameters 12/12, KG parameters 5/5, derivatives 5/5,
  provenance 5/5, complexity 5/5, retrieval recall 6/6. Several categories are easy by construction.
- **25 multi-hop problems: 23/25.** data → prediction 12/12, text → derivatives 3/5, parameter → derivatives 4/4,
  nesting 4/4. Both failures are top-1 lexical retrieval returning "advection" for combined mechanisms.

**The selection-rule change, stated in full:**
- **Original rule (Phase 6):** gate, then candidates within 5% of the best held-out error, then lowest complexity,
  then lowest BIC.
- **Failure (Phase 7):** on clean two-mode diffusion data, finite-difference u_xx gives each sine mode a slightly wrong
  eigenvalue. The combination a·u_xx + b·u can correct both modes exactly (two equations, two unknowns). So
  reaction-diffusion fits the *discretization error* to about 1e-14 relative error and is the only candidate inside the
  5% band.
- **Why its support is spurious:** the fitted reaction mechanism (r = 0.004, K ≈ 4×10¹²) contributes **0.27%** of
  the dynamics. It is numerically supported but dynamically negligible.
- **Revised rule:** Phase 6 already computed term necessity at a pre-fixed 5% threshold, but used it only to
  *explain* selections. Phase 7 makes it a selection step: if the selected candidate's extra terms are inactive
  and the reduced equation is itself supported, select the reduced equation. The flag is opt-in
  (`prefer_reduced`).
- **Effect:** the original rule scores 10/12 and the revised rule 12/12 on the benchmark. Both rules agree on Phase 6
  (H1) and on the PINN generalization run (H4). The revision never removes a real reaction term (3/3).
- **Why this is not hiding the earlier result:** the original rule's results are kept and reported side by side in
  every table (`full_system_phase6_rule`, adversarial case A10, and the `selection_original_phase6_rule` field), and a
  test asserts that the original rule fails A10. Phase 6's outputs were never regenerated under the new rule. The
  change was made after observing a failure, which is disclosed here; the threshold it uses was fixed in Phase 6.

## 13. Leakage and reproducibility

- **Static audit:** 6 decision functions contain no ground-truth identifiers, and a test confirms the audit catches a
  cheating selector. `validate_and_select` takes no dataset argument.
- **Ground truth enters only scoring functions,** which run after selection.
- **Determinism:** the 50-question suite re-runs identically. Seeds are fixed in `config.yaml`.
- **Phase 8 full re-run:**
  - pytest: **93 passed, 0 failed, 0 skipped**.
  - Re-run byte-identical: Phases 2, 3 (03, 04, 05), 5 (07 including its PINN), 6 (prepare and select), Phase 7
    generalization (prepare and select), and Phase 7 evaluation (except wall-clock runtime fields).
  - Re-trained from scratch and byte-identical: Phase 4 H3, Phase 6 H1 and H3, generalization H4 and H2.
  - Not re-trained in Phase 8: the remaining cached PINN candidates (Phase 4 H1, H2, H4; Phase 6 H2, H4, K3;
    generalization H1, H3). Re-training is deterministic everywhere it was tested.

## 14. Limitations

1. **No real LLM was run.** No API key was available and the sandbox blocks `api.openai.com` (HTTP 403). All LLM
   candidates come from a canned mock whose pool contains the correct families by construction.
2. **RAG and KG show no discovery benefit.** Retrieval is lexical with wrong top-1 rankings on 9/12 datasets; the
   corpus has 5 documents; the KG is built by rules from document frontmatter.
3. **Noise robustness depends on the PINN path.** SINDy and the fast regression engine fail or abstain on noisy data.
   Only two noisy systems were tested, at a single 5% noise level.
4. **No uniqueness.** Nested models fit equally well, and the selection is a parsimony rule.
5. **Uncertainty** is an empirical finite-difference bootstrap, not Bayesian, and it disagrees with the PINN on H3.
6. **The boundary and initial conditions are given,** not inferred.
7. **Scope is synthetic 1D scalar transport-reaction PDEs** with no real experimental data.
8. **Thresholds are hand-set,** fixed before each phase's run, and not calibrated on an independent benchmark. The
   selection rule was revised after a failure (disclosed in section 12).

## 15. What was actually demonstrated

| Demonstrated | Partially demonstrated | Not demonstrated |
|---|---|---|
| Rejection of structurally invalid, physically invalid and data-inconsistent candidates | Generalization: 4 families on clean data with the regression engine; 1 new system with noisy data with the PINN | LLM-driven discovery; the effect of real LLM context |
| Inverse-PINN parameter recovery within 0.5% on two systems from noisy sparse data | Noisy-data robustness: PINN at 5% noise only; the fast engine abstains | Any accuracy benefit from RAG ranking or the KG |
| Parsimonious selection among nested models | Uncertainty: empirical bootstrap, with disagreement | Uniqueness of a selected law |
| Prediction beyond the data window (diffusion: oracle RMSE 0.00025; advection-diffusion: 0.58% relative) | RAG/KG utility: structured multi-hop queries and provenance, not accuracy | BC/IC inference |
| Gains over SINDy alone and over generation without validation | Selection rule: correct on all tests only after the disclosed revision | Real-world scientific data; PDE classes beyond 1D transport-reaction |
| Oracle-free decisions; deterministic, leakage-audited runs | | Discovery of any new physical law |

## 16. Future work

- Run `experiments/09_evaluation.py --real-llm` with a real provider to measure whether context changes the candidates.
- Use semantic embeddings and a larger, real literature corpus to test whether RAG and the KG can earn an accuracy
  contribution.
- Sweep noise levels for the PINN path, and retrain the PINN on bootstrap subsets for uncertainty.
- Calibrate the thresholds on an independent benchmark, and test on real experimental data.

## 17. Conclusion

The system turns noisy, sparse observations into a small set of candidate equations and rejects the ones that
violate physics or fail held-out data. It estimates parameters to within 0.5% with an inverse PINN, selects the
simplest supported law, and predicts accurately beyond the observed window. It does this on two different synthetic
systems, and across four mechanism families with a cheaper engine.

The evidence also says clearly *where* the capability comes from. It comes from **physics-constrained
validation and selection**, not from the (mock) LLM, retrieval ranking, or the knowledge graph. The result is a
validated, parsimonious **recovery** of known synthetic laws. It is not the discovery of new physics, and not a
demonstration of LLM-driven scientific reasoning.
