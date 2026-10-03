# Phase 7 -- Evaluation, Adversarial Testing & Generalization

All numbers below are produced by `experiments/09_evaluation.py` and `experiments/10_generalization.py`.

## Headline

- 50 scientific questions: **50/50**; 25 multi-hop problems: **23/25**; 20 adversarial cases: **20/20** passed (see qualifications).
- Discovery generalizes across 4 mechanism families with the regression engine (full system 12/12), and to a second physical system with the full PINN pipeline (advection-diffusion selected, see section 5).
- Ablation: physics validation is the component that produces correct identification; the mock LLM and the KG add no measurable discovery accuracy on this benchmark (section 4).

## 1. 50 scientific questions

| Category | Correct |
|---|---|
| structure_identification | 12/12 |
| parameter_estimation | 12/12 |
| kg_parameters | 5/5 |
| equation_derivatives | 5/5 |
| kg_provenance | 5/5 |
| complexity | 5/5 |
| retrieval | 6/6 |

Retrieval questions score *recall* (true mechanism anywhere in the top-5 chunks). **Top-1 retrieval accuracy is 1/6**: the lexical embedding ranks 'advection' first for every dataset. With 5 corpus documents, top-5 recall is weak evidence. The KG/compiler/complexity questions check internal consistency against answer keys written from the corpus frontmatter; they are easy by construction.

## 2. 25 multi-hop problems

| Category | Hops | Correct |
|---|---|---|
| data_to_prediction | 6 | 12/12 |
| text_to_derivatives | 4 | 3/5 |
| parameter_to_derivatives | 3 | 4/4 |
| nesting | 3 | 4/4 |

- FAIL M16: expected {'mechanism': 'reaction-diffusion', 'derivatives': ['u_t', 'u_xx']}, got {'mechanism': 'advection', 'parameters': ['c'], 'derivatives': ['u_t', 'u_x']}

- FAIL M17: expected {'mechanism': 'advection-diffusion', 'derivatives': ['u_t', 'u_x', 'u_xx']}, got {'mechanism': 'advection', 'parameters': ['c'], 'derivatives': ['u_t', 'u_x']}

Both failures come from top-1 retrieval returning plain 'advection' for observation text describing a combined mechanism (lexical embedding limitation).

## 3. 20 adversarial cases

| ID | Target | Expected safe behavior | Actual | Pass |
|---|---|---|---|---|
| A01 | incorrect candidate | rejected (structural) | `"rejected"` | PASS |
| A02 | degenerate equation | rejected (not an evolution equation) | `"rejected"` | PASS |
| A03 | degenerate equation | rejected | `"rejected"` | PASS |
| A04 | physically invalid parameters | rejected | `"rejected"` | PASS |
| A05 | physically invalid parameters | rejected | `"rejected"` | PASS |
| A06 | physically invalid parameters | diffusion candidate rejected by the D>0 bound; no diffusion selected | `{"fitted_D": -0.100287777944186, "selected_family": null}` | PASS |
| A07 | incorrect candidate | 1 unique candidate | `"1 unique"` | PASS |
| A08 | incorrect candidate | prose -> 0 candidates; invalid item dropped, valid kept | `{"prose_candidates": 0, "kept": ["G1"]}` | PASS |
| A09 | degenerate equation | identifiability diagnostic flags u vs u_xx as indistinguishable | `{"max_abs_correlation": 1.0, "pair": ["u", "u_xx"]}` | PASS |
| A10 | degenerate equation / nested model | diffusion selected; reaction terms reported inactive | `{"original_rule": "reaction-diffusion", "revised_rule": "diffusion", "reduction": {"from": "H3", "to": "H1", "` | PASS |
| A11 | nested models | diffusion selected; nesting candidates may be supported but not selected | `{"selected": "diffusion", "nesting_candidates_supported": 1}` | PASS |
| A12 | model-selection failure | full system selects advection-diffusion (not the simpler advection SINDy reports) | `{"sindy_alone": "advection", "full_system": "advection-diffusion"}` | PASS |
| A13 | nested models | parsimony does NOT override fit: diffusion fails the gate, reaction-diffusion selected | `{"selected": "reaction-diffusion", "diffusion_gate": false, "diffusion_rel_err": 0.851}` | PASS |
| A14 | incorrect candidate | diffusion rejected by the gate; advection selected | `{"selected": "advection", "diffusion_rel_err": 1.0, "diffusion_gate": false}` | PASS |
| A15 | misleading correlation | retrieval does not decide: diffusion still selected by validation | `{"top1_retrieved": "advection", "selected": "diffusion"}` | PASS |
| A16 | model-selection failure | advection fails the gate (residual from missing diffusion > 5%); advection-diffusion selected | `{"selected": "advection-diffusion", "advection_rel_err": 0.457}` | PASS |
| A17 | noisy observations | no false discovery: diffusion or nothing selected | `{"selected": null, "n_gate_passed": 0}` | PASS |
| A18 | noisy observations | no false discovery: diffusion or nothing selected | `{"selected": null, "n_gate_passed": 0}` | PASS |
| A19 | sparse observations | no false discovery: diffusion or nothing selected | `{"selected": null, "n_gate_passed": 0}` | PASS |
| A20 | answer leakage / injection | real corpus: no leaks; poisoned doc flagged; it adds 0 KG relationships (no declared structure) | `{"real_corpus_leaks": {}, "poison_flagged": {"doc_poison": ["0.1"]}, "kg_relationships_added": 0}` | PASS |

**Qualifications.** A17-A19 (1% noise, 10% noise, 10% sparse) pass by **abstention** -- no candidate clears the gate -- not by discovery: the regression engine has no noise robustness. The PINN path handles this regime (Phase 6: 5% noise, 400 points; section 5). A10 passes only under the revised Phase 7 selection rule; the original Phase 6 rule selects reaction-diffusion there.

## 4. Ablation (12 datasets, 4 mechanism families)

| Arm | Correct model | By family | Wrong candidates rejected | Params within 5% | Median held-out rel. error | Median OOD rel. RMSE | OOD within 5% | Runtime/dataset |
|---|---|---|---|---|---|---|---|---|
| discovery_only | 0.67 | {'advection': '3/3', 'advection-diffusion': '0/3', 'diffusion': '3/3', 'reaction-diffusion': '2/3'} | - | 0.67 | 0.0028 | 0.0133 | 0.50 | 0.26s |
| rag_generation | 0.25 | {'advection': '0/3', 'advection-diffusion': '0/3', 'diffusion': '3/3', 'reaction-diffusion': '0/3'} | 0.00 | 0.25 | 0.8791 | 1.0465 | 0.25 | 0.20s |
| rag_kg_generation | 0.25 | {'advection': '0/3', 'advection-diffusion': '0/3', 'diffusion': '3/3', 'reaction-diffusion': '0/3'} | 0.00 | 0.25 | 0.8791 | 1.0465 | 0.25 | 0.18s |
| full_system | 1.00 | {'advection': '3/3', 'advection-diffusion': '3/3', 'diffusion': '3/3', 'reaction-diffusion': '3/3'} | 1.00 | 1.00 | 0.0025 | 0.0041 | 1.00 | 0.18s |
| full_system_phase6_rule | 0.83 | {'advection': '3/3', 'advection-diffusion': '3/3', 'diffusion': '1/3', 'reaction-diffusion': '3/3'} | 1.00 | 0.83 | 0.0025 | 0.0040 | 1.00 | 0.19s |
| full_without_llm | 1.00 | {'advection': '3/3', 'advection-diffusion': '3/3', 'diffusion': '3/3', 'reaction-diffusion': '3/3'} | 1.00 | 1.00 | 0.0025 | 0.0041 | 1.00 | 0.15s |
| rag_validation_no_kg_no_llm | 1.00 | {'advection': '3/3', 'advection-diffusion': '3/3', 'diffusion': '3/3', 'reaction-diffusion': '3/3'} | 1.00 | 1.00 | 0.0025 | 0.0041 | 1.00 | 0.13s |

Without validation, arms 2-3 select `{'diffusion'}` for every dataset: generation alone repeats the top-ranked prior. Equation discovery alone (SINDy) misses all 3 advection-diffusion systems (its threshold drops the small real u_xx term) and one reaction-diffusion system. `full_without_llm` and `rag_validation_no_kg_no_llm` match the full system exactly: on this corpus the decisive ingredients are (a) a pool containing the true structure -- retrieved evidence alone supplies it -- and (b) physics validation + selection. The KG's demonstrated value is structured multi-hop querying and provenance (section 2), not discovery accuracy. The revised selection rule changes 2 of 12 outcomes (both diffusion datasets).

## 5. Generalization: second physical system, full PINN pipeline

Hidden truth `u_t + c*u_x = D*u_xx`, c=0.4, D=0.02, x in [0,1.5]; 400 random points, 5% noise; same splits, gate and selection as Phase 6.

| Candidate | Fitted params | Held-out RMSE | Status | OOD RMSE (noisy obs) | OOD RMSE vs clean (oracle) |
|---|---|---|---|---|---|
| H1 `u_t = D*u_xx` | D=0.07515 | 0.0958 | rejected under the tested conditions | 0.1153 | 0.1110 |
| H2 `u_t + c*u_x = 0` | c=0.3685 | 0.0635 | rejected under the tested conditions | 0.1120 | 0.1127 |
| H3 `u_t = D*u_xx + r*u*(1 - u/K)` | D=0.1171, r=1.217, K=1.439 | 0.0841 | rejected under the tested conditions | 0.1006 | 0.0946 |
| H4 `u_t + c*u_x = D*u_xx` | c=0.399, D=0.02008 | 0.0099 | supported by the available observations and physics constraints | 0.0096 | 0.0010 |
| data-only MLP | - | 0.0124 | - | 0.0119 | 0.0068 |

Selected: **H4 (advection-diffusion)** under the revised rule and **H4** under the original rule. Parameter errors (oracle): {'c': '0.26%', 'D': '0.40%'}. Novel prediction (selected law solved to t=1.3 from the IC, no observations): relative RMSE 0.58%. SINDy on the same noisy data returned a spurious equation (see generalization/prepare.json).

## 6. Leakage and reproducibility

- Decision functions referencing ground-truth identifiers: none (checked 6 functions).
- Corpus documents containing any of the 15 hidden benchmark values: none.
- Determinism: re-running the 50-question suite gives identical answers: **True**. Seeds fixed in config.yaml (phase7.seed, phase6.split_seed, pinn.training.seed).
- Benchmark scoring reads ground truth only in `score_run` / `truth_field` / `oracle_evaluation`, after selection.

## 7. Real LLM

Available: **False**. not requested. No real-LLM results exist. Nothing in this report is attributed to a real LLM.

## 8. What was and was not demonstrated

**Demonstrated:** rejection of structurally invalid, physically invalid, and data-inconsistent candidates; parsimonious selection among nested models; correct identification across 4 mechanism families (clean data, regression engine) and on a second system from noisy sparse data (PINN); parameter recovery within 0.5%; prediction beyond the observed window; superiority over SINDy-alone and over generation-without-validation; deterministic, leakage-audited runs.

**Not demonstrated:** LLM-driven hypothesis generation (mock only); any discovery-accuracy benefit from RAG ranking or the KG (lexical top-1 retrieval is wrong for 9/12 datasets); robustness of the cheap engine to noise (it abstains); uniqueness of the selected law; generalization beyond 1D scalar transport-reaction PDEs; real experimental data.

**Disclosed rule change:** Phase 7 promoted Phase 6's term-necessity criterion into selection after observing that the original rule fails on noise-free two-mode diffusion data. Both rules are reported everywhere.
