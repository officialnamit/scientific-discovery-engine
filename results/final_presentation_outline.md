# Presentation outline — Generative Scientific Discovery Engine (12 slides)

Every number here comes from `results/final_report.md`.

---

**1. Problem**
- Given sparse, noisy observations of a field u(x,t) and *no* governing equation: propose laws, test them against data
  and physics, reject the wrong ones, estimate the parameters, and predict outside the data.
- Visual: the scattered noisy observations (`data/generated/verification_plot.png`).

**2. Motivation**
- A neural network can fit data without obeying physics. A PINN given the right PDE never has to *find* it.
  SINDy finds equations but breaks on noise.
- Measured on this project:
  - SINDy fails on 400 points at 5% noise.
  - A data-only MLP is 40× worse than the selected PINN outside the data (Phase 6).

**3. Proposed system**
- Generate candidate laws, constrain them physically, test each with an inverse PINN, select the simplest supported
  one, and predict.
- Principle: *the LLM proposes; physics and held-out data decide.*

**4. Architecture**
- RAG + KG → hypotheses (LLM ∪ KG templates) → structural and physics filters → inverse PINN per candidate →
  oracle-free gate → parsimony selection → OOD and forward prediction.
- One reusable PINN. Equations are compiled with sympy; nothing is hand-coded.

**5. Equation discovery (baseline)**
- On clean or sparse data, SINDy recovers `u_t = 0.1009·u_xx` (0.9% error).
- On noisy data it returns a 5-term spurious equation.
- Lesson learned: a single-sine initial condition makes u and u_xx indistinguishable (u_xx = −π²u). A two-mode initial
  condition was used from Phase 2 on.

**6. Hypothesis generation + RAG/KG**
- A 5-document corpus with provenance, and a KG with 22 entities and 40 relationships, all sourced.
- The KG contributed a candidate the LLM did not propose (pure reaction).
- Be candid:
  - The LLM is a mock.
  - Top-1 retrieval is wrong for 9/12 datasets (lexical embedding).
  - The ablation shows no accuracy gain from the KG or the LLM.

**7. PINN validation**
- Forward: RMSE 0.0007.
- Inverse: D̂ = 0.0999 from an initial guess of 0.2 (0.11% error).
- Diffusion vs. advection: OOD RMSE 0.0012 vs. 0.33, so advection is rejected.

**8. Physics-constrained selection**
- Diffusion data, 5 candidates: 2 are rejected. The 3 survivors all contain diffusion.
- H4 had the *lowest* raw error, but its advection term is inactive, so H1 `u_t = D·u_xx` is selected
  (D̂ = 0.10002). BIC agrees.
- Message: "can explain the data" ≠ "needs this term".

**9. Generalization (headline slide)**
- A new system, advection-diffusion, from 400 noisy points.
- Diffusion, advection and reaction-diffusion are rejected; advection-diffusion is selected.
- ĉ = 0.399, D̂ = 0.0201 (errors under 0.4%). The prediction beyond the data has 0.58% error.
- Visual: the candidate table from `results/demo/demo_run.md`.

**10. Adversarial + ablation results**
- Adversarial: 20/20.
  - D < 0 rejected (A06).
  - Occam does not override fit (A13).
  - SINDy's dropped term is recovered (A12).
  - 3 cases pass by abstention.
- Ablation, correct model on 12 datasets: full system 12/12; SINDy 8/12; generation without validation 3/12.
- Benchmark: 50/50 questions; multi-hop 23/25.
- Disclosed: the selection rule was revised after a failure on clean two-mode data. Original rule 10/12, revised
  12/12; both are reported.

**11. Limitations**
- No real LLM was run (no key, API blocked). The RAG and KG show no accuracy benefit.
- Synthetic 1D systems only. The boundary and initial conditions are given. Uncertainty is a bootstrap only.
- No uniqueness: selection is by parsimony. Thresholds are hand-set.

**12. Conclusion**
- The system validates, rejects and predicts. Its demonstrated value comes from physics-constrained validation and
  selection.
- What it achieved is recovery of known synthetic laws, not the discovery of new physics.
- Next steps: a real LLM run, a semantic and real corpus, a noise sweep, and real data.
