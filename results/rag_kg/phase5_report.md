# Phase 5: Scientific RAG + Knowledge Graph Report

## 1. Retrieved evidence

Query (= the evidence-based problem description):
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

| Evidence ID | Document | Section | Score | Mechanisms |
|---|---|---|---|---|
| evidence_doc_advection_chunk_003 | doc_advection | Governing equation | 0.547 | ['advection'] |
| evidence_doc_advection_chunk_004 | doc_advection | Typical observational signatures | 0.533 | ['advection'] |
| evidence_doc_diffusion_chunk_003 | doc_diffusion | Governing equation | 0.528 | ['diffusion'] |
| evidence_doc_diffusion_chunk_004 | doc_diffusion | Typical observational signatures | 0.520 | ['diffusion'] |
| evidence_doc_reaction_chunk_003 | doc_reaction | Governing equation | 0.474 | ['reaction'] |

## 2. Graph-derived relationships

Full graph: 22 entities, 40 relationships (40 sourced, 0 unsourced).

Entities by type: {'Document': 5, 'Mechanism': 5, 'Variable': 3, 'Parameter': 4, 'Equation': 5}

Relationships relevant to retrieved mechanisms (21 total):

- `mechanism_advection` **SUPPORTED_BY** `document_doc_advection` (source: doc_advection)
- `mechanism_advection` **GOVERNS** `equation_4d6a6a15f4` (source: doc_advection)
- `equation_4d6a6a15f4` **SUPPORTED_BY** `document_doc_advection` (source: doc_advection)
- `equation_4d6a6a15f4` **HAS_VARIABLE** `variable_u` (source: doc_advection)
- `equation_4d6a6a15f4` **HAS_VARIABLE** `variable_x` (source: doc_advection)
- `equation_4d6a6a15f4` **HAS_VARIABLE** `variable_t` (source: doc_advection)
- `equation_4d6a6a15f4` **HAS_PARAMETER** `parameter_c` (source: doc_advection)
- `mechanism_diffusion` **SUPPORTED_BY** `document_doc_diffusion` (source: doc_diffusion)
- `mechanism_diffusion` **GOVERNS** `equation_ae6ecf8e2e` (source: doc_diffusion)
- `equation_ae6ecf8e2e` **SUPPORTED_BY** `document_doc_diffusion` (source: doc_diffusion)
- `equation_ae6ecf8e2e` **HAS_VARIABLE** `variable_u` (source: doc_diffusion)
- `equation_ae6ecf8e2e` **HAS_VARIABLE** `variable_x` (source: doc_diffusion)
- `equation_ae6ecf8e2e` **HAS_VARIABLE** `variable_t` (source: doc_diffusion)
- `equation_ae6ecf8e2e` **HAS_PARAMETER** `parameter_D` (source: doc_diffusion)
- `mechanism_reaction` **SUPPORTED_BY** `document_doc_reaction` (source: doc_reaction)
- ... and 6 more (see knowledge_graph.json)

## 3. Generated hypotheses

### Mode A -- evidence-free

```json
{
  "mode": "evidence-free",
  "n_raw_candidates": 4,
  "n_schema_errors": 0,
  "n_structurally_valid": 4,
  "n_duplicates_removed": 0,
  "duplicate_rate": 0.0,
  "n_evidence_supported_or_grounded": 4,
  "n_successfully_compiled_for_pinn": 4
}
```

### Mode B -- evidence-grounded

```json
{
  "mode": "evidence-grounded",
  "n_raw_candidates": 4,
  "n_schema_errors": 0,
  "n_structurally_valid": 4,
  "n_duplicates_removed": 0,
  "duplicate_rate": 0.0,
  "n_evidence_supported_or_grounded": 4,
  "n_successfully_compiled_for_pinn": 4
}
```

**Mode A and Mode B produced the same candidate equations in this run** (expected with the mock provider -- see README Limitations).

| ID | Name | Equation | Composite score (Mode B) |
|---|---|---|---|
| H1 | Diffusion | `u_t = D*u_xx` | 0.835 |
| H2 | Advection | `u_t + c*u_x = 0` | 0.785 |
| H3 | Reaction-diffusion (logistic growth/decay) | `u_t = D*u_xx + r*u*(1 - u/K)` | 0.757 |
| H4 | Advection-diffusion | `u_t + c*u_x = D*u_xx` | 0.750 |

## 4. PINN validation

| ID | Status | Data loss | Physics residual | Prediction RMSE | Parameters |
|---|---|---|---|---|---|
| H1 | supported by the available observations and physics constraints | 0.000132 | 0.000026 | 0.000849 | {'D': 0.09990929812192917} |

## 5. Final interpretation

Retrieval surfaced documented background on the mechanisms listed above, used as GENERAL context, not as a stated answer for this dataset (no fitted parameter value or dataset-specific equation claim was ever placed in the context -- see README's answer-leakage-prevention section). The knowledge graph aggregates which variables, parameters, and generic equation forms are documented for each mechanism, with full source provenance. Scientific acceptance/rejection of any hypothesis remains entirely the responsibility of Phase 3's PINN validation (section 4 above) -- nothing in this report should be read as RAG or the knowledge graph having 'proven' anything.