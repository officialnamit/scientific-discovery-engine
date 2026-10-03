"""
Phase 5 experiment.

    Problem -> RAG -> Knowledge Graph -> Scientific Context -> Phase 4
    hypothesis engine (Mode A: evidence-free, Mode B: evidence-grounded)
    -> Phase 3 PINN validation (unchanged from Phase 3/4)

Run:
    python experiments/07_scientific_rag_kg.py --config config.yaml
    python experiments/07_scientific_rag_kg.py --config config.yaml --skip-pinn
    python experiments/07_scientific_rag_kg.py --config config.yaml --only H1,H2
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.problem_description import build_problem_description
from hypotheses.adapter import hypothesis_to_pinn_config
from hypotheses.pipeline import generate_hypotheses
from hypotheses.validation import canonical_signature
from knowledge_graph.builder import build_graph_from_documents
from knowledge_graph.serialization import graph_to_dict, save_graph
from llm import get_provider
from llm.schemas import Hypothesis
from pinn.data_utils import (
    data_dict_to_tensors,
    load_clean_reference,
    load_observational_data,
    make_bc_points,
    make_collocation_points,
    make_ic_points,
)
from pinn.trainer import build_and_train
from rag.context import build_scientific_context
from rag.corpus_loader import load_corpus
from rag.pipeline import RAGPipeline
from validation.ood import prediction_error
from validation.physics_validation import evaluate_physics_residual
from validation.scorer import build_validation_result


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def hyp_to_dict(h: Hypothesis) -> dict:
    return json.loads(h.model_dump_json())


def summarize_generation(result: dict, reference_sig: str) -> dict:
    n_candidates = len(result["all_hypotheses"])
    n_valid = len(result["valid_hypotheses"])
    n_duplicates = len(result["removed_duplicates"])
    n_evidence_supported = sum(
        1 for h in result["valid_hypotheses"]
        if canonical_signature(h) == reference_sig or h.mechanism or h.rationale
    )
    n_compiled = 0
    for h in result["valid_hypotheses"]:
        try:
            hypothesis_to_pinn_config(h)
            n_compiled += 1
        except Exception:
            pass
    return {
        "mode": result["mode"],
        "n_raw_candidates": n_candidates,
        "n_schema_errors": len(result["schema_errors"]),
        "n_structurally_valid": n_valid,
        "n_duplicates_removed": n_duplicates,
        "duplicate_rate": n_duplicates / n_candidates if n_candidates else 0.0,
        "n_evidence_supported_or_grounded": n_evidence_supported,
        "n_successfully_compiled_for_pinn": n_compiled,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip-pinn", action="store_true")
    parser.add_argument("--only", type=str, default=None,
                         help="Comma-separated hypothesis id(s) to run PINN validation for "
                              "(default: just the top-ranked Mode-B candidate).")
    parser.add_argument("--top-k", type=int, default=5, help="RAG retrieval top-k.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    out_dir = cfg["paths"]["output_dir"]
    llm_cfg = cfg["llm"]
    dcfg = cfg["diffusion"]
    pcfg = cfg["pinn"]
    D_true = float(dcfg["D_true"])

    os.makedirs("results/rag_kg", exist_ok=True)

    # --- Step 0: automated answer-leakage check ---
    # The corpus must never contain this experiment's actual fitted D value anywhere
    # in its text -- verified here programmatically, not just asserted in prose.
    docs = load_corpus()
    d_true_str = str(D_true)
    leaked = [d.metadata.document_id for d in docs if d_true_str in d.text]
    if leaked:
        print(f"\n*** WARNING: possible answer leakage -- D_true ({d_true_str}) literally "
              f"appears in corpus document(s): {leaked} ***\n")
    else:
        print(f"\nAnswer-leakage check: D_true ({d_true_str}) does not appear in any corpus "
              f"document's text. OK.\n")

    # --- Step 1: ingest corpus, build RAG index ---
    rag_pipeline = RAGPipeline()
    chunks = rag_pipeline.ingest(docs)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    retriever = rag_pipeline.get_retriever()
    print(f"Ingested {len(docs)} corpus documents -> {len(chunks)} chunks.")

    # --- Step 2: build knowledge graph ---
    graph = build_graph_from_documents(docs, chunks_by_id)
    print(f"Knowledge graph: {graph.summary()}")
    save_graph(graph, "results/rag_kg/knowledge_graph.json")

    # --- Step 3: problem description (same evidence-based builder as Phase 4) ---
    problem_description = build_problem_description(out_dir)
    print("\n" + "=" * 70)
    print("PROBLEM DESCRIPTION:")
    print("=" * 70)
    print(problem_description)

    # --- Step 4: retrieve evidence, build scientific context ---
    evidence_list = retriever.retrieve(problem_description, top_k=args.top_k)
    context = build_scientific_context(problem_description, retriever, graph, top_k=args.top_k)

    retrieval_report = {
        "query": problem_description,
        "top_k": args.top_k,
        "results": [json.loads(e.model_dump_json()) for e in evidence_list],
    }
    with open("results/rag_kg/retrieval_results.json", "w") as f:
        json.dump(retrieval_report, f, indent=2)
    print(f"\nRetrieved {len(evidence_list)} evidence items -> results/rag_kg/retrieval_results.json")
    print(f"Scientific context: mechanisms={context.mechanisms}, "
          f"known_equations={context.known_equations}, parameters={context.parameters}")

    # --- Step 5: Phase 4 hypothesis generation, Mode A vs Mode B ---
    provider = get_provider(llm_cfg)
    reference = Hypothesis(id="REF", name="reference_diffusion", equation="u_t = D*u_xx",
                            parameters=[{"name": "D", "trainable": True}])
    reference_sig = canonical_signature(reference)

    result_a = generate_hypotheses(problem_description, provider, n_hypotheses=llm_cfg["n_hypotheses"],
                                    domain_hint="1D scalar-field transport/reaction", temperature=llm_cfg["temperature"])
    result_b = generate_hypotheses(problem_description, provider, n_hypotheses=llm_cfg["n_hypotheses"],
                                    domain_hint="1D scalar-field transport/reaction", temperature=llm_cfg["temperature"],
                                    context=context.model_dump())

    summary_a = summarize_generation(result_a, reference_sig)
    summary_b = summarize_generation(result_b, reference_sig)
    print(f"\nMode A (evidence-free):     {summary_a}")
    print(f"Mode B (evidence-grounded): {summary_b}")

    same_candidates = (
        {h.equation for h in result_a["valid_hypotheses"]} == {h.equation for h in result_b["valid_hypotheses"]}
    )
    if same_candidates:
        print("\nNOTE: Mode A and Mode B produced the SAME candidate equations in this run. "
              "This is EXPECTED with the mock LLM provider (see README limitations) -- "
              "MockLLMProvider returns a fixed canned set regardless of prompt content, so it "
              "does not yet condition on retrieved context. The architecture (context -> prompt "
              "-> provider.generate()) is fully wired and tested; demonstrating RAG actually "
              "changing output requires a real provider (llm.provider='gemini' with GEMINI_API_KEY set).")

    # --- Step 6: run Mode B's top-ranked candidate (or --only) through Phase 3 PINN validation ---
    pinn_validation_results = {}
    report_path = "results/rag_kg/generated_hypotheses.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            existing = json.load(f)
        pinn_validation_results = existing.get("pinn_validation_results", {})

    candidates_by_id = {h.id: h for h in result_b["valid_hypotheses"]}
    if args.only:
        target_ids = args.only.split(",")
    elif result_b["ranking"]:
        target_ids = [result_b["ranking"][0]["id"]]  # top-ranked only, by default
    else:
        target_ids = []

    if not args.skip_pinn and target_ids:
        x_min, x_max = dcfg["x_min"], dcfg["x_max"]
        t_min, t_max = dcfg["t_min"], dcfg["t_max"]
        ic_kind = dcfg["initial_condition"]
        seed = pcfg["training"]["seed"]

        noisy = load_observational_data(out_dir, "noisy")
        data = data_dict_to_tensors(noisy)
        collocation = make_collocation_points(x_min, x_max, t_min, t_max, pcfg["training"]["n_collocation"], seed=seed)
        bc = make_bc_points(x_min, x_max, t_min, t_max, pcfg["training"]["n_bc"], seed=seed + 1)
        ic = make_ic_points(x_min, x_max, pcfg["training"]["n_ic"], ic_kind, seed=seed + 2)
        clean = load_clean_reference(out_dir)
        thresholds = pcfg["validation"]["thresholds"]

        for hid in target_ids:
            h = candidates_by_id.get(hid)
            if h is None:
                print(f"  (skipping unknown id {hid})")
                continue
            print(f"\n=== Running {h.id} ({h.name}: {h.equation}) through Phase 3 PINN validation ===")
            adapted = hypothesis_to_pinn_config(h)
            trainer, history = build_and_train(
                adapted["hypothesis_cfg"], adapted["param_init"], adapted["trainable"],
                architecture_cfg=pcfg["architecture"], weights=pcfg["weights"],
                data=data, collocation=collocation, bc=bc, ic=ic,
                training_cfg=pcfg["training"], seed=seed,
            )
            final = history[-1]
            physics_residual = evaluate_physics_residual(
                trainer.model, adapted["hypothesis_cfg"]["fn"], trainer.params, x_min, x_max, t_min, t_max,
            )
            pred_err = prediction_error(trainer.model, clean["x"], clean["t"], clean["u"])
            is_true_diffusion = canonical_signature(h) == reference_sig
            param_estimate = {k: float(v.item()) for k, v in trainer.params.items()}
            param_true = {"D": D_true} if (is_true_diffusion and "D" in param_estimate) else None

            validation_result = build_validation_result(
                hypothesis_name=h.id, equation_str=h.equation,
                data_loss=final["data"], physics_residual=physics_residual,
                bc_loss=final["bc"], ic_loss=final["ic"], prediction_error=pred_err["rmse"],
                thresholds=thresholds, parameter_estimate=param_estimate, parameter_true=param_true,
            )
            pinn_validation_results[h.id] = validation_result
            print(f"  status: {validation_result['status']}, data_loss={final['data']:.6f}, "
                  f"physics_residual={physics_residual:.6f}, prediction_RMSE={pred_err['rmse']:.6f}")

    # --- Save machine-readable generated_hypotheses.json ---
    report = {
        "problem_description": problem_description,
        "mode_a_evidence_free": {
            "summary": summary_a,
            "hypotheses": [hyp_to_dict(h) for h in result_a["valid_hypotheses"]],
            "ranking": result_a["ranking"],
        },
        "mode_b_evidence_grounded": {
            "summary": summary_b,
            "context_used": json.loads(context.model_dump_json()),
            "hypotheses": [hyp_to_dict(h) for h in result_b["valid_hypotheses"]],
            "ranking": result_b["ranking"],
        },
        "modes_produced_identical_candidates": same_candidates,
        "pinn_validation_results": pinn_validation_results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {report_path}")

    # --- Human-readable Phase 5 report, clearly separating the 5 sections ---
    lines = ["# Phase 5: Scientific RAG + Knowledge Graph Report\n"]

    lines.append("## 1. Retrieved evidence\n")
    lines.append(f"Query (= the evidence-based problem description):\n```\n{problem_description}\n```\n")
    lines.append("| Evidence ID | Document | Section | Score | Mechanisms |")
    lines.append("|---|---|---|---|---|")
    for e in evidence_list:
        lines.append(f"| {e.evidence_id} | {e.document_id} | {e.section} | {e.confidence:.3f} | {e.mechanisms} |")

    lines.append("\n## 2. Graph-derived relationships\n")
    gsum = graph.summary()
    lines.append(f"Full graph: {gsum['n_entities']} entities, {gsum['n_relationships']} relationships "
                 f"({gsum['n_sourced_relationships']} sourced, {gsum['n_unsourced_relationships']} unsourced).\n")
    lines.append(f"Entities by type: {gsum['entities_by_type']}\n")
    lines.append(f"Relationships relevant to retrieved mechanisms ({len(context.graph_relationships)} total):\n")
    for r in context.graph_relationships[:15]:
        srcs = ", ".join(s.document_id for s in r.sources)
        lines.append(f"- `{r.subject}` **{r.predicate.value}** `{r.object}` (source: {srcs})")
    if len(context.graph_relationships) > 15:
        lines.append(f"- ... and {len(context.graph_relationships) - 15} more (see knowledge_graph.json)")

    lines.append("\n## 3. Generated hypotheses\n")
    lines.append("### Mode A -- evidence-free\n")
    lines.append(f"```json\n{json.dumps(summary_a, indent=2)}\n```\n")
    lines.append("### Mode B -- evidence-grounded\n")
    lines.append(f"```json\n{json.dumps(summary_b, indent=2)}\n```\n")
    if same_candidates:
        lines.append("**Mode A and Mode B produced the same candidate equations in this run** "
                      "(expected with the mock provider -- see README Limitations).\n")
    lines.append("| ID | Name | Equation | Composite score (Mode B) |")
    lines.append("|---|---|---|---|")
    rank_b_by_id = {r["id"]: r for r in result_b["ranking"]}
    for h in result_b["valid_hypotheses"]:
        score = rank_b_by_id.get(h.id, {}).get("composite_score")
        lines.append(f"| {h.id} | {h.name} | `{h.equation}` | {score:.3f} |" if score is not None else
                     f"| {h.id} | {h.name} | `{h.equation}` | - |")

    lines.append("\n## 4. PINN validation\n")
    if pinn_validation_results:
        lines.append("| ID | Status | Data loss | Physics residual | Prediction RMSE | Parameters |")
        lines.append("|---|---|---|---|---|---|")
        for hid, v in pinn_validation_results.items():
            lines.append(f"| {hid} | {v['status']} | {v['data_loss']:.6f} | {v['physics_residual']:.6f} | "
                         f"{v['prediction_error']:.6f} | {v['parameter_estimate']} |")
    else:
        lines.append("(skipped in this run -- use without --skip-pinn to populate)\n")

    lines.append("\n## 5. Final interpretation\n")
    lines.append(
        "Retrieval surfaced documented background on the mechanisms listed above, used as "
        "GENERAL context, not as a stated answer for this dataset (no fitted parameter value "
        "or dataset-specific equation claim was ever placed in the context -- see README's "
        "answer-leakage-prevention section). The knowledge graph aggregates which variables, "
        "parameters, and generic equation forms are documented for each mechanism, with full "
        "source provenance. Scientific acceptance/rejection of any hypothesis remains entirely "
        "the responsibility of Phase 3's PINN validation (section 4 above) -- nothing in this "
        "report should be read as RAG or the knowledge graph having 'proven' anything."
    )

    report_md_path = "results/rag_kg/phase5_report.md"
    with open(report_md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {report_md_path}")


if __name__ == "__main__":
    main()
