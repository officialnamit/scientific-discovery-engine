"""
Phase 4 experiment.

1. Build a problem description from ACTUAL computed statistics of the
   Phase 1 dataset (amplitude decay, peak-position stability, spatial
   smoothing) -- observations only, never the governing equation or D.
2. Generate candidate hypotheses (mock provider by default; set
   llm.provider: "gemini" + GEMINI_API_KEY env var for a real LLM).
3. Schema-validate, deterministically validate, deduplicate, rank.
4. Convert each surviving candidate to a Phase 3 PINN config via
   hypotheses/adapter.py and run it through the SAME PINN
   trainer/validator used in Phase 3 -- no second PINN implementation.
5. Save machine-readable + human-readable reports.

Run:
    python experiments/06_hypothesis_generation.py --config config.yaml
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
from validation.ood import prediction_error
from validation.physics_validation import evaluate_physics_residual
from validation.scorer import build_validation_result


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip-pinn", action="store_true",
                         help="Only generate/validate/rank hypotheses; skip the (slow) Phase 3 PINN runs.")
    parser.add_argument("--only", type=str, default=None,
                         help="Comma-separated hypothesis id(s) (e.g. 'H1' or 'H1,H2') to run PINN "
                              "validation for in this invocation; merges into any existing report "
                              "JSON. Since the mock provider is deterministic, re-running "
                              "generation produces the same candidate set every time, so this can "
                              "be called once per hypothesis across separate (time-limited) runs.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    out_dir = cfg["paths"]["output_dir"]
    llm_cfg = cfg["llm"]
    dcfg = cfg["diffusion"]
    pcfg = cfg["pinn"]
    D_true = float(dcfg["D_true"])

    problem_description = build_problem_description(out_dir)
    print("=" * 70)
    print("PROBLEM DESCRIPTION GIVEN TO THE HYPOTHESIS GENERATOR:")
    print("=" * 70)
    print(problem_description)
    print("=" * 70)

    provider = get_provider(llm_cfg)
    print(f"\nUsing LLM provider: {provider.name}")

    result = generate_hypotheses(
        problem_description, provider,
        n_hypotheses=llm_cfg["n_hypotheses"], domain_hint="1D scalar-field transport/reaction",
        temperature=llm_cfg["temperature"],
    )

    print(f"\nGenerated {len(result['all_hypotheses'])} raw hypotheses, "
          f"{len(result['schema_errors'])} schema errors, "
          f"{len(result['invalid_hypotheses'])} failed deterministic validation, "
          f"{len(result['removed_duplicates'])} duplicates removed, "
          f"{len(result['valid_hypotheses'])} unique valid candidates remain.")

    for h in result["all_hypotheses"]:
        v = result["validations"][h.id]
        print(f"\n[{h.id}] {h.name}: {h.equation}")
        print(f"    valid={v['valid']}  warnings={v['warnings']}  errors={v['errors']}")

    if result["removed_duplicates"]:
        print("\nDuplicates removed:")
        for d in result["removed_duplicates"]:
            print(f"  {d['id']} ({d['name']}) is structurally a duplicate of {d['duplicate_of']}")

    print("\nRanking (investigation-order triage, NOT a scientific validity judgment):")
    for r in result["ranking"]:
        print(f"  {r['id']} ({r['name']}, {r['equation']}): composite_score={r['composite_score']:.3f}")

    # --- Check: does the correct diffusion equation appear as a candidate? ---
    reference = Hypothesis(
        id="REF", name="reference_diffusion", equation="u_t = D*u_xx",
        parameters=[{"name": "D", "trainable": True}],
    )
    reference_sig = canonical_signature(reference)
    matches = [h for h in result["valid_hypotheses"] if canonical_signature(h) == reference_sig]
    if matches:
        print(f"\n*** Structural match to true diffusion equation found: "
              f"{matches[0].id} ({matches[0].name}), equation='{matches[0].equation}' ***")
    else:
        print("\n*** No generated candidate is structurally equivalent to the true diffusion "
              "equation u_t=D*u_xx. ***")

    os.makedirs("results/hypotheses", exist_ok=True)

    report_path = "results/hypotheses/phase4_generated_hypotheses.json"
    pinn_validation_results = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            existing_report = json.load(f)
        pinn_validation_results = existing_report.get("pinn_validation_results", {})
        print(f"\nLoaded {len(pinn_validation_results)} previously computed PINN validation "
              f"result(s) from {report_path} to merge with this run.")

    candidates_to_run = result["valid_hypotheses"]
    if args.only:
        only_ids = set(args.only.split(","))
        candidates_to_run = [h for h in candidates_to_run if h.id in only_ids]
        print(f"\n--only specified: restricting this invocation's PINN runs to {sorted(only_ids)}")

    # --- Optionally run each unique candidate through the EXISTING Phase 3 PINN validator ---
    if not args.skip_pinn and candidates_to_run:
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

        for h in candidates_to_run:
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
                trainer.model, adapted["hypothesis_cfg"]["fn"], trainer.params,
                x_min, x_max, t_min, t_max,
            )
            pred_err = prediction_error(trainer.model, clean["x"], clean["t"], clean["u"])

            is_true_diffusion = canonical_signature(h) == reference_sig
            param_estimate = {k: float(v.item()) for k, v in trainer.params.items()}
            param_true = {"D": D_true} if (is_true_diffusion and "D" in param_estimate) else None

            validation_result = build_validation_result(
                hypothesis_name=h.id, equation_str=h.equation,
                data_loss=final["data"], physics_residual=physics_residual,
                bc_loss=final["bc"], ic_loss=final["ic"],
                prediction_error=pred_err["rmse"], thresholds=thresholds,
                parameter_estimate=param_estimate, parameter_true=param_true,
            )
            pinn_validation_results[h.id] = validation_result
            print(f"  status: {validation_result['status']}")
            print(f"  data_loss={final['data']:.6f}  physics_residual={physics_residual:.6f}  "
                  f"prediction_RMSE={pred_err['rmse']:.6f}")

    # --- Save machine-readable report ---
    def hyp_to_dict(h: Hypothesis) -> dict:
        return json.loads(h.model_dump_json())

    report = {
        "problem_description": problem_description,
        "provider": result["provider"],
        "all_hypotheses": [hyp_to_dict(h) for h in result["all_hypotheses"]],
        "schema_errors": result["schema_errors"],
        "validations": result["validations"],
        "removed_duplicates": result["removed_duplicates"],
        "valid_hypotheses": [hyp_to_dict(h) for h in result["valid_hypotheses"]],
        "ranking": result["ranking"],
        "structural_match_to_true_diffusion": matches[0].id if matches else None,
        "pinn_validation_results": pinn_validation_results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved machine-readable report to {report_path}")

    # --- Human-readable markdown report ---
    lines = ["# Phase 4: Generated Hypotheses Report\n", "## Problem description\n", "```", problem_description, "```\n"]
    lines.append(f"Provider: `{result['provider']}`\n")
    lines.append("## Candidates\n")
    lines.append("| ID | Name | Equation | Valid | Composite score | PINN status |")
    lines.append("|---|---|---|---|---|---|")
    rank_by_id = {r["id"]: r for r in result["ranking"]}
    for h in result["all_hypotheses"]:
        v = result["validations"][h.id]
        rank = rank_by_id.get(h.id)
        score_str = f"{rank['composite_score']:.3f}" if rank else "-"
        pinn_status = pinn_validation_results.get(h.id, {}).get("status", "-")
        lines.append(f"| {h.id} | {h.name} | `{h.equation}` | {v['valid']} | {score_str} | {pinn_status} |")
    lines.append("")
    if matches:
        lines.append(f"**Structural match to true diffusion equation:** {matches[0].id} ({matches[0].name})\n")
    else:
        lines.append("**No candidate structurally matched the true diffusion equation.**\n")

    report_md_path = "results/hypotheses/phase4_hypothesis_report.md"
    with open(report_md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved human-readable report to {report_md_path}")


if __name__ == "__main__":
    main()
