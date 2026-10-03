"""
Conservation-law validation of the saved Phase 6 PINNs (no retraining of candidates).

For every candidate of the mock-LLM run (results/phase6) and the Gemini run
(results/evaluation/real_llm_gemini/phase6), measure how strongly the trained
network violates the global balance law dM/dt = integral f dx implied by its own
equation (validation/conservation.py), in three windows:

    train  t in [0, 0.8]    (fitted region)
    ood    t in (0.8, 1.0]  (held-out observations' window)
    beyond t in (1.0, 1.5]  (no data at all)

The conventional baseline (same network, data loss only) is retrained exactly as
in Phase 6 and judged against the SELECTED law with its fitted parameters.

Run:
    python experiments/12_conservation.py --config config.yaml
Output: results/evaluation/conservation/
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypotheses.adapter import hypothesis_to_pinn_config
from hypotheses.search import load_candidate_model, make_splits, save_json, train_data_only_mlp
from llm.schemas import Hypothesis
from pinn.data_utils import load_observational_data
from validation.conservation import conservation_violation

OUT = "results/evaluation/conservation"
RUNS = {"mock": "results/phase6", "gemini": "results/evaluation/real_llm_gemini/phase6"}
WINDOWS = {"train": (0.0, 0.8), "ood": (0.8, 1.0), "beyond": (1.0, 1.5)}


def _windows(model, residual_fn, params, d):
    return {w: conservation_violation(model, residual_fn, params, d["x_min"], d["x_max"], lo, hi)
            for w, (lo, hi) in WINDOWS.items()}


def evaluate_run(run_dir, cfg):
    pool = {p["id"]: Hypothesis(**p["hypothesis"])
            for p in json.load(open(os.path.join(run_dir, "generated_candidates.json")))["pool"]}
    pinn = json.load(open(os.path.join(run_dir, "pinn_validation.json")))
    selected = json.load(open(os.path.join(run_dir, "model_selection.json")))["selected"]
    out = {"selected": selected, "candidates": {}}
    for cid, res in pinn.items():
        model = load_candidate_model(os.path.join(run_dir, "models"), cid, cfg["pinn"])
        fn = hypothesis_to_pinn_config(pool[cid])["hypothesis_cfg"]["fn"]
        params = {k: torch.tensor(v) for k, v in res["params"].items()}
        out["candidates"][cid] = {"equation": res["equation"], "gate_passed": res["gate_passed"],
                                  **_windows(model, fn, params, cfg["diffusion"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    p6 = cfg["phase6"]
    results = {"definition": "relative_violation = RMS_t(integral residual dx) / RMS_t(dM/dt); "
                             "0 means the model's total amount changes exactly as its law says",
               "windows": WINDOWS}
    for name, run_dir in RUNS.items():
        if os.path.exists(os.path.join(run_dir, "pinn_validation.json")):
            results[name] = evaluate_run(run_dir, cfg)

    # conventional NN baseline, judged against the mock run's selected law
    sel = results["mock"]["selected"]
    mock_pool = {p["id"]: Hypothesis(**p["hypothesis"])
                 for p in json.load(open(os.path.join(RUNS["mock"], "generated_candidates.json")))["pool"]}
    sel_params = json.load(open(os.path.join(RUNS["mock"], "pinn_validation.json")))[sel]["params"]
    splits = make_splits(load_observational_data(cfg["paths"]["output_dir"], "noisy"),
                         p6["t_cutoff"], p6["val_fraction"], p6["split_seed"])
    mlp = train_data_only_mlp(splits, cfg["pinn"], epochs=cfg["pinn"]["training"]["adam_epochs"])
    fn = hypothesis_to_pinn_config(mock_pool[sel])["hypothesis_cfg"]["fn"]
    results["mlp_baseline_vs_selected_law"] = {
        "law": mock_pool[sel].equation, "params": sel_params,
        **_windows(mlp, fn, {k: torch.tensor(v) for k, v in sel_params.items()}, cfg["diffusion"])}

    save_json(results, os.path.join(OUT, "conservation.json"))
    lines = ["| Run | Candidate | Equation | Gate | train | OOD | beyond |", "|---|---|---|---|---|---|---|"]
    for name in RUNS:
        for cid, r in results.get(name, {}).get("candidates", {}).items():
            lines.append(f"| {name} | {cid} | `{r['equation']}` | {'pass' if r['gate_passed'] else 'fail'} | "
                         + " | ".join(f"{r[w]['relative_violation']:.2e}" for w in WINDOWS) + " |")
    b = results["mlp_baseline_vs_selected_law"]
    lines.append(f"| baseline | MLP (no physics) | `{b['law']}` | - | "
                 + " | ".join(f"{b[w]['relative_violation']:.2e}" for w in WINDOWS) + " |")
    with open(os.path.join(OUT, "conservation_table.md"), "w") as f:
        f.write("Relative global balance violation (0 = perfectly conserving w.r.t. its own law)\n\n"
                + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
