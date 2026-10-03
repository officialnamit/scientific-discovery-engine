"""
Phase 2, experiment script.

Runs equation discovery independently on:
  A. Clean   (full.npz  -- already a regular grid, no reconstruction needed)
  B. Sparse  (sparse.npz -- noise-free scattered subsample)
  C. Noisy   (noisy.npz  -- noisy scattered subsample), TWICE: once with no
     denoising smoothing (to honestly show how much it hurts) and once
     with the configured smoothing (the "improved" result).

The discovery pipeline is only ever given x, t, u. D_true and the ground
truth equation are loaded separately and used ONLY in the evaluation step.

Run:
    python experiments/02_equation_discovery.py --config config.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equation_discovery.discovery_engine import (
    discover_from_grid,
    discover_from_scatter,
    equation_to_string,
    evaluate_against_ground_truth,
    evaluate_competing_hypotheses,
)
from equation_discovery.library import parse_term_specs


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_case(name, fn, *args, **kwargs):
    print(f"\n--- Running discovery: {name} ---")
    result, flat_fields, y = fn(*args, **kwargs)
    print(f"  backend: {result['backend']}")
    print(f"  discovered: {equation_to_string(result)}")
    return result, flat_fields, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["paths"]["output_dir"]
    dcfg = cfg["discovery"]
    term_specs = parse_term_specs(dcfg["candidate_terms"])
    crop_clean = dcfg["finite_difference"]["crop_clean"]
    sindy_kwargs = dict(
        threshold=dcfg["sindy"]["threshold"],
        alpha=dcfg["sindy"]["alpha"],
        max_iterations=dcfg["sindy"]["max_iterations"],
    )
    D_true = float(cfg["diffusion"]["D_true"])

    full = np.load(os.path.join(out_dir, "full.npz"))
    sparse = np.load(os.path.join(out_dir, "sparse.npz"))
    noisy = np.load(os.path.join(out_dir, "noisy.npz"))

    x_grid, t_grid, U_full = full["x_grid"], full["t_grid"], full["U"]

    rcfg = dcfg["reconstruction"]
    interp_method = rcfg["interp_method"]

    def recon_grid(sub_cfg):
        x_r = np.linspace(x_grid.min(), x_grid.max(), sub_cfg["n_x_recon"])
        t_r = np.linspace(t_grid.min(), t_grid.max(), sub_cfg["n_t_recon"])
        return x_r, t_r

    x_recon_sparse, t_recon_sparse = recon_grid(rcfg["sparse"])
    x_recon_noisy, t_recon_noisy = recon_grid(rcfg["noisy"])

    results = {}
    fields_cache = {}

    # A. Clean (full grid, exact FD, no reconstruction)
    res, flds, y = run_case(
        "A. Clean (full grid)",
        discover_from_grid, U_full, x_grid, t_grid, term_specs, crop_clean, sindy_kwargs,
    )
    results["clean"] = res
    fields_cache["clean"] = (flds, y)

    # B. Sparse (noise-free scatter, reconstructed, no smoothing needed)
    scfg = rcfg["sparse"]
    res, flds, y = run_case(
        "B. Sparse (noise-free scatter)",
        discover_from_scatter,
        sparse["x"], sparse["t"], sparse["u"], x_recon_sparse, t_recon_sparse, term_specs,
        scfg["crop"], scfg["smoothing_sigma"], interp_method, sindy_kwargs,
    )
    results["sparse"] = res
    fields_cache["sparse"] = (flds, y)

    # C1. Noisy, NO smoothing (honest "before" result -- Step 8)
    ncfg = rcfg["noisy"]
    res, flds, y = run_case(
        "C1. Noisy scatter, NO smoothing",
        discover_from_scatter,
        noisy["x"], noisy["t"], noisy["u"], x_recon_noisy, t_recon_noisy, term_specs,
        ncfg["crop"], 0.0, interp_method, sindy_kwargs,
    )
    results["noisy_raw"] = res
    fields_cache["noisy_raw"] = (flds, y)

    # C2. Noisy, WITH smoothing (the "improved" result -- Step 8)
    res, flds, y = run_case(
        "C2. Noisy scatter, WITH Gaussian smoothing",
        discover_from_scatter,
        noisy["x"], noisy["t"], noisy["u"], x_recon_noisy, t_recon_noisy, term_specs,
        ncfg["crop"], ncfg["smoothing_sigma"], interp_method, sindy_kwargs,
    )
    results["noisy_smoothed"] = res
    fields_cache["noisy_smoothed"] = (flds, y)

    # --- Evaluation against hidden ground truth (grading only) ---
    evaluations = {
        key: evaluate_against_ground_truth(res, true_term="u_xx", coefficient_true=D_true)
        for key, res in results.items()
    }

    # --- Step 9: competing hypotheses (H1 diffusion vs H2 advection) ---
    hyp_cfg = dcfg["hypotheses"]
    hypothesis_scores = {
        key: evaluate_competing_hypotheses(flds, y, hyp_cfg)
        for key, (flds, y) in fields_cache.items()
    }

    # --- Save everything ---
    results_dir = "results/equations"
    os.makedirs(results_dir, exist_ok=True)

    def jsonable(d):
        return json.loads(json.dumps(d, default=lambda o: float(o)))

    full_report = {
        "D_true": D_true,
        "results": {k: jsonable(v) for k, v in results.items()},
        "evaluations": {k: jsonable(v) for k, v in evaluations.items()},
        "hypothesis_scores": {k: jsonable(v) for k, v in hypothesis_scores.items()},
        "equations_human_readable": {k: equation_to_string(v) for k, v in results.items()},
    }
    report_path = os.path.join(results_dir, "phase2_discovery_report.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved full report to {report_path}")

    # --- Markdown table ---
    labels = {
        "clean": "Clean",
        "sparse": "Sparse",
        "noisy_raw": "Noisy (no smoothing)",
        "noisy_smoothed": "Noisy (smoothed)",
    }
    lines = [
        "| Dataset | Discovered Equation | D estimate | Rel. param error | Structure recovered |",
        "|---|---|---|---|---|",
    ]
    for key, label in labels.items():
        eq = equation_to_string(results[key])
        ev = evaluations[key]
        lines.append(
            f"| {label} | `{eq}` | {ev['coefficient_pred']:.4f} | "
            f"{ev['relative_parameter_error']*100:.2f}% | {ev['structure_recovered']} |"
        )
    table_md = "\n".join(lines)
    print("\n" + table_md)
    table_path = os.path.join(results_dir, "phase2_results_table.md")
    with open(table_path, "w") as f:
        f.write(table_md + "\n")
    print(f"\nSaved table to {table_path}")

    # --- Hypothesis comparison printout ---
    print("\n--- Step 9: Competing hypothesis scores (unrestricted least squares) ---")
    for dataset_key, scores in hypothesis_scores.items():
        print(f"\n[{labels.get(dataset_key, dataset_key)}]")
        for hyp_name, s in scores.items():
            coeff_str = ", ".join(f"{k}={v:.4f}" for k, v in s["coefficients"].items())
            print(f"  {hyp_name} ({s['terms']}): R^2={s['r2']:.4f}, coeff: {coeff_str}")

    # --- Plot: discovered coefficients per dataset ---
    all_term_names = list(results["clean"]["all_coefficients"].keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.2
    xpos = np.arange(len(all_term_names))
    for i, (key, label) in enumerate(labels.items()):
        coeffs = [results[key]["all_coefficients"].get(t, 0.0) for t in all_term_names]
        ax.bar(xpos + i * width, coeffs, width=width, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(D_true, color="red", linestyle="--", linewidth=1, label=f"D_true={D_true}")
    ax.set_xticks(xpos + width * 1.5)
    ax.set_xticklabels(all_term_names)
    ax.set_ylabel("discovered coefficient")
    ax.set_title("Phase 2: discovered coefficients per term, by dataset")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = os.path.join(results_dir, "phase2_coefficients.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nSaved coefficient comparison plot to {plot_path}")

    print("\nPhase 2 experiment complete.")


if __name__ == "__main__":
    main()
