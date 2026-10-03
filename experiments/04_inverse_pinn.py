"""
Phase 3, Part 7: inverse PINN.

D is UNKNOWN: trainable, initialized to D_init=0.2 (config), while
D_true=0.1 is retained ONLY for evaluation afterward. Network weights
and D are optimized jointly.

Run:
    python experiments/04_inverse_pinn.py --config config.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinn.data_utils import (
    data_dict_to_tensors,
    load_clean_reference,
    load_observational_data,
    make_bc_points,
    make_collocation_points,
    make_ic_points,
)
from pinn.inverse import run_inverse_pinn
from pinn.residuals import RESIDUAL_REGISTRY
from validation.ood import prediction_error
from validation.physics_validation import evaluate_physics_residual


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    out_dir = cfg["paths"]["output_dir"]
    dcfg = cfg["diffusion"]
    pcfg = cfg["pinn"]
    x_min, x_max = dcfg["x_min"], dcfg["x_max"]
    t_min, t_max = dcfg["t_min"], dcfg["t_max"]
    ic_kind = dcfg["initial_condition"]
    D_true = float(dcfg["D_true"])
    D_init = float(pcfg["inverse"]["D_init"])

    seed = pcfg["training"]["seed"]
    noisy = load_observational_data(out_dir, "noisy")
    data = data_dict_to_tensors(noisy)
    collocation = make_collocation_points(x_min, x_max, t_min, t_max, pcfg["training"]["n_collocation"], seed=seed)
    bc = make_bc_points(x_min, x_max, t_min, t_max, pcfg["training"]["n_bc"], seed=seed + 1)
    ic = make_ic_points(x_min, x_max, pcfg["training"]["n_ic"], ic_kind, seed=seed + 2)

    print(f"Inverse PINN: hypothesis=diffusion, D_init={D_init} (D_true={D_true} NOT given to the model)")

    trainer, history = run_inverse_pinn(
        hypothesis_name="diffusion",
        param_init={"D": D_init},
        architecture_cfg=pcfg["architecture"],
        weights=pcfg["weights"],
        data=data, collocation=collocation, bc=bc, ic=ic,
        training_cfg=pcfg["training"],
        seed=seed,
    )

    final = history[-1]
    D_hat = final["D"]
    abs_err = abs(D_hat - D_true)
    rel_err = abs_err / abs(D_true)
    print(f"\nFinal training losses: {final}")
    print(f"D_hat = {D_hat:.6f}, D_true = {D_true}, abs_error = {abs_err:.6f}, rel_error = {rel_err*100:.3f}%")

    residual_fn = RESIDUAL_REGISTRY["diffusion"]["fn"]
    physics_residual_fresh = evaluate_physics_residual(
        trainer.model, residual_fn, trainer.params, x_min, x_max, t_min, t_max
    )
    clean = load_clean_reference(out_dir)
    pred_err = prediction_error(trainer.model, clean["x"], clean["t"], clean["u"])

    results = {
        "hypothesis": "diffusion",
        "D_init": D_init,
        "D_true": D_true,
        "D_hat": D_hat,
        "absolute_parameter_error": abs_err,
        "relative_parameter_error": rel_err,
        "final_training_losses": final,
        "physics_residual_fresh_grid": physics_residual_fresh,
        "prediction_error_vs_clean": pred_err,
    }

    os.makedirs("results/pinn", exist_ok=True)
    with open("results/pinn/phase3_inverse_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved report to results/pinn/phase3_inverse_report.json")
    print(json.dumps(results, indent=2))

    # --- Plots: D convergence + training curve ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    epochs = [h["epoch"] for h in history if isinstance(h["epoch"], int)]
    D_vals = [h["D"] for h in history if isinstance(h["epoch"], int)]
    axes[0].plot(epochs, D_vals, marker="o", markersize=3, label="D_hat")
    axes[0].axhline(D_true, color="red", linestyle="--", label=f"D_true={D_true}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("D estimate")
    axes[0].set_title("Inverse PINN: D convergence")
    axes[0].legend()

    for key, label in [("total", "total"), ("data", "data"), ("physics", "physics"),
                        ("bc", "bc"), ("ic", "ic")]:
        vals = [h[key] for h in history if isinstance(h["epoch"], int)]
        axes[1].plot(epochs, vals, label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss (log scale)")
    axes[1].set_title("Inverse PINN training curve")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    plot_path = "results/pinn/phase3_inverse_plot.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
