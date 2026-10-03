"""
Phase 3, Part 6: forward PINN.

D = 0.1 is KNOWN and FIXED. The PINN is trained on the noisy, sparse
observations (data loss) plus the known physics/BC/IC constraints, and
evaluated against the CLEAN full-field reference (never used in
training) for prediction error.

Run:
    python experiments/03_forward_pinn.py --config config.yaml
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
from pinn.forward import run_forward_pinn
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
    D_fixed = float(pcfg["forward"]["D_fixed"])

    seed = pcfg["training"]["seed"]
    noisy = load_observational_data(out_dir, "noisy")
    data = data_dict_to_tensors(noisy)
    collocation = make_collocation_points(x_min, x_max, t_min, t_max, pcfg["training"]["n_collocation"], seed=seed)
    bc = make_bc_points(x_min, x_max, t_min, t_max, pcfg["training"]["n_bc"], seed=seed + 1)
    ic = make_ic_points(x_min, x_max, pcfg["training"]["n_ic"], ic_kind, seed=seed + 2)

    print(f"Forward PINN: hypothesis=diffusion, D fixed = {D_fixed} (true D = {D_true})")
    print(f"  training data: {len(noisy['u'])} noisy observations")
    print(f"  collocation: {pcfg['training']['n_collocation']}, bc: {2*pcfg['training']['n_bc']}, ic: {pcfg['training']['n_ic']}")

    trainer, history = run_forward_pinn(
        hypothesis_name="diffusion",
        param_values={"D": D_fixed},
        architecture_cfg=pcfg["architecture"],
        weights=pcfg["weights"],
        data=data, collocation=collocation, bc=bc, ic=ic,
        training_cfg=pcfg["training"],
        seed=seed,
    )

    final = history[-1]
    print(f"\nFinal training losses: {final}")

    # Physics residual on a FRESH grid (not the training collocation points)
    residual_fn = RESIDUAL_REGISTRY["diffusion"]["fn"]
    physics_residual_fresh = evaluate_physics_residual(
        trainer.model, residual_fn, trainer.params, x_min, x_max, t_min, t_max
    )

    # Prediction error against the clean reference field (never used in training)
    clean = load_clean_reference(out_dir)
    pred_err = prediction_error(trainer.model, clean["x"], clean["t"], clean["u"])

    results = {
        "hypothesis": "diffusion",
        "D_fixed": D_fixed,
        "D_true": D_true,
        "final_training_losses": final,
        "physics_residual_fresh_grid": physics_residual_fresh,
        "prediction_error_vs_clean": pred_err,
    }

    os.makedirs("results/pinn", exist_ok=True)
    with open("results/pinn/phase3_forward_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved report to results/pinn/phase3_forward_report.json")
    print(json.dumps(results, indent=2))

    # --- Plots: training curve + predicted vs clean field ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = [h["epoch"] for h in history if isinstance(h["epoch"], int)]
    for key, label in [("total", "total"), ("data", "data"), ("physics", "physics"),
                        ("bc", "bc"), ("ic", "ic")]:
        vals = [h[key] for h in history if isinstance(h["epoch"], int)]
        axes[0].plot(epochs, vals, label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].set_title("Forward PINN training curve")
    axes[0].legend(fontsize=8)

    full = np.load(os.path.join(out_dir, "full.npz"))
    x_grid, t_grid, U_clean = full["x_grid"], full["t_grid"], full["U"]
    Xg, Tg = np.meshgrid(x_grid, t_grid)
    with torch.no_grad():
        xt = torch.tensor(Xg.ravel(), dtype=torch.float32).reshape(-1, 1)
        tt = torch.tensor(Tg.ravel(), dtype=torch.float32).reshape(-1, 1)
        U_pred = trainer.model(xt, tt).numpy().reshape(U_clean.shape)

    im1 = axes[1].imshow(U_clean, aspect="auto", origin="lower",
                          extent=[x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()], cmap="viridis")
    axes[1].set_title("Clean reference u(x,t)")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(U_pred, aspect="auto", origin="lower",
                          extent=[x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()], cmap="viridis")
    axes[2].set_title(f"Forward PINN prediction (RMSE={pred_err['rmse']:.4f})")
    fig.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    plot_path = "results/pinn/phase3_forward_plot.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
