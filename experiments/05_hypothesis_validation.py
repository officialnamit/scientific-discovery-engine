"""
Phase 3, Parts 9-12: hypothesis validation.

Trains a PINN under EACH candidate hypothesis (H1: u_t=D*u_xx, H2:
u_t=c*u_x), both with the physical parameter TRAINABLE (neither is
told the "right" answer -- H1's D is initialized away from D_true, and
H2's c has no true value at all since advection is the wrong physics).

For the OOD test: BOTH PINNs are trained using ONLY data/collocation/IC
with t <= t_cutoff (config.pinn.ood.t_cutoff = 0.7). The BC is still
imposed at all t (a boundary condition is knowledge about the domain
edge, not a future-time observation, so restricting it does not leak
future information; only the interior data/physics/IC training signal
is time-restricted). Both are then evaluated against the clean
reference field on t <= t_cutoff (in-distribution) and t > t_cutoff
(OOD, never seen in training).

Run:
    python experiments/05_hypothesis_validation.py --config config.yaml
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
from pinn.residuals import RESIDUAL_REGISTRY
from pinn.trainer import build_and_train
from validation.ood import prediction_error, split_by_time
from validation.physics_validation import evaluate_physics_residual
from validation.scorer import build_validation_result


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_hypothesis(name, hyp_cfg, param_init, pcfg, data, collocation, bc, ic, seed):
    print(f"\n=== Training hypothesis: {name} ({hyp_cfg['equation_str']}) ===")
    trainer, history = build_and_train(
        hyp_cfg, param_init, trainable=True,
        architecture_cfg=pcfg["architecture"], weights=pcfg["weights"],
        data=data, collocation=collocation, bc=bc, ic=ic,
        training_cfg=pcfg["training"], seed=seed,
    )
    final = history[-1]
    print(f"  final losses: {final}")
    return trainer, history, final


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
    t_cutoff = float(pcfg["ood"]["t_cutoff"])
    seed = pcfg["training"]["seed"]

    # --- Training data restricted to t <= t_cutoff for the OOD setup ---
    noisy_full = load_observational_data(out_dir, "noisy")
    train_mask = noisy_full["t"] <= t_cutoff
    noisy_train = {k: v[train_mask] for k, v in noisy_full.items()}
    print(f"Training observations: {len(noisy_train['u'])} of {len(noisy_full['u'])} "
          f"(restricted to t <= {t_cutoff} for the OOD experiment)")

    data = data_dict_to_tensors(noisy_train)
    collocation = make_collocation_points(x_min, x_max, t_min, t_cutoff, pcfg["training"]["n_collocation"], seed=seed)
    bc = make_bc_points(x_min, x_max, t_min, t_max, pcfg["training"]["n_bc"], seed=seed + 1)  # BC known for all t
    ic = make_ic_points(x_min, x_max, pcfg["training"]["n_ic"], ic_kind, seed=seed + 2)

    clean = load_clean_reference(out_dir)
    in_dist, ood = split_by_time(clean["x"], clean["t"], clean["u"], t_cutoff)
    print(f"Evaluation split: {len(in_dist['u'])} in-distribution (t<={t_cutoff}) / "
          f"{len(ood['u'])} OOD (t>{t_cutoff}) points from the clean reference")

    hyp_specs = pcfg["hypotheses"]
    thresholds = pcfg["validation"]["thresholds"]

    validation_results = {}
    trainers = {}
    histories = {}

    for hyp_key, spec in hyp_specs.items():
        residual_name = spec["residual"]
        hyp_cfg = RESIDUAL_REGISTRY[residual_name]
        param_init = spec["param_init"]

        trainer, history, final = run_hypothesis(
            hyp_key, hyp_cfg, param_init, pcfg, data, collocation, bc, ic, seed
        )
        trainers[hyp_key] = trainer
        histories[hyp_key] = history

        physics_residual_fresh = evaluate_physics_residual(
            trainer.model, hyp_cfg["fn"], trainer.params, x_min, x_max, t_min, t_max
        )
        in_dist_err = prediction_error(trainer.model, in_dist["x"], in_dist["t"], in_dist["u"])
        ood_err = prediction_error(trainer.model, ood["x"], ood["t"], ood["u"])

        param_estimate = {k: float(v.item()) for k, v in trainer.params.items()}
        param_true = {"D": D_true} if residual_name == "diffusion" else None

        result = build_validation_result(
            hypothesis_name=hyp_key,
            equation_str=hyp_cfg["equation_str"],
            data_loss=final["data"],
            physics_residual=physics_residual_fresh,
            bc_loss=final["bc"],
            ic_loss=final["ic"],
            prediction_error=in_dist_err["rmse"],
            thresholds=thresholds,
            parameter_estimate=param_estimate,
            parameter_true=param_true,
            ood_error=ood_err["rmse"],
        )
        result["in_distribution_rmse"] = in_dist_err["rmse"]
        result["ood_rmse"] = ood_err["rmse"]
        validation_results[hyp_key] = result

        print(f"\n[{hyp_key}] status: {result['status']}")
        print(f"  data_loss={result['data_loss']:.6f}  physics_residual={result['physics_residual']:.6f}  "
              f"bc_loss={result['bc_loss']:.6f}  ic_loss={result['ic_loss']:.6f}")
        print(f"  in-distribution RMSE={in_dist_err['rmse']:.6f}  OOD RMSE={ood_err['rmse']:.6f}")
        if param_true:
            print(f"  parameter_estimate={param_estimate}  parameter_error={result['parameter_error']}")
        else:
            print(f"  parameter_estimate={param_estimate} (no true value -- wrong hypothesis)")

    os.makedirs("results/pinn", exist_ok=True)
    report_path = "results/pinn/phase3_hypothesis_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    print(f"\nSaved validation report to {report_path}")

    # --- Comparison table ---
    lines = [
        "| Hypothesis | Equation | Status | Data loss | Physics residual | BC loss | IC loss | In-dist RMSE | OOD RMSE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for hyp_key, result in validation_results.items():
        lines.append(
            f"| {hyp_key} | `{result['equation']}` | {result['status']} | "
            f"{result['data_loss']:.5f} | {result['physics_residual']:.5f} | "
            f"{result['bc_loss']:.5f} | {result['ic_loss']:.5f} | "
            f"{result['in_distribution_rmse']:.5f} | {result['ood_rmse']:.5f} |"
        )
    table_md = "\n".join(lines)
    print("\n" + table_md)
    with open("results/pinn/phase3_validation_table.md", "w") as f:
        f.write(table_md + "\n")

    # --- Plots: OOD comparison ---
    full = np.load(os.path.join(out_dir, "full.npz"))
    x_grid, t_grid, U_clean = full["x_grid"], full["t_grid"], full["U"]
    Xg, Tg = np.meshgrid(x_grid, t_grid)
    xt = torch.tensor(Xg.ravel(), dtype=torch.float32).reshape(-1, 1)
    tt = torch.tensor(Tg.ravel(), dtype=torch.float32).reshape(-1, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(U_clean, aspect="auto", origin="lower",
                          extent=[x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()], cmap="viridis")
    axes[0].axhline(t_cutoff, color="white", linestyle="--", linewidth=1.5)
    axes[0].set_title("Clean reference (dashed = train/OOD split)")
    fig.colorbar(im0, ax=axes[0])

    for ax, hyp_key in zip(axes[1:], validation_results.keys()):
        with torch.no_grad():
            U_pred = trainers[hyp_key].model(xt, tt).numpy().reshape(U_clean.shape)
        im = ax.imshow(U_pred, aspect="auto", origin="lower",
                        extent=[x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()], cmap="viridis")
        ax.axhline(t_cutoff, color="white", linestyle="--", linewidth=1.5)
        ax.set_title(f"{hyp_key}\nOOD RMSE={validation_results[hyp_key]['ood_rmse']:.4f}")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    plot_path = "results/pinn/phase3_ood_comparison.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved OOD comparison plot to {plot_path}")

    # --- Plot: training curves side by side ---
    fig2, axes2 = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4))
    if len(histories) == 1:
        axes2 = [axes2]
    for ax, (hyp_key, history) in zip(axes2, histories.items()):
        epochs = [h["epoch"] for h in history if isinstance(h["epoch"], int)]
        for key in ["total", "data", "physics", "bc", "ic"]:
            vals = [h[key] for h in history if isinstance(h["epoch"], int)]
            ax.plot(epochs, vals, label=key)
        ax.set_yscale("log")
        ax.set_title(f"{hyp_key} training curve")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig2.tight_layout()
    plot_path2 = "results/pinn/phase3_hypothesis_training_curves.png"
    fig2.savefig(plot_path2, dpi=150)
    print(f"Saved training-curve plot to {plot_path2}")


if __name__ == "__main__":
    main()
