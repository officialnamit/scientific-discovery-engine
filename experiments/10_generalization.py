"""
Phase 7 generalization: the SAME Phase 6 pipeline (inverse PINN per candidate,
physical bounds, gate on held-out noisy observations, selection, OOD, forward
prediction) on a genuinely different system:

    hidden truth: u_t + c*u_x = D*u_xx,  c = 0.4, D = 0.02,  x in [0, 1.5], Gaussian IC
    observations: 400 random points, 5% Gaussian noise (same regime as Phase 1)

No new architecture: hypotheses/search.py, pinn/, validation/ are used unchanged.
Truth (c, D, exact solution) is read only in oracle_evaluation(), after selection.

Stages (each PINN ~4 min on CPU, cached per candidate):
    --stage prepare | pinn [--only H1,H4] | select | all (default)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equation_discovery.derivatives import reconstruct_grid, smooth_grid
from equation_discovery.discovery_engine import discover_from_grid, equation_to_string
from equation_discovery.library import parse_term_specs
from evaluation.datasets import benchmark_datasets
from evaluation.engine import PROBLEM_SPEC, Resources, build_pool, family_of
from hypotheses.search import (_rmse, load_candidate_model, make_splits, save_json,
                               structural_and_physics_filter, train_data_only_mlp, validate_candidate)
from llm.schemas import Hypothesis
from validation.forward_solver import solve_forward
from validation.model_selection import select_model

OUT = "results/evaluation/generalization"
CAND = os.path.join(OUT, "candidates")
MODELS = os.path.join(OUT, "models")
DATASET = "advdiff_c0.4_D0.02"
N_OBS, NOISE, SEED = 400, 0.05, 0


def _truth_dataset():
    return next(d for d in benchmark_datasets() if d.name == DATASET)


def observations():
    """400 noisy scattered samples -- the ONLY data the pipeline sees (x, t, u)."""
    ds = _truth_dataset()
    rng = np.random.default_rng(SEED)
    X, T = np.meshgrid(ds.x, ds.t)
    idx = rng.choice(X.size, size=N_OBS, replace=False)
    u = ds.U.ravel()[idx]
    return {"x": X.ravel()[idx], "t": T.ravel()[idx], "u": u + rng.normal(0, NOISE * np.std(ds.U), N_OBS)}


def _recon(obs):
    xg, tg = np.linspace(0, 1.5, 31), np.linspace(0, 1.0, 31)
    U, _, _ = reconstruct_grid(obs["x"], obs["t"], obs["u"], xg, tg)
    return xg, tg, smooth_grid(U, 0.8)


def _setup(ds):
    return {"x_min": 0.0, "x_max": 1.5, "t_min": 0.0, "ic_kind": ds.ic}


def stage_prepare(cfg):
    ds = _truth_dataset()
    obs = observations()
    np.savez(os.path.join("data/generated", "advdiff_noisy.npz"), **obs)
    xg, tg, Ug = _recon(obs)
    specs = parse_term_specs(cfg["discovery"]["candidate_terms"])
    sres, _, _ = discover_from_grid(Ug, xg, tg, specs, crop=2,
                                    sindy_kwargs=dict(threshold=cfg["discovery"]["sindy"]["threshold"], alpha=0.0, max_iterations=20))
    res = Resources(cfg)
    pool, origin, ctx, desc = build_pool(xg, tg, Ug, res)
    survivors, filt = structural_and_physics_filter(pool, PROBLEM_SPEC, cfg["phase6"]["parameter_constraints"])
    save_json({"dataset": DATASET, "n_obs": N_OBS, "noise_frac": NOISE,
               "description_given_to_system": desc,
               "sindy_on_reconstructed_noisy_grid": equation_to_string(sres),
               "retrieved_mechanisms": ctx.mechanisms,
               "pool": [{"id": h.id, "origin": origin[h.id], "equation": h.equation,
                         "hypothesis": json.loads(h.model_dump_json())} for h in pool],
               "filter_report": filt, "survivors": [h.id for h in survivors]},
              os.path.join(OUT, "prepare.json"))
    print(desc)
    print("SINDy (noisy, reconstructed):", equation_to_string(sres))
    print("Survivors:", [(h.id, h.equation) for h in survivors])


def _survivors():
    prep = json.load(open(os.path.join(OUT, "prepare.json")))
    by = {p["id"]: Hypothesis(**p["hypothesis"]) for p in prep["pool"]}
    return [by[i] for i in prep["survivors"]]


def _splits(cfg):
    return make_splits(observations(), cfg["phase6"]["t_cutoff"], cfg["phase6"]["val_fraction"], cfg["phase6"]["split_seed"])


def stage_pinn(cfg, only):
    p6, pcfg = cfg["phase6"], cfg["pinn"]
    ds = _truth_dataset()
    splits = _splits(cfg)
    for h in _survivors():
        if only and h.id not in only:
            continue
        print(f"=== inverse PINN {h.id}: {h.equation}")
        r, _ = validate_candidate(h, splits, _setup(ds), pcfg, pcfg["training"], p6["gate"],
                                  p6["parameter_constraints"], p6["inactive_threshold"], MODELS)
        save_json(r, os.path.join(CAND, f"{h.id}.json"))
        print(f"  {r['status']} params={ {k: round(v, 5) for k, v in r['params'].items()} } "
              f"val_rmse={r['val_rmse']:.4f} failed={r['failed_checks']} inactive={r['inactive_parameters']}")


def stage_select(cfg):
    p6, p7, pcfg = cfg["phase6"], cfg["phase7"], cfg["pinn"]
    hyps = {h.id: h for h in _survivors()}
    results = {}
    for cid in hyps:
        path = os.path.join(CAND, f"{cid}.json")
        if not os.path.exists(path):
            raise SystemExit(f"missing PINN result for {cid}: run --stage pinn --only {cid}")
        results[cid] = json.load(open(path))
    rows = list(results.values())
    sel = select_model(rows, p6["equivalence_tol"], p6["inactive_threshold"], prefer_reduced=True)
    sel_orig = select_model(rows, p6["equivalence_tol"], p6["inactive_threshold"], prefer_reduced=False)
    splits = _splits(cfg)
    ds = _truth_dataset()
    models = {cid: load_candidate_model(MODELS, cid, pcfg) for cid in results}
    mlp = train_data_only_mlp(splits, pcfg, epochs=pcfg["training"]["adam_epochs"])
    sid = sel["selected"]
    out = {"selection": sel, "selection_original_phase6_rule": sel_orig["selected"],
           "pinn_validation": results,
           "selected_family": family_of(hyps[sid]) if sid else None,
           "ood_rmse_noisy_obs": {**{c: results[c]["ood_obs_rmse"] for c in results}, "MLP_data_only": _rmse(mlp, splits["ood"])},
           "heldout_rmse_noisy_obs": {**{c: results[c]["val_rmse"] for c in results}, "MLP_data_only": _rmse(mlp, splits["val"])}}

    # ---------------- oracle evaluation (after selection; never feeds back) ----------------
    X, T = np.meshgrid(ds.x, ds.t)
    m = T > p6["t_cutoff"]
    def nn_err(model):
        with torch.no_grad():
            pr = model(torch.tensor(X[m], dtype=torch.float32).reshape(-1, 1),
                       torch.tensor(T[m], dtype=torch.float32).reshape(-1, 1)).numpy().ravel()
        return float(np.sqrt(np.mean((pr - ds.U[m]) ** 2)))
    oracle = {"truth": ds.truth, "ood_rmse_vs_clean_field": {**{c: nn_err(models[c]) for c in models}, "MLP_data_only": nn_err(mlp)}}
    if sid:
        params = results[sid]["params"]
        oracle["selected_param_rel_error"] = {k: abs(params[k] - v) / abs(v) for k, v in ds.truth.items() if k in params}
        sol = solve_forward(hyps[sid], params, ds.ic, ds.x, p7["ood_t_end"], n_out=int(p7["ood_t_end"] * 100) + 1, method="rk4")
        w = sol["t"] > 1.0 + 1e-9
        Xe, Te = np.meshgrid(ds.x, sol["t"][w])
        tr = ds.exact(Xe, Te)
        oracle["novel_prediction_rel_rmse_t_1_to_1.3"] = float(np.sqrt(np.mean((sol["U"][w] - tr) ** 2)) / np.sqrt(np.mean(tr ** 2)))
    out["oracle_benchmark_evaluation"] = oracle
    save_json(out, os.path.join(OUT, "generalization_results.json"))
    print(json.dumps({k: out[k] for k in ("selected_family", "selection_original_phase6_rule", "ood_rmse_noisy_obs")}, indent=2, default=float))
    print(json.dumps(oracle, indent=2, default=float))
    print("selection:", sel["selected"], sel["equivalence_set"], sel.get("reduction_applied"), sel["explanations"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", default="all", choices=["prepare", "pinn", "select", "all"])
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    os.makedirs(OUT, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    if a.stage in ("prepare", "all"):
        stage_prepare(cfg)
    if a.stage in ("pinn", "all"):
        stage_pinn(cfg, only)
    if a.stage in ("select", "all"):
        stage_select(cfg)


if __name__ == "__main__":
    main()
