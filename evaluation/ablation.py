"""
Phase 7: minimal ablation on the 12 benchmark datasets (4 mechanism families).

Arms
  1. discovery_only        SINDy full-library STLSQ (Phase 2) -> match to a documented mechanism
  2. rag_generation        pool = mock-LLM proposals + equations appearing in retrieved evidence;
                           NO physics validation: Phase 4 ranking top-1 is the answer
  3. rag_kg_generation     pool = mock-LLM proposals + KG templates for retrieved mechanisms; NO validation
  4. full_system           pool as (3) + fit, physical bounds, gate, selection (revised Phase 7 rule)
  5. full_system_phase6_rule  as (4) but with the original Phase 6 selection rule (prefer_reduced=False)
  6. full_without_llm      as (4) with the mock LLM removed (RAG + KG + validation only)
  7. rag_validation_no_kg_no_llm  equations from retrieved evidence only + validation (isolates the KG)

Arms 2-6 use the regression fitting engine (evaluation/engine.py); fitted parameters are
recorded for arms 2/3 too, so parameter and OOD error are comparable across arms.
"""

import time
from typing import Dict, List

import numpy as np

from equation_discovery.discovery_engine import discover_from_grid, equation_to_string
from equation_discovery.library import build_library, parse_term_specs
from evaluation.engine import build_pool, family_of, score_run, truth_field, validate_and_select
from evaluation.identify import discovered_coefficients, grid_fields, match_mechanism, recover_parameters
from llm.schemas import Hypothesis
from validation.forward_solver import solve_forward

ARMS = ["discovery_only", "rag_generation", "rag_kg_generation", "full_system",
        "full_system_phase6_rule", "full_without_llm", "rag_validation_no_kg_no_llm"]


def _discovery_only(ds, res, cfg) -> Dict:
    p7 = cfg["phase7"]
    specs = parse_term_specs(cfg["discovery"]["candidate_terms"])
    sres, flat, y = discover_from_grid(ds.U, ds.x, ds.t, specs, crop=3,
                                       sindy_kwargs=dict(threshold=cfg["discovery"]["sindy"]["threshold"],
                                                         alpha=0.0, max_iterations=20))
    templates = [Hypothesis(id=d.metadata.document_id, name=d.declared_mechanism, equation=d.declared_equations[0],
                            parameters=[{"name": q} for q in d.declared_parameters]) for d in res.docs]
    coeffs = discovered_coefficients(sres)
    tmpl = match_mechanism(coeffs, templates)
    fam = tmpl.name if tmpl else ("none" if not coeffs else "other")
    params = recover_parameters(tmpl, coeffs) if tmpl else {}
    rng = np.random.default_rng(p7["seed"])
    val = rng.random(len(y)) < 0.2
    Theta, names = build_library(flat, specs)
    xi = np.array([sres["all_coefficients"][n] for n in names])
    rel_val = float(np.sqrt(np.mean((Theta[val] @ xi - y[val]) ** 2)) / np.sqrt(np.mean(y[val] ** 2)))
    disc = Hypothesis(id="S", name="sindy", equation=equation_to_string(sres) if coeffs else "u_t = 0*u", parameters=[])
    ood = None
    try:
        sol = solve_forward(disc, {}, ds.ic, ds.x, p7["ood_t_end"], n_out=int(round(p7["ood_t_end"] * 100)) + 1, method="rk4")
        m = sol["t"] > 1.0 + 1e-9
        if sol["stable"] and m.any():
            tr = truth_field(ds, sol["t"][m])
            ood = float(np.sqrt(np.mean((sol["U"][m] - tr) ** 2)) / np.sqrt(np.mean(tr ** 2)))
    except Exception:
        pass
    correct = fam == ds.family
    perr = (max(abs(params[k] - v) / abs(v) for k, v in ds.truth.items() if k in params)
            if correct and params else None)
    return {"dataset": ds.name, "truth_family": ds.family, "selected_family": fam, "correct_model": correct,
            "pool_contains_truth": None, "n_structurally_valid": 1 if tmpl else 0,
            "selected_params": params, "max_rel_param_error": perr, "selected_rel_val_error": rel_val,
            "selected_complexity": None, "ood_rel_rmse_t_1_to_end": ood,
            "n_wrong_candidates": 0, "n_wrong_rejected": 0, "discovered_equation": disc.equation}


def run_ablation(datasets, res, cfg) -> Dict[str, List[Dict]]:
    p7 = cfg["phase7"]
    settings = {
        "rag_generation": dict(pool=dict(use_kg=False, use_llm=True), validate=False, reduced=True),
        "rag_kg_generation": dict(pool=dict(use_kg=True, use_llm=True), validate=False, reduced=True),
        "full_system": dict(pool=dict(use_kg=True, use_llm=True), validate=True, reduced=True),
        "full_system_phase6_rule": dict(pool=dict(use_kg=True, use_llm=True), validate=True, reduced=False),
        "full_without_llm": dict(pool=dict(use_kg=True, use_llm=False), validate=True, reduced=True),
        # isolates the KG: evidence objects already carry their document's declared equations
        "rag_validation_no_kg_no_llm": dict(pool=dict(use_kg=False, use_llm=False), validate=True, reduced=True),
    }
    out = {a: [] for a in ARMS}
    for ds in datasets:
        t0 = time.perf_counter()
        r = _discovery_only(ds, res, cfg)
        r["runtime_s"] = time.perf_counter() - t0
        out["discovery_only"].append(r)
        for arm, st in settings.items():
            t0 = time.perf_counter()
            pool, _, _, _ = build_pool(ds.x, ds.t, ds.U, res, **st["pool"])
            run = validate_and_select(pool, ds.x, ds.t, ds.U, cfg, seed=p7["seed"],
                                      apply_validation=st["validate"], prefer_reduced=st["reduced"])
            elapsed = time.perf_counter() - t0
            s = score_run(ds, run, p7["ood_t_end"])
            if not st["validate"]:
                s["n_wrong_rejected"] = 0  # nothing is rejected without validation, by definition
            s["runtime_s"] = elapsed
            out[arm].append(s)
    return out


def _median(vals):
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else None


def summarize_ablation(results: Dict[str, List[Dict]], cfg) -> List[Dict]:
    p7 = cfg["phase7"]
    rows = []
    for arm in ARMS:
        rs = results[arm]
        n = len(rs)
        wrong = sum(r["n_wrong_candidates"] for r in rs)
        rows.append({
            "arm": arm,
            "valid_candidates_mean": float(np.mean([r["n_structurally_valid"] for r in rs])),
            "pool_recall": (None if rs[0]["pool_contains_truth"] is None
                            else sum(r["pool_contains_truth"] for r in rs) / n),
            "correct_model_rate": sum(r["correct_model"] for r in rs) / n,
            "correct_by_family": {f: f"{sum(r['correct_model'] for r in rs if r['truth_family'] == f)}/"
                                     f"{sum(1 for r in rs if r['truth_family'] == f)}"
                                  for f in sorted({r['truth_family'] for r in rs})},
            "incorrect_model_rejection_rate": (sum(r["n_wrong_rejected"] for r in rs) / wrong) if wrong else None,
            "median_param_error_when_correct": _median([r["max_rel_param_error"] for r in rs]),
            "params_within_tol": sum(1 for r in rs if r["max_rel_param_error"] is not None
                                     and r["max_rel_param_error"] <= p7["param_tol"]) / n,
            "median_heldout_rel_error": _median([r["selected_rel_val_error"] for r in rs]),
            "median_ood_rel_rmse": _median([r["ood_rel_rmse_t_1_to_end"] for r in rs]),
            "ood_within_tol": sum(1 for r in rs if r["ood_rel_rmse_t_1_to_end"] is not None
                                  and r["ood_rel_rmse_t_1_to_end"] <= p7["ood_tol"]) / n,
            "median_selected_complexity": _median([r["selected_complexity"] for r in rs]),
            "mean_runtime_s": float(np.mean([r["runtime_s"] for r in rs])),
        })
    return rows
