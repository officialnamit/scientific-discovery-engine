"""
Phase 6 experiment: physics-constrained generative search.

    Problem -> RAG -> KG -> ScientificContext -> hypothesis generation (+ KG templates)
      -> structural validation -> physics pre-checks -> inverse PINN (per candidate)
      -> post-fit physical bounds -> gate on held-out observations
      -> term necessity / complexity / BIC -> model selection
      -> empirical stability -> OOD prediction -> novel prediction (solve the selected law forward)

Stages (each PINN fit takes ~4 min on CPU, so they are cached per candidate):
    --stage prepare   RAG/KG/context/generation/filters -> generated_candidates.json, filtered_candidates.json
    --stage pinn      inverse PINN for surviving candidates (use --only K1,H1 to run a subset); cached
    --stage select    stability, model selection, baselines, OOD + novel prediction, report
    --stage all       (default) everything in order

Ground-truth policy: D_true and the clean field are loaded ONLY inside
oracle_evaluation() below, and only to REPORT benchmark error after all
decisions are made. Nothing they produce feeds back into selection.

Run:
    python experiments/08_physics_constrained_search.py --config config.yaml
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

from data.generate_diffusion import initial_condition, solve_diffusion_fd
from data.problem_description import build_problem_description
from hypotheses.pipeline import generate_hypotheses
from hypotheses.search import (build_candidate_pool, kg_template_candidates, load_candidate_model,
                               make_splits, save_json, structural_and_physics_filter,
                               train_data_only_mlp, validate_candidate)
from knowledge_graph.builder import build_graph_from_documents
from llm import get_provider
from llm.schemas import Hypothesis
from pinn.data_utils import load_observational_data
from rag.context import build_scientific_context
from rag.corpus_loader import load_corpus
from rag.pipeline import RAGPipeline
from validation.forward_solver import solve_forward
from validation.model_selection import select_model
from validation.stability import bootstrap_stability

OUT = "results/phase6"
CAND_DIR = os.path.join(OUT, "candidates")
MODEL_DIR = os.path.join(OUT, "models")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def stage_prepare(cfg):
    out_dir = cfg["paths"]["output_dir"]
    p6 = cfg["phase6"]

    docs = load_corpus()
    rag = RAGPipeline()
    chunks = rag.ingest(docs)
    graph = build_graph_from_documents(docs, {c.chunk_id: c for c in chunks})
    problem = build_problem_description(out_dir)
    context = build_scientific_context(problem, rag.get_retriever(), graph, top_k=5)

    leaked = [d.metadata.document_id for d in docs if str(cfg["diffusion"]["D_true"]) in d.text]
    provider = get_provider(cfg["llm"])
    gen = generate_hypotheses(problem, provider, n_hypotheses=cfg["llm"]["n_hypotheses"],
                              domain_hint="1D scalar-field transport/reaction",
                              temperature=cfg["llm"]["temperature"], context=context.model_dump())
    kg_cands = kg_template_candidates(context.model_dump(), graph, p6["parameter_defaults"])
    pool, removed, origin = build_candidate_pool(gen["valid_hypotheses"], kg_cands)

    problem_spec = {"has_ic": True, "dirichlet_both_ends": True}  # what the observational setup provides
    survivors, filter_report = structural_and_physics_filter(pool, problem_spec, p6["parameter_constraints"])

    save_json({
        "problem_description": problem,
        "provider": gen["provider"], "generation_mode": gen["mode"],
        "answer_leakage_check": {"D_true_string_found_in_corpus_docs": leaked},
        "scientific_evidence": {
            "retrieved": [{"evidence_id": e.evidence_id, "document_id": e.document_id, "section": e.section,
                           "retrieval_score": e.confidence} for e in context.retrieved_evidence],
            "mechanisms": context.mechanisms, "documented_equation_forms": context.known_equations,
            "parameters": context.parameters, "n_graph_relationships": len(context.graph_relationships),
            "graph_summary": graph.summary(),
        },
        "llm_candidates": [json.loads(h.model_dump_json()) for h in gen["valid_hypotheses"]],
        "kg_template_candidates": [json.loads(h.model_dump_json()) for h in kg_cands],
        "duplicates_removed": removed,
        "pool": [{"id": h.id, "origin": origin[h.id], "equation": h.equation,
                  "hypothesis": json.loads(h.model_dump_json())} for h in pool],
    }, os.path.join(OUT, "generated_candidates.json"))
    save_json({"filter_report": filter_report, "survivors": [h.id for h in survivors]},
              os.path.join(OUT, "filtered_candidates.json"))
    print(f"Pool: {[h.id + ':' + h.equation for h in pool]}")
    print(f"Duplicates removed: {[(r['id'], r['duplicate_of']) for r in removed]}")
    print(f"Survivors after structural + physics filters: {[h.id for h in survivors]}")
    for cid, r in filter_report.items():
        if r["physics_precheck"].get("warnings"):
            print(f"  {cid} physics warnings: {r['physics_precheck']['warnings']}")


def _load_survivors():
    gen = json.load(open(os.path.join(OUT, "generated_candidates.json")))
    filt = json.load(open(os.path.join(OUT, "filtered_candidates.json")))
    by_id = {p["id"]: Hypothesis(**p["hypothesis"]) for p in gen["pool"]}
    origin = {p["id"]: p["origin"] for p in gen["pool"]}
    return [by_id[i] for i in filt["survivors"]], origin


def _splits(cfg):
    p6 = cfg["phase6"]
    obs = load_observational_data(cfg["paths"]["output_dir"], "noisy")  # x, t, u only
    return make_splits(obs, p6["t_cutoff"], p6["val_fraction"], p6["split_seed"])


def _setup(cfg):
    d = cfg["diffusion"]
    return {"x_min": d["x_min"], "x_max": d["x_max"], "t_min": d["t_min"], "ic_kind": d["initial_condition"]}


def stage_pinn(cfg, only=None):
    p6, pcfg = cfg["phase6"], cfg["pinn"]
    survivors, _ = _load_survivors()
    splits = _splits(cfg)
    targets = [h for h in survivors if (not only or h.id in only)]
    for h in targets:
        print(f"\n=== Inverse PINN: {h.id}  {h.equation} ===")
        res, _ = validate_candidate(h, splits, _setup(cfg), pcfg, pcfg["training"], p6["gate"],
                                    p6["parameter_constraints"], p6["inactive_threshold"], MODEL_DIR)
        save_json(res, os.path.join(CAND_DIR, f"{h.id}.json"))
        print(f"  {res['status']}  params={ {k: round(v, 5) for k, v in res['params'].items()} }")
        print(f"  val_rmse={res['val_rmse']:.5f} physics_residual={res['physics_residual']:.2e} "
              f"ic={res['ic_loss']:.2e} failed={res['failed_checks']} inactive={res['inactive_parameters']}")


def oracle_evaluation(cfg, selected_hyp, selected_params, models, mlp, splits):
    """BENCHMARK-ONLY: uses the hidden synthetic truth to REPORT error after selection.
    Nothing returned here influences any decision."""
    out_dir = cfg["paths"]["output_dir"]
    full = np.load(os.path.join(out_dir, "full.npz"))
    x_grid, t_grid, U = full["x_grid"], full["t_grid"], full["U"]
    D_true = float(full["D_true"])
    tc = cfg["phase6"]["t_cutoff"]
    Xg, Tg = np.meshgrid(x_grid, t_grid)
    ood_mask = Tg > tc

    def nn_rmse(m):
        with torch.no_grad():
            p = m(torch.tensor(Xg[ood_mask], dtype=torch.float32).reshape(-1, 1),
                  torch.tensor(Tg[ood_mask], dtype=torch.float32).reshape(-1, 1)).numpy().ravel()
        return float(np.sqrt(np.mean((p - U[ood_mask]) ** 2)))

    per_model = {cid: nn_rmse(m) for cid, m in models.items()}
    per_model["MLP_data_only"] = nn_rmse(mlp)

    t_end = cfg["phase6"]["forward_prediction_t_end"]
    ic = lambda xx: initial_condition(xx, cfg["diffusion"]["initial_condition"])
    sol = solve_forward(selected_hyp, selected_params, ic, x_grid, t_end)
    # truth beyond the observed window: re-run the Phase 1 generator's solver with D_true (evaluation only)
    n_t_ext = int(round((len(t_grid) - 1) * t_end / t_grid[-1])) + 1
    t_ext = np.linspace(0, t_end, n_t_ext)
    U_true_ext = solve_diffusion_fd(D_true, x_grid, t_ext, cfg["diffusion"]["initial_condition"])
    U_true_at_sol = np.array([U_true_ext[np.argmin(np.abs(t_ext - tt))] for tt in sol["t"]])
    win = lambda lo, hi: (sol["t"] > lo) & (sol["t"] <= hi)
    err = lambda m: float(np.sqrt(np.mean((sol["U"][m] - U_true_at_sol[m]) ** 2))) if m.any() else float("nan")
    rel = abs(selected_params.get("D", float("nan")) - D_true) / D_true if "D" in selected_params else None
    return {
        "note": "ORACLE evaluation on the synthetic benchmark -- computed AFTER selection, never used for it.",
        "D_true": D_true, "selected_D_relative_error": rel,
        "pinn_ood_rmse_vs_clean_field_t_gt_cutoff": per_model,
        "forward_solve_rmse_vs_truth": {"t_in_(cutoff,1.0]": err(win(tc, 1.0)),
                                        "t_in_(1.0,t_end]_novel": err(win(1.0, t_end))},
    }, sol, (t_ext, U_true_ext)


def stage_select(cfg):
    p6, pcfg = cfg["phase6"], cfg["pinn"]
    survivors, origin = _load_survivors()
    splits = _splits(cfg)
    results = {}
    for h in survivors:
        path = os.path.join(CAND_DIR, f"{h.id}.json")
        if not os.path.exists(path):
            raise SystemExit(f"Missing PINN result for {h.id}; run --stage pinn --only {h.id} first.")
        results[h.id] = json.load(open(path))
    hyps = {h.id: h for h in survivors}
    save_json(results, os.path.join(OUT, "pinn_validation.json"))

    # empirical stability (cheap bootstrap on training observations only)
    rc = cfg["discovery"]["reconstruction"]["noisy"]
    xg = np.linspace(cfg["diffusion"]["x_min"], cfg["diffusion"]["x_max"], rc["n_x_recon"])
    tg = np.linspace(cfg["diffusion"]["t_min"], p6["t_cutoff"], rc["n_t_recon"])
    stability = {cid: bootstrap_stability(hyps[cid], splits["train"], xg, tg, rc["smoothing_sigma"], rc["crop"],
                                          n_boot=p6["stability"]["n_boot"], frac=p6["stability"]["subset_fraction"])
                 for cid in results}

    selection = select_model(list(results.values()), p6["equivalence_tol"], p6["inactive_threshold"])
    selection["stability"] = stability
    save_json(selection, os.path.join(OUT, "model_selection.json"))
    sel = selection["selected"]
    print(f"\nSelected: {sel}  ({selection['rule']})")
    print(f"Equivalence set: {selection['equivalence_set']}  BIC minimizer: {selection['bic_minimizer']}")

    # OOD on held-out observations (no oracle): PINN extrapolation per candidate + data-only MLP
    models = {cid: load_candidate_model(MODEL_DIR, cid, pcfg) for cid in results}
    mlp = train_data_only_mlp(splits, pcfg, epochs=pcfg["training"]["adam_epochs"])
    from hypotheses.search import _rmse
    ood_obs = {cid: results[cid]["ood_obs_rmse"] for cid in results}
    ood_obs["MLP_data_only"] = _rmse(mlp, splits["ood"])
    val_obs = {cid: results[cid]["val_rmse"] for cid in results}
    val_obs["MLP_data_only"] = _rmse(mlp, splits["val"])

    sel_hyp, sel_params = hyps[sel], results[sel]["params"]
    # forward solve of the selected LAW, compared with OOD observations (no oracle)
    x_grid = np.linspace(cfg["diffusion"]["x_min"], cfg["diffusion"]["x_max"], 61)
    ic = lambda xx: initial_condition(xx, cfg["diffusion"]["initial_condition"])
    sol = solve_forward(sel_hyp, sel_params, ic, x_grid, p6["forward_prediction_t_end"])
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((sol["t"], x_grid), sol["U"])
    pred_ood = interp(np.column_stack([splits["ood"]["t"], splits["ood"]["x"]]))
    ood_obs[f"forward_solve_{sel}"] = float(np.sqrt(np.mean((pred_ood - splits["ood"]["u"]) ** 2)))

    oracle, sol_or, (t_ext, U_true_ext) = oracle_evaluation(cfg, sel_hyp, sel_params, models, mlp, splits)
    ood = {
        "split": {"n_train": len(splits["train"]["u"]), "n_val": len(splits["val"]["u"]),
                  "n_ood": len(splits["ood"]["u"]), "t_cutoff": p6["t_cutoff"]},
        "in_distribution_val_rmse_noisy_obs": val_obs,
        "ood_rmse_noisy_obs_t_gt_cutoff": ood_obs,
        "noise_floor_note": "RMSE against NOISY observations cannot go below the measurement noise (~0.012).",
        "oracle_benchmark_evaluation": oracle,
        "novel_prediction": {"law": sel_hyp.equation, "params": sel_params,
                             "t_window": [1.0, p6["forward_prediction_t_end"]],
                             "max_amplitude_at_t_end": float(np.max(np.abs(sol_or["U"][-1])))},
    }
    save_json(ood, os.path.join(OUT, "ood_predictions.json"))
    _plots(cfg, results, models, mlp, splits, sol_or, t_ext, U_true_ext, sel)
    _report(cfg, results, selection, stability, ood)
    print(json.dumps({k: ood[k] for k in ("in_distribution_val_rmse_noisy_obs", "ood_rmse_noisy_obs_t_gt_cutoff")}, indent=2))
    print(json.dumps(oracle, indent=2))


def _plots(cfg, results, models, mlp, splits, sol, t_ext, U_true_ext, sel):
    tc = cfg["phase6"]["t_cutoff"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    ids = list(results)
    axes[0].bar(ids, [results[i]["val_rmse"] for i in ids], color=["tab:green" if i == sel else
                ("tab:blue" if results[i]["gate_passed"] else "tab:red") for i in ids])
    axes[0].axhline(cfg["phase6"]["gate"]["val_rmse_max"], ls="--", c="k", lw=1, label="gate")
    axes[0].set_yscale("log"); axes[0].set_title("Held-out RMSE (green=selected, red=rejected)"); axes[0].legend()
    axes[1].bar(ids, [results[i]["complexity_score"] for i in ids], color="tab:gray")
    axes[1].set_title("Equation complexity score")
    x_probe = 0.25
    xi = np.argmin(np.abs(np.linspace(0, 1, sol["U"].shape[1]) - x_probe))
    axes[2].plot(t_ext, U_true_ext[:, np.argmin(np.abs(np.linspace(0, 1, U_true_ext.shape[1]) - x_probe))],
                 "k-", lw=2, label="truth (oracle, eval only)")
    axes[2].plot(sol["t"], sol["U"][:, xi], "g--", label=f"forward solve of selected {sel}")
    tt = torch.linspace(0, 1.5, 200).reshape(-1, 1)
    with torch.no_grad():
        xx = torch.full_like(tt, x_probe)
        axes[2].plot(tt.numpy(), models[sel](xx, tt).numpy(), "b:", label=f"PINN {sel} (network extrapolation)")
        axes[2].plot(tt.numpy(), mlp(xx, tt).numpy(), "r-.", label="data-only MLP")
    m = np.abs(splits["train"]["x"] - x_probe) < 0.04
    axes[2].scatter(splits["train"]["t"][m], splits["train"]["u"][m], s=10, c="gray", label="train obs near x")
    m2 = np.abs(splits["ood"]["x"] - x_probe) < 0.04
    axes[2].scatter(splits["ood"]["t"][m2], splits["ood"]["u"][m2], s=14, c="orange", label="OOD obs (held out)")
    axes[2].axvline(tc, c="k", lw=0.8); axes[2].axvline(1.0, c="k", lw=0.8, ls=":")
    axes[2].set_title(f"u(x={x_probe}, t): train | OOD | novel (t>1)"); axes[2].set_xlabel("t")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "phase6_summary.png"), dpi=140)


def _report(cfg, results, selection, stability, ood):
    gen = json.load(open(os.path.join(OUT, "generated_candidates.json")))
    filt = json.load(open(os.path.join(OUT, "filtered_candidates.json")))
    sel = selection["selected"]
    L = ["# Phase 6 -- Physics-Constrained Generative Search: Report\n",
         f"LLM provider: `{gen['provider']}` (mode: {gen['generation_mode']}). "
         "**With the mock provider, LLM candidates are canned; nothing here is evidence that an LLM discovered a law.**\n",
         "## 1. Scientific evidence (RAG + KG)\n",
         f"Mechanisms surfaced by retrieval: {gen['scientific_evidence']['mechanisms']}  ",
         f"Documented generic equation forms: {gen['scientific_evidence']['documented_equation_forms']}  ",
         f"Answer-leakage check (D_true string in corpus): {gen['answer_leakage_check']['D_true_string_found_in_corpus_docs'] or 'none found'}\n",
         "## 2. Hypotheses (candidate pool)\n", "| ID | Origin | Equation |", "|---|---|---|"]
    L += [f"| {p['id']} | {p['origin']} | `{p['equation']}` |" for p in gen["pool"]]
    if gen["duplicates_removed"]:
        L.append("\nRemoved as structural duplicates: " + ", ".join(f"{d['id']} (= {d['duplicate_of']})" for d in gen["duplicates_removed"]))
    L += ["\n## 3. Structural validation and 4. Physics filtering\n", "| ID | Structural | Physics pre-check | Warnings |", "|---|---|---|---|"]
    for cid, r in filt["filter_report"].items():
        L.append(f"| {cid} | {r['structural']['valid']} | {r['physics_precheck']['passed']} | "
                 f"{'; '.join(r['physics_precheck'].get('warnings', [])) or '-'} |")
    L += ["\n## 5. Parameter estimation (inverse PINN) and 6. PINN validation\n",
          "Trained on t <= %.1f (train split only). Gate uses the held-out in-distribution split of NOISY observations." % cfg["phase6"]["t_cutoff"],
          "\n| ID | Params | Val RMSE | Physics residual | IC loss | Bounds OK | Status | Failed checks |",
          "|---|---|---|---|---|---|---|---|"]
    for cid, r in results.items():
        L.append(f"| {cid} | {', '.join(f'{k}={v:.5g}' for k, v in r['params'].items())} | {r['val_rmse']:.4f} | "
                 f"{r['physics_residual']:.2e} | {r['ic_loss']:.2e} | {r['physical_bounds']['passed']} | {r['status']} | "
                 f"{', '.join(r['failed_checks']) or '-'} |")
    L += ["\n## 7. Model selection\n", f"Rule: {selection['rule']}.\n",
          f"Best held-out RMSE {selection.get('best_val_rmse', float('nan')):.5f}; equivalence cutoff "
          f"{selection.get('equivalence_cutoff', float('nan')):.5f}; equivalence set {selection['equivalence_set']}.\n",
          "| ID | Complexity | BIC | Per-parameter contribution (RMS share of u_t) | Inactive | Reduced form | Bootstrap sign-stable |",
          "|---|---|---|---|---|---|---|"]
    for cid, r in results.items():
        L.append(f"| {cid} | {r['complexity_score']} | {r['bic']:.1f} | "
                 f"{', '.join(f'{k}:{v:.3f}' for k, v in r['contributions'].items())} | {r['inactive_parameters'] or '-'} | "
                 f"{('`' + r['reduced_equation_if_inactive_terms_dropped'] + '`') if r['inactive_parameters'] else '-'} | "
                 f"{stability[cid]['all_terms_sign_stable']} |")
    L.append(f"\n**Selected: {sel}** (`{results[sel]['equation']}`). BIC minimizer: {selection['bic_minimizer']} "
             f"(agrees with selection: {selection['bic_agrees_with_selection']}).\n")
    for e in selection["explanations"]:
        L.append(f"- {e['id']}: {e['role']}" + (f"; inactive parameters {e['inactive_parameters']}" if e['inactive_parameters'] else "")
                 + ("; with inactive terms removed it reduces to the selected equation" if e.get("reduces_to_selected") else ""))
    L += ["\nEmpirical stability (bootstrap OLS on finite-difference derivatives of 20 random 80% subsets; NOT Bayesian):\n"]
    for cid, s in stability.items():
        L.append(f"- {cid}: " + "; ".join(f"`{m}` mean {v['mean']:.4g} ± {v['std']:.2g}, sign-consistent {v['sign_consistency']:.0%}"
                                           for m, v in s["monomials"].items()))
    # Cross-check: does the cheap FD bootstrap agree with the PINN term-necessity analysis?
    from validation.stability import candidate_monomials
    from hypotheses.equation_compiler import parse_equation as _pe
    hyps = {p["id"]: Hypothesis(**p["hypothesis"]) for p in gen["pool"]}
    disagreements = []
    for cid, r in results.items():
        if not r["inactive_parameters"]:
            continue
        h = hyps[cid]
        _, _, _, ld = _pe(h.equation, [q.name for q in h.parameters])
        monos, _ = candidate_monomials(h)
        rhs_terms = __import__("sympy").Add.make_args(__import__("sympy").expand(
            -( _pe(h.equation, [q.name for q in h.parameters])[0] - ld["u_t"])))
        for term in rhs_terms:
            if any(term.has(ld[pn]) for pn in r["inactive_parameters"]):
                for m in monos:
                    st = stability[cid]["monomials"].get(str(m))
                    if st and term.has(m) and st["sign_consistency"] >= 0.9 and str(m) not in [str(x) for x in disagreements]:
                        disagreements.append(f"{cid}: PINN finds {r['inactive_parameters']} inactive, but the FD bootstrap "
                                             f"gives a sign-stable `{m}` coefficient ({st['mean']:.3g})")
    if disagreements:
        L.append("\n**Methods disagree** (reported, not resolved):")
        L += [f"- {d}" for d in sorted(set(disagreements))]
        L.append("The same spurious `u`/`u**2` terms appear in Phase 2's noisy full-library SINDy result, which points to a "
                 "finite-difference reconstruction/smoothing artifact on 400 noisy points rather than real reaction physics; the "
                 "PINN analysis fits the scattered data directly without grid reconstruction. This is an interpretation, not a proof. "
                 "The FD bootstrap's `u_xx` coefficient (~0.11) is also biased high relative to the PINN estimate, consistent with that artifact.")
    L += ["\n## 8. OOD prediction\n",
          "Held-out observations with t > %.1f were never used for training or selection.\n" % cfg["phase6"]["t_cutoff"],
          "| Model | In-dist held-out RMSE (noisy obs) | OOD RMSE (noisy obs) | OOD RMSE vs clean field (oracle, eval only) |",
          "|---|---|---|---|"]
    orc = ood["oracle_benchmark_evaluation"]["pinn_ood_rmse_vs_clean_field_t_gt_cutoff"]
    for k, v in ood["ood_rmse_noisy_obs_t_gt_cutoff"].items():
        L.append(f"| {k} | {ood['in_distribution_val_rmse_noisy_obs'].get(k, float('nan')):.5f} | {v:.5f} | "
                 f"{orc.get(k, float('nan')):.5f} |")
    fs = ood["oracle_benchmark_evaluation"]["forward_solve_rmse_vs_truth"]
    L += [f"\nNoisy-observation RMSE has a floor at the measurement noise (~0.012); the oracle column shows error against the noise-free field.\n",
          f"**Novel prediction** (selected law solved forward from the known IC to t = {cfg['phase6']['forward_prediction_t_end']}, "
          f"beyond the observed window, using no observations at all): oracle RMSE {fs['t_in_(cutoff,1.0]']:.5f} on (0.8, 1.0] and "
          f"{fs['t_in_(1.0,t_end]_novel']:.5f} on (1.0, {cfg['phase6']['forward_prediction_t_end']}].",
          f"Selected-parameter relative error vs hidden truth (oracle): "
          f"{ood['oracle_benchmark_evaluation']['selected_D_relative_error']}\n",
          "![summary](phase6_summary.png)\n",
          "## 9. Limitations -- what cannot be concluded\n",
          "- The candidate was **supported by the available observations, physics constraints and OOD validation**. That is not proof, and not uniqueness: nested candidates fit equally well; selection between them is a stated parsimony rule.",
          "- Synthetic data from a known PDE; one physical system; one noise level; 400 points.",
          "- Mock LLM: candidate proposals are canned. KG templates come from a 5-document hand-written fixture corpus.",
          "- Empirical stability is a finite-difference regression bootstrap, not PINN retraining and not Bayesian uncertainty.",
          "- The initial condition is supplied as known physics, not inferred.",
          "- Thresholds are hand-set (fixed before the run) and were not calibrated on an independent benchmark."]
    with open(os.path.join(OUT, "phase6_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Saved {OUT}/phase6_report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", default="all", choices=["prepare", "pinn", "select", "all"])
    ap.add_argument("--only", default=None, help="comma-separated candidate ids for --stage pinn")
    args = ap.parse_args()
    cfg = load_config(args.config)
    only = set(args.only.split(",")) if args.only else None
    if args.stage in ("prepare", "all"):
        stage_prepare(cfg)
    if args.stage in ("pinn", "all"):
        stage_pinn(cfg, only)
    if args.stage in ("select", "all"):
        stage_select(cfg)


if __name__ == "__main__":
    main()
