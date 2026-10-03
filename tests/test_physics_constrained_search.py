"""
Phase 6 tests. Offline, no API key (MockLLMProvider), deterministic.
The inverse-PINN integration test uses a tiny training budget -- it checks
wiring, not accuracy (accuracy is checked by the saved-results test, which
skips if the full experiment hasn't been run).

    python -m pytest tests/test_physics_constrained_search.py -v
"""

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypotheses.complexity import equation_complexity
from hypotheses.constraints import check_fitted_bounds, precheck_physics, resolve_bounds
from hypotheses.pipeline import generate_hypotheses
from hypotheses.search import (build_candidate_pool, kg_template_candidates, make_splits,
                               structural_and_physics_filter, validate_candidate)
from hypotheses.validation import canonical_signature
from llm.mock_provider import MockLLMProvider
from llm.schemas import Hypothesis
from validation.forward_solver import solve_forward
from validation.model_selection import parameter_contributions, reduced_signature, select_model

SPEC = {"has_ic": True, "dirichlet_both_ends": True}
FALLBACK = {"D": {"lower": 0.0}, "K": {"lower": 0.0}}


def H(id_, eq, params, **kw):
    return Hypothesis(id=id_, name=id_, equation=eq,
                      parameters=[p if isinstance(p, dict) else {"name": p, "initial_value": 0.2} for p in params], **kw)


def _mock_candidates():
    return generate_hypotheses("field decays and smooths", MockLLMProvider())["valid_hypotheses"]


def _context_and_graph():
    from knowledge_graph.builder import build_graph_from_documents
    from rag.context import build_scientific_context
    from rag.corpus_loader import load_corpus
    from rag.pipeline import RAGPipeline
    docs = load_corpus()
    rag = RAGPipeline()
    chunks = rag.ingest(docs)
    graph = build_graph_from_documents(docs, {c.chunk_id: c for c in chunks})
    ctx = build_scientific_context("amplitude decreases and the profile smooths out, no bulk translation",
                                   rag.get_retriever(), graph, top_k=5)
    return ctx, graph


# --- candidate generation (Phase 4 + Phase 5 compatibility) ---
def test_candidate_pool_merges_llm_and_kg_with_dedup():
    ctx, graph = _context_and_graph()
    kg = kg_template_candidates(ctx.model_dump(), graph, {"_default": 0.5})
    assert len(kg) >= 1 and all(h.id.startswith("K") for h in kg)
    pool, removed, origin = build_candidate_pool(_mock_candidates(), kg)
    sigs = [canonical_signature(h) for h in pool]
    assert len(sigs) == len(set(sigs)), "pool contains structural duplicates"
    assert set(origin.values()) <= {"llm", "kg_template"}
    assert all(r["duplicate_of"] for r in removed)


def test_kg_templates_carry_no_fitted_values():
    ctx, graph = _context_and_graph()
    for h in kg_template_candidates(ctx.model_dump(), graph, {"D": 0.2, "_default": 0.5}):
        assert "0.1" not in h.equation  # generic forms only, never the benchmark's hidden value


# --- structural validation / invalid rejection ---
def test_structural_filter_rejects_malformed_candidates():
    good = H("A", "u_t = D*u_xx", ["D"])
    undeclared = H("B", "u_t = D*u_xx + q*u", ["D"])  # 'q' undeclared
    static = H("C", "0 = D*u_xx", ["D"])               # not an evolution equation
    survivors, report = structural_and_physics_filter([good, undeclared, static], SPEC, FALLBACK)
    assert [h.id for h in survivors] == ["A"]
    assert not report["B"]["structural"]["valid"]
    assert report["C"]["structural"]["valid"] and not report["C"]["physics_precheck"]["passed"]


# --- physical constraints ---
def test_bounds_are_per_parameter_not_blanket():
    h = H("A", "u_t + c*u_x = D*u_xx", ["c", "D"])
    b = resolve_bounds(h, FALLBACK)
    assert b["D"]["lower"] == 0.0 and b["c"]["lower"] is None


def test_hypothesis_declared_bound_overrides_fallback():
    h = H("A", "u_t = D*u_xx", [{"name": "D", "initial_value": 0.5, "lower_bound": 0.3}])
    assert resolve_bounds(h, FALLBACK)["D"]["lower"] == 0.3


def test_post_fit_bound_violation_detected():
    h = H("A", "u_t = D*u_xx", ["D"])
    assert not check_fitted_bounds({"D": -0.05}, resolve_bounds(h, FALLBACK))["passed"]
    assert check_fitted_bounds({"D": 0.05}, resolve_bounds(h, FALLBACK))["passed"]


def test_precheck_warns_on_overdetermined_advection():
    p = precheck_physics(H("A", "u_t + c*u_x = 0", ["c"]), SPEC, FALLBACK)
    assert p["passed"] and any("over-determined" in w for w in p["warnings"])


def test_precheck_rejects_bad_initial_value():
    p = precheck_physics(H("A", "u_t = D*u_xx", [{"name": "D", "initial_value": -1.0}]), SPEC, FALLBACK)
    assert not p["passed"]


# --- complexity ---
def test_complexity_orders_nested_models():
    c1 = equation_complexity(H("1", "u_t = D*u_xx", ["D"]))
    c3 = equation_complexity(H("3", "u_t = D*u_xx + r*u*(1-u/K)", ["D", "r", "K"]))
    c4 = equation_complexity(H("4", "u_t + c*u_x = D*u_xx", ["c", "D"]))
    assert c1["complexity_score"] < c4["complexity_score"] < c3["complexity_score"]
    assert c3["n_nonlinear_terms"] == 1 and c1["n_nonlinear_terms"] == 0
    assert c1["max_derivative_order"] == 2


# --- model selection / nested-model handling ---
def _cand(id_, rmse, cx, k, gate=True, contrib=None, sig=None, red=None):
    return {"id": id_, "gate_passed": gate, "val_rmse": rmse, "complexity_score": cx,
            "bic": 100 * math.log(rmse ** 2) + k * math.log(100), "contributions": contrib or {},
            "signature": sig or id_, "reduced_signature": red or sig or id_}


def test_selection_prefers_simplest_among_equivalent_fits():
    cands = [_cand("simple", 0.01215, 4.1, 1, sig="S"),
             _cand("nested", 0.01213, 10.7, 3, contrib={"D": 1.0, "r": 0.01}, sig="N", red="S"),  # marginally better fit
             _cand("wrong", 0.20, 3.2, 1, gate=False)]
    sel = select_model(cands, equivalence_tol=0.05, inactive_threshold=0.05)
    assert sel["selected"] == "simple"
    assert set(sel["equivalence_set"]) == {"simple", "nested"}
    nested_note = next(e for e in sel["explanations"] if e["id"] == "nested")
    assert nested_note["inactive_parameters"] == ["r"] and nested_note.get("reduces_to_selected")


def test_selection_does_not_pick_simple_model_that_fits_worse():
    cands = [_cand("simple", 0.030, 3.0, 1), _cand("complex", 0.012, 9.0, 3)]
    assert select_model(cands, 0.05, 0.05)["selected"] == "complex"


def test_selection_handles_no_supported_candidates():
    assert select_model([_cand("a", 0.5, 1, 1, gate=False)], 0.05, 0.05)["selected"] is None


def test_reduced_signature_maps_nested_model_to_simpler_one():
    h3 = H("3", "u_t = D*u_xx + r*u*(1-u/K)", ["D", "r", "K"])
    h1 = H("1", "u_t = D*u_xx", ["D"])
    sig, eq = reduced_signature(h3, ["r", "K"])
    assert sig == canonical_signature(h1)


class _ExactDiffusion(nn.Module):
    """u = sin(pi x) exp(-pi^2 D t): an exact diffusion solution."""
    def __init__(self, D=0.1):
        super().__init__()
        self.D = D

    def forward(self, x, t):
        return torch.sin(math.pi * x) * torch.exp(-math.pi ** 2 * self.D * t)


def test_parameter_contributions_flag_inactive_reaction():
    h3 = H("3", "u_t = D*u_xx + r*u*(1-u/K)", ["D", "r", "K"])
    c = parameter_contributions(_ExactDiffusion(), h3, {"D": 0.1, "r": 0.0, "K": 1.0}, (0, 1), (0, 0.8), n=20)
    assert abs(c["D"] - 1.0) < 1e-3 and c["r"] < 1e-6


# --- OOD integration / forward solver ---
def test_splits_are_disjoint_and_ood_is_strictly_later():
    rng = np.random.default_rng(0)
    obs = {"x": rng.uniform(0, 1, 300), "t": rng.uniform(0, 1, 300), "u": rng.normal(size=300)}
    s = make_splits(obs, 0.8, 0.2, seed=0)
    assert s["ood"]["t"].min() > 0.8 and s["train"]["t"].max() <= 0.8 and s["val"]["t"].max() <= 0.8
    assert len(s["train"]["u"]) + len(s["val"]["u"]) + len(s["ood"]["u"]) == 300
    train_pts = set(zip(s["train"]["x"], s["train"]["t"]))
    assert not train_pts & set(zip(s["val"]["x"], s["val"]["t"]))


def test_forward_solver_matches_exact_diffusion():
    x = np.linspace(0, 1, 61)
    sol = solve_forward(H("1", "u_t = D*u_xx", ["D"]), {"D": 0.1}, lambda xx: np.sin(np.pi * xx), x, 0.5)
    exact = np.sin(np.pi * x) * np.exp(-np.pi ** 2 * 0.1 * 0.5)
    assert sol["stable"] and np.abs(sol["U"][-1] - exact).max() < 2e-3


# --- inverse PINN integration (tiny budget: wiring, not accuracy) ---
def test_inverse_pinn_integration_runs_and_reports():
    rng = np.random.default_rng(1)
    x, t = rng.uniform(0, 1, 120), rng.uniform(0, 1, 120)
    obs = {"x": x, "t": t, "u": np.sin(np.pi * x) * np.exp(-np.pi ** 2 * 0.1 * t)}
    splits = make_splits(obs, 0.8, 0.2, seed=0)
    setup = {"x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "ic_kind": "sin"}
    pcfg = {"architecture": {"hidden_dims": [16, 16], "activation": "tanh"},
            "weights": {"lambda_data": 1.0, "lambda_physics": 1.0, "lambda_bc": 1.0, "lambda_ic": 1.0}}
    tcfg = {"n_collocation": 100, "n_bc": 20, "n_ic": 20, "adam_epochs": 30, "adam_lr": 1e-3, "lbfgs_epochs": 0, "seed": 0}
    gate = {"val_rmse_max": 0.03, "physics_residual_max": 0.01, "bc_loss_max": 0.005, "ic_loss_max": 0.01}
    res, model = validate_candidate(H("A", "u_t = D*u_xx", ["D"]), splits, setup, pcfg, tcfg, gate, FALLBACK, 0.05)
    for key in ("params", "val_rmse", "ood_obs_rmse", "physics_residual", "gate_passed", "complexity",
                "contributions", "bic", "physical_bounds"):
        assert key in res
    assert res["decision_data"].startswith("held-out noisy observations")
    assert np.isfinite(res["ood_obs_rmse"])


# --- saved full-experiment results (skip if not yet run) ---
def test_phase6_saved_results_if_present():
    path = "results/phase6/model_selection.json"
    if not os.path.exists(path):
        print("[SKIP] run experiments/08_physics_constrained_search.py first")
        return
    sel = json.load(open(path))
    pinn = json.load(open("results/phase6/pinn_validation.json"))
    assert sel["selected"] == "H1"
    assert not pinn["H2"]["gate_passed"]
    assert pinn["H3"]["gate_passed"] and set(pinn["H3"]["inactive_parameters"]) == {"r", "K"}
    assert abs(pinn["H1"]["params"]["D"] - 0.1) / 0.1 < 0.05


# --- conservation-law validation ---
def test_conservation_violation_exact_solution():
    """Exact diffusion solution: ~0 global balance violation under the true law,
    a clear violation under a wrong diffusivity."""
    from hypotheses.equation_compiler import compile_residual
    from validation.conservation import conservation_violation

    class Exact(nn.Module):
        def forward(self, x, t):
            return torch.exp(-0.1 * math.pi ** 2 * t) * torch.sin(math.pi * x)

    residual_fn = compile_residual("u_t = D*u_xx", ["D"])[0]
    good = conservation_violation(Exact(), residual_fn, {"D": torch.tensor(0.1)}, 0.0, 1.0, 0.0, 1.0)
    bad = conservation_violation(Exact(), residual_fn, {"D": torch.tensor(0.2)}, 0.0, 1.0, 0.0, 1.0)
    assert good["relative_violation"] < 1e-3
    assert bad["relative_violation"] > 0.5
