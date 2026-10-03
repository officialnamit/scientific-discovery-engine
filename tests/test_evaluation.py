"""
Phase 7 tests. Offline, deterministic, no API key.

    python -m pytest tests/test_evaluation.py -v
"""

import json
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.datasets import benchmark_datasets
from evaluation.describe import describe_observations, observation_stats
from evaluation.engine import Resources, build_pool, family_of, score_run, validate_and_select
from evaluation.identify import (discovered_coefficients, grid_fields, library_collinearity,
                                 match_mechanism, recover_parameters, term_set)
from evaluation.leakage import decision_functions, ground_truth_references
from llm.schemas import Hypothesis


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open("config.yaml"))


@pytest.fixture(scope="module")
def res(cfg):
    return Resources(cfg)


@pytest.fixture(scope="module")
def datasets():
    return {d.name: d for d in benchmark_datasets()}


# --- datasets: exact solutions really satisfy their PDE (answer key is trustworthy) ---
def test_datasets_cover_four_families(datasets):
    assert {d.family for d in datasets.values()} == {"diffusion", "advection", "reaction-diffusion", "advection-diffusion"}
    assert len(datasets) == 12


@pytest.mark.parametrize("name", ["diffusion_D0.1", "advection_c0.5", "advdiff_c0.4_D0.02"])
def test_exact_solutions_satisfy_their_equation(datasets, name):
    d = datasets[name]
    f = grid_fields(d.U, d.x, d.t, crop=5)
    c, D = d.truth.get("c", 0.0), d.truth.get("D", 0.0)
    resid = f["u_t"] + c * f["u_x"] - D * f["u_xx"]
    assert np.sqrt(np.mean(resid ** 2)) / np.sqrt(np.mean(f["u_t"] ** 2)) < 0.01


def test_dataset_cache_is_bit_identical():
    a, b = benchmark_datasets(use_cache=False), benchmark_datasets(use_cache=True)
    assert all(np.array_equal(x.U, y.U) for x, y in zip(a, b))


# --- describer: no mechanism names, deterministic ---
def test_description_never_names_a_mechanism(datasets):
    for d in datasets.values():
        text = describe_observations(d.x, d.t, d.U).lower()
        for word in ["diffusion", "advection", "reaction", "fick", "logistic"]:
            assert word not in text


def test_observation_stats_detect_translation(datasets):
    assert abs(observation_stats(*(lambda d: (d.x, d.t, d.U))(datasets["advection_c0.5"]))["drift_frac"]) > 0.2
    assert abs(observation_stats(*(lambda d: (d.x, d.t, d.U))(datasets["diffusion_D0.1"]))["drift_frac"]) < 0.05


# --- identification / metrics ---
def test_term_set_and_parameter_recovery():
    rd = Hypothesis(id="r", name="r", equation="u_t = D*u_xx + r*u*(1-u/K)", parameters=[{"name": n} for n in "DrK"])
    assert term_set(rd) == frozenset({"u_xx", "u", "u**2"})
    p = recover_parameters(rd, {"u_xx": 0.05, "u": 1.5, "u**2": -1.875})
    assert abs(p["D"] - 0.05) < 1e-9 and abs(p["r"] - 1.5) < 1e-9 and abs(p["K"] - 0.8) < 1e-9


def test_discovered_coefficients_and_matching():
    sindy = {"terms": [{"term": "u_x", "coefficient": -0.3}, {"term": "u_xx", "coefficient": 0.01}]}
    co = discovered_coefficients(sindy)
    ad = Hypothesis(id="a", name="advection-diffusion", equation="u_t + c*u_x = D*u_xx", parameters=[{"name": "c"}, {"name": "D"}])
    assert match_mechanism(co, [ad]) is ad
    assert recover_parameters(ad, co) == pytest.approx({"c": 0.3, "D": 0.01})


def test_collinearity_flags_eigenfunction_trap():
    x = t = np.linspace(0, 1, 81)
    X, T = np.meshgrid(x, t)
    single = np.sin(np.pi * X) * np.exp(-np.pi ** 2 * 0.1 * T)
    two = single + 0.5 * np.sin(3 * np.pi * X) * np.exp(-9 * np.pi ** 2 * 0.1 * T)
    assert not library_collinearity(grid_fields(single, x, t))["identifiable"]
    assert library_collinearity(grid_fields(two, x, t))["identifiable"]


def test_score_run_metrics_on_known_case(datasets, res, cfg):
    d = datasets["advdiff_c0.3_D0.01"]
    run = validate_and_select(build_pool(d.x, d.t, d.U, res)[0], d.x, d.t, d.U, cfg)
    s = score_run(d, run, cfg["phase7"]["ood_t_end"])
    assert s["correct_model"] and s["selected_family"] == "advection-diffusion"
    assert s["max_rel_param_error"] < 0.05 and s["ood_rel_rmse_t_1_to_end"] < 0.05
    assert s["n_wrong_rejected"] == s["n_wrong_candidates"]


# --- selection rule: revised vs original, both reported ---
def test_revised_rule_fixes_discretization_degeneracy_original_does_not(datasets, res, cfg):
    d = datasets["diffusion_D0.1"]
    pool = build_pool(d.x, d.t, d.U, res)[0]
    orig = validate_and_select(pool, d.x, d.t, d.U, cfg, prefer_reduced=False)
    rev = validate_and_select(pool, d.x, d.t, d.U, cfg, prefer_reduced=True)
    assert family_of(orig["selected_hypothesis"]) == "reaction-diffusion"   # disclosed failure of the Phase 6 rule
    assert family_of(rev["selected_hypothesis"]) == "diffusion"
    assert rev["selection"]["reduction_applied"]["inactive_parameters"]


def test_revised_rule_does_not_strip_real_reaction(datasets, res, cfg):
    d = datasets["reacdiff_D0.1_r1.5_K0.8"]
    run = validate_and_select(build_pool(d.x, d.t, d.U, res)[0], d.x, d.t, d.U, cfg)
    assert family_of(run["selected_hypothesis"]) == "reaction-diffusion"


# --- determinism ---
def test_engine_is_deterministic(datasets, res, cfg):
    d = datasets["advection_c-0.4"]
    a = validate_and_select(build_pool(d.x, d.t, d.U, res)[0], d.x, d.t, d.U, cfg)
    b = validate_and_select(build_pool(d.x, d.t, d.U, res)[0], d.x, d.t, d.U, cfg)
    assert json.dumps(a["rows"], default=str) == json.dumps(b["rows"], default=str)


# --- leakage ---
def test_decision_functions_do_not_reference_ground_truth():
    assert {k: v for k, v in ((n, ground_truth_references(f)) for n, f in decision_functions().items()) if v} == {}


def _cheating_selector(rows, ds):
    return min(rows, key=lambda r: abs(r["D"] - ds.truth["D"]))


def test_leakage_audit_is_not_vacuous():
    assert "truth" in ground_truth_references(_cheating_selector)


def test_validate_and_select_cannot_receive_truth():
    import inspect
    assert "ds" not in inspect.signature(validate_and_select).parameters


# --- suites (fast checks; full numbers checked from saved results) ---
def test_adversarial_suite_runs_and_has_20_cases(datasets, res, cfg):
    from evaluation.adversarial import run_adversarial
    cases = run_adversarial(list(datasets.values()), res, cfg)
    assert len(cases) == 20 and len({c["id"] for c in cases}) == 20
    assert all(c["expected"] for c in cases)


def test_benchmark_counts(datasets, res, cfg):
    from evaluation.benchmark import run_multihop, run_single_questions
    q, runs = run_single_questions(list(datasets.values()), res, cfg)
    m = run_multihop(list(datasets.values()), res, cfg, runs)
    assert len(q) == 50 and len({x["id"] for x in q}) == 50
    assert len(m) == 25 and all(x["hops"] >= 3 for x in m)


def test_saved_phase7_results_if_present():
    p = "results/evaluation/ablation.json"
    if not os.path.exists(p):
        pytest.skip("run experiments/09_evaluation.py first")
    summ = {r["arm"]: r for r in json.load(open(p))["summary"]}
    assert summ["full_system"]["correct_model_rate"] == 1.0
    assert summ["rag_kg_generation"]["correct_model_rate"] < summ["full_system"]["correct_model_rate"]
    assert summ["discovery_only"]["correct_model_rate"] < 1.0
    leak = json.load(open("results/evaluation/leakage_and_reproducibility.json"))
    assert leak["passed"] and leak["deterministic_rerun_identical"]
    llm = json.load(open("results/evaluation/real_llm.json"))
    if not llm["available"]:
        assert llm["results"] is None  # no fabricated real-LLM results
    g = "results/evaluation/generalization/generalization_results.json"
    if os.path.exists(g):
        gen = json.load(open(g))
        assert gen["selected_family"] == "advection-diffusion"
        assert all(e < 0.05 for e in gen["oracle_benchmark_evaluation"]["selected_param_rel_error"].values())
