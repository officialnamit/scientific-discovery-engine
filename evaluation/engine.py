"""
Phase 7: the discovery pipeline as used by the benchmark, ablation and
adversarial suites. It is the Phase 6 search with ONE substitution: the
fitting engine is OLS on finite-difference derivatives of the (clean,
gridded) benchmark data instead of a ~4-minute inverse PINN per candidate.
Everything else is the existing code:

  describe_observations -> RAG retrieval -> KG templates (+ mock-LLM pool)   [Phases 4-5]
  -> structural + physics pre-filter                                          [Phase 6]
  -> fit (regression engine) -> post-fit physical bounds                     [Phase 6 bounds]
  -> gate on held-out points -> select_model (equivalence + complexity + BIC) [Phase 6]
  -> forward solve of the selected law beyond the data window (OOD)          [Phase 6 solver]

The PINN version of this exact pipeline is run on a second physical system in
experiments/10_generalization.py.

Ground truth (ds.truth / ds.exact) is read ONLY in score_* functions, which run
after every decision has been made.
"""

import math
import time
from typing import Dict, List, Optional

import numpy as np
import sympy

from evaluation.describe import describe_observations
from evaluation.identify import fit_candidate_on_grid, grid_fields, term_set
from hypotheses.complexity import equation_complexity
from hypotheses.constraints import check_fitted_bounds, resolve_bounds
from hypotheses.equation_compiler import FIELD_SYMBOLS, parse_equation
from hypotheses.pipeline import generate_hypotheses
from hypotheses.search import build_candidate_pool, kg_template_candidates, structural_and_physics_filter
from llm.mock_provider import MockLLMProvider
from llm.schemas import Hypothesis
from rag.context import build_scientific_context
from validation.forward_solver import solve_forward
from validation.model_selection import select_model

PROBLEM_SPEC = {"has_ic": True, "dirichlet_both_ends": True}


class Resources:
    """RAG index + KG built once and shared (deterministic)."""
    def __init__(self, cfg: Dict):
        from knowledge_graph.builder import build_graph_from_documents
        from rag.corpus_loader import load_corpus
        from rag.pipeline import RAGPipeline
        self.cfg = cfg
        self.docs = load_corpus()
        self.rag = RAGPipeline()
        chunks = self.rag.ingest(self.docs)
        self.graph = build_graph_from_documents(self.docs, {c.chunk_id: c for c in chunks})
        self.retriever = self.rag.get_retriever()


def build_pool(ds_x, ds_t, ds_U, res: Resources, use_rag=True, use_kg=True, use_llm=True, top_k=5):
    """Returns (pool, context, description). use_kg=False keeps only equations that literally
    appear in retrieved evidence (RAG-only); use_llm adds the (mock) LLM proposals."""
    desc = describe_observations(ds_x, ds_t, ds_U)
    ctx = build_scientific_context(desc, res.retriever, res.graph if use_kg else None, top_k=top_k) if use_rag else None
    ctx_d = ctx.model_dump() if ctx else None
    llm = (generate_hypotheses(desc, MockLLMProvider(), context=ctx_d)["valid_hypotheses"] if use_llm else [])
    if use_rag and use_kg:
        evid = kg_template_candidates(ctx_d, res.graph, res.cfg["phase6"]["parameter_defaults"])
    elif use_rag:
        evid = []
        for i, eq in enumerate(sorted({q for e in ctx.retrieved_evidence for q in e.equations}), start=1):
            names = sorted({str(s) for s in sympy.sympify(eq.split("=")[1]).free_symbols} |
                           {str(s) for s in sympy.sympify(eq.split("=")[0]).free_symbols} - set(FIELD_SYMBOLS))
            names = [n for n in names if n not in FIELD_SYMBOLS]
            evid.append(Hypothesis(id=f"R{i}", name=f"RAG evidence equation", equation=eq,
                                   parameters=[{"name": n, "initial_value": 0.5} for n in names]))
    else:
        evid = []
    pool, removed, origin = build_candidate_pool(llm, evid)
    return pool, origin, ctx, desc


def _grid_param_contributions(hyp, params, fields):
    names = [p.name for p in hyp.parameters]
    residual, _, used, local = parse_equation(hyp.equation, names)
    u_t = local["u_t"]
    ex = sympy.expand(residual)
    rhs = sympy.expand(-(ex - ex.coeff(u_t) * u_t))
    syms = [local[f] for f in FIELD_SYMBOLS]
    ut_rms = float(np.sqrt(np.mean(fields["u_t"] ** 2))) or 1e-12
    out = {}
    if not all(p in params for p in used):
        return out
    for p in used:
        grp = sympy.Add(*[t for t in sympy.Add.make_args(rhs) if t.has(local[p])])
        f = sympy.lambdify(syms + [local[q] for q in used], grp, "numpy")
        vals = np.broadcast_to(np.asarray(f(*[fields.get(s, np.zeros_like(fields["u"])) for s in FIELD_SYMBOLS],
                                            *[params[q] for q in used]), dtype=float), fields["u"].shape)
        out[p] = float(np.sqrt(np.mean(vals ** 2)) / ut_rms)
    return out


def _reduced_term_set(hyp: Hypothesis, inactive: List[str]) -> Optional[str]:
    if not inactive:
        return None
    names = [p.name for p in hyp.parameters]
    residual, _, _, local = parse_equation(hyp.equation, names)
    u_t = local["u_t"]
    ex = sympy.expand(residual)
    rhs = sympy.expand(-(ex - ex.coeff(u_t) * u_t))
    fsyms = [local[f] for f in FIELD_SYMBOLS if f != "u_t"]
    kept = set()
    for term in sympy.Add.make_args(rhs):
        if any(term.has(local[p]) for p in inactive):
            continue
        _, field_part = term.as_independent(*fsyms, as_Add=False)
        if field_part != 1:
            kept.add(str(field_part))
    return str(sorted(kept))


def validate_and_select(pool: List[Hypothesis], x, t, U, cfg: Dict, seed: int = 0,
                        apply_validation: bool = True, prefer_reduced: bool = True) -> Dict:
    """apply_validation=False reproduces 'no physics validation' ablation arms: the
    top of the Phase 4 ranking is taken as-is (candidates are still fitted so that
    parameter/OOD error can be measured for the arm)."""
    p6, p7 = cfg["phase6"], cfg["phase7"]
    survivors, filt = structural_and_physics_filter(pool, PROBLEM_SPEC, p6["parameter_constraints"])
    fields = grid_fields(U, x, t, crop=3)
    rng = np.random.default_rng(seed)
    val_mask = rng.random(len(fields["u_t"])) < 0.2
    ut_val_rms = float(np.sqrt(np.mean(fields["u_t"][val_mask] ** 2))) or 1e-12
    n_train = int((~val_mask).sum())

    rows = []
    for h in survivors:
        fit = fit_candidate_on_grid(h, fields, val_mask)
        params = fit["params"]
        bounds = check_fitted_bounds(params, resolve_bounds(h, p6["parameter_constraints"])) if params else \
            {"passed": False, "violations": ["parameters could not be recovered from fitted coefficients"]}
        rel = fit["val_rmse"] / ut_val_rms
        contrib = _grid_param_contributions(h, params, fields)
        k = len(h.parameters)
        rows.append({
            "id": h.id, "equation": h.equation, "params": params, "val_rmse": fit["val_rmse"],
            "rel_val_error": rel, "physical_bounds": bounds,
            "gate_passed": bool(rel <= p7["gate_rel_val_error"] and bounds["passed"]),
            "complexity_score": equation_complexity(h)["complexity_score"],
            "bic": n_train * math.log(max(fit["val_rmse"] ** 2, 1e-300)) + k * math.log(n_train),
            "contributions": contrib,
            "inactive_parameters": [q for q, v in contrib.items() if v < p6["inactive_threshold"]],
            "signature": str(sorted(term_set(h))),
            "reduced_signature": _reduced_term_set(h, [q for q, v in contrib.items() if v < p6["inactive_threshold"]]),
        })
    by_id = {h.id: h for h in survivors}
    if apply_validation:
        sel = select_model(rows, p6["equivalence_tol"], p6["inactive_threshold"], prefer_reduced=prefer_reduced)
        selected = sel["selected"]
    else:
        from hypotheses.ranking import rank_hypotheses
        from hypotheses.validation import validate_hypothesis
        ranking = rank_hypotheses(survivors, {h.id: validate_hypothesis(h) for h in survivors})
        selected = ranking[0]["id"] if ranking else None
        sel = {"selected": selected, "rule": "Phase 4 ranking top-1 (no physics validation)"}
    return {"rows": rows, "selection": sel, "selected": selected,
            "selected_hypothesis": by_id.get(selected), "filter_report": filt,
            "n_pool": len(pool), "n_structurally_valid": len(survivors)}


def predict_ood(hyp: Optional[Hypothesis], params: Dict, ds, t_end: float):
    """Solve the selected law forward from the known IC (no observations used) to t_end."""
    if hyp is None or not params:
        return None
    try:
        return solve_forward(hyp, params, ds.ic, ds.x, t_end, n_out=int(round(t_end * 100)) + 1, method="rk4")
    except Exception:
        return None


# ------------------------------------------------------------------ scoring (ground truth used ONLY here)
FAMILY_TERMS = {
    "diffusion": frozenset({"u_xx"}), "advection": frozenset({"u_x"}),
    "reaction": frozenset({"u", "u**2"}), "reaction-diffusion": frozenset({"u_xx", "u", "u**2"}),
    "advection-diffusion": frozenset({"u_x", "u_xx"}),
}


def family_of(hyp: Optional[Hypothesis]) -> Optional[str]:
    if hyp is None:
        return None
    ts = term_set(hyp)
    for fam, terms in FAMILY_TERMS.items():
        if ts == terms:
            return fam
    return "other"


def truth_field(ds, t_eval: np.ndarray) -> np.ndarray:
    if ds.exact is not None:
        X, T = np.meshgrid(ds.x, t_eval)
        return ds.exact(X, T)
    h = Hypothesis(id="truth", name="truth", equation=ds.true_equation,
                   parameters=[{"name": k} for k in ds.truth])
    sol = solve_forward(h, ds.truth, ds.ic, ds.x, float(t_eval[-1]), n_out=len(t_eval), method="rk4")
    return np.array([sol["U"][np.argmin(np.abs(sol["t"] - tt))] for tt in t_eval])


def score_run(ds, run: Dict, t_end: float) -> Dict:
    hyp = run["selected_hypothesis"]
    row = next((r for r in run["rows"] if r["id"] == run["selected"]), None)
    params = row["params"] if row else {}
    fam = family_of(hyp)
    correct = fam == ds.family
    param_err = None
    if correct and params:
        param_err = max(abs(params[k] - v) / abs(v) for k, v in ds.truth.items() if k in params)
    sol = predict_ood(hyp, params, ds, t_end)
    ood = None
    if sol is not None and sol["stable"]:
        mask = sol["t"] > 1.0 + 1e-9
        if mask.any():
            truth = truth_field(ds, sol["t"][mask])
            ood = float(np.sqrt(np.mean((sol["U"][mask] - truth) ** 2)) / (np.sqrt(np.mean(truth ** 2)) or 1e-12))
    wrong_in_pool = [r for r in run["rows"] if family_of_row(r) not in (ds.family,)]
    rejected_wrong = [r for r in wrong_in_pool if not r["gate_passed"]]
    nests_truth = lambda r: FAMILY_TERMS.get(ds.family, frozenset()) < frozenset(eval(r["signature"]))
    strictly_wrong = [r for r in wrong_in_pool if not nests_truth(r)]
    return {
        "dataset": ds.name, "truth_family": ds.family, "selected": run["selected"],
        "selected_family": fam, "correct_model": bool(correct),
        "pool_contains_truth": any(family_of_row(r) == ds.family for r in run["rows"]),
        "n_pool": run["n_pool"], "n_structurally_valid": run["n_structurally_valid"],
        "selected_params": params, "max_rel_param_error": param_err,
        "selected_rel_val_error": row["rel_val_error"] if row else None,
        "selected_complexity": row["complexity_score"] if row else None,
        "ood_rel_rmse_t_1_to_end": ood,
        "n_wrong_candidates": len(strictly_wrong),
        "n_wrong_rejected": sum(1 for r in strictly_wrong if not r["gate_passed"]),
        "n_nesting_candidates_supported": sum(1 for r in wrong_in_pool if nests_truth(r) and r["gate_passed"]),
    }


def family_of_row(row: Dict) -> Optional[str]:
    ts = frozenset(eval(row["signature"]))
    for fam, terms in FAMILY_TERMS.items():
        if ts == terms:
            return fam
    return "other"
