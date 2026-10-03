"""
Phase 6: physics-constrained generative search.

    candidate pool (Phase 4 LLM proposals  U  Phase 5 KG-documented equation templates)
      -> structural validation        (hypotheses/validation.py, Phase 4)
      -> physics pre-checks           (hypotheses/constraints.py)
      -> inverse PINN per candidate   (pinn/trainer.build_and_train via hypotheses/adapter.py, Phase 3/4)
      -> post-fit physical bounds     (hypotheses/constraints.py)
      -> gate on HELD-OUT NOISY OBSERVATIONS (never the clean oracle field)
      -> term necessity + complexity + BIC -> model selection (validation/model_selection.py)

No second hypothesis pipeline or PINN implementation exists here -- this
module only orchestrates existing ones. Ground truth (D_true, full.npz)
is never an input to anything in this file; the experiment script loads
it separately and only for clearly-labeled oracle evaluation.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hypotheses.adapter import hypothesis_to_pinn_config
from hypotheses.complexity import equation_complexity
from hypotheses.constraints import check_fitted_bounds, precheck_physics, resolve_bounds
from hypotheses.equation_compiler import parse_equation
from hypotheses.validation import canonical_signature, deduplicate, validate_hypothesis
from llm.schemas import Hypothesis
from pinn.data_utils import make_bc_points, make_collocation_points, make_ic_points
from pinn.network import PINN
from pinn.trainer import build_and_train
from validation.model_selection import bic, parameter_contributions, reduced_signature
from validation.physics_validation import evaluate_physics_residual


# ---------------------------------------------------------------- candidate pool
def kg_template_candidates(context: Dict, graph, param_defaults: Dict[str, float]) -> List[Hypothesis]:
    """Turn equation forms DOCUMENTED in the knowledge graph for the retrieved
    mechanisms into candidate hypotheses. These are generic textbook forms with
    provenance, never fitted values (see Phase 5 answer-leakage design)."""
    from knowledge_graph.query import equations_for_mechanism, parameters_for_mechanism

    out = []
    for i, mech in enumerate(context.get("mechanisms", []), start=1):
        params = [p.name for p in parameters_for_mechanism(graph, mech)]
        for eq in equations_for_mechanism(graph, mech):
            eq_params = [p for p in params if p in eq.name]
            _, used_fields, _, _ = parse_equation(eq.name, eq_params)
            out.append(Hypothesis(
                id=f"K{i}", name=f"KG template: {mech}", domain="from knowledge graph",
                equation=eq.name,
                parameters=[{"name": p, "trainable": True,
                             "initial_value": param_defaults.get(p, param_defaults.get("_default", 0.5))}
                            for p in eq_params],
                required_derivatives=[f for f in used_fields if f != "u"],
                mechanism=mech,
                rationale=f"Equation form documented for '{mech}' in the knowledge graph "
                          f"(evidence-derived template, not an LLM proposal).",
            ))
    return out


def build_candidate_pool(llm_hypotheses: List[Hypothesis], kg_hypotheses: List[Hypothesis]) -> Tuple[List[Hypothesis], List[Dict], Dict[str, str]]:
    origin = {h.id: "llm" for h in llm_hypotheses}
    origin.update({h.id: "kg_template" for h in kg_hypotheses})
    unique, removed = deduplicate(list(llm_hypotheses) + list(kg_hypotheses))
    return unique, removed, origin


def structural_and_physics_filter(candidates: List[Hypothesis], problem_spec: Dict,
                                  fallback_bounds: Dict) -> Tuple[List[Hypothesis], Dict]:
    report = {}
    passed = []
    for h in candidates:
        s = validate_hypothesis(h)
        p = precheck_physics(h, problem_spec, fallback_bounds) if s["valid"] else {"passed": False, "errors": ["skipped: structural failure"], "warnings": []}
        report[h.id] = {"equation": h.equation, "structural": s, "physics_precheck": p,
                        "survives": bool(s["valid"] and p["passed"])}
        if s["valid"] and p["passed"]:
            passed.append(h)
    return passed, report


# ---------------------------------------------------------------- data splits
def make_splits(obs: Dict[str, np.ndarray], t_cutoff: float, val_fraction: float, seed: int) -> Dict:
    """train / in-distribution validation (random holdout from t<=cutoff) / OOD (t>cutoff).
    OOD observations are NEVER used for training or for model selection."""
    in_mask = obs["t"] <= t_cutoff
    idx_in = np.where(in_mask)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx_in)
    n_val = int(round(val_fraction * len(idx_in)))
    val_idx, train_idx = np.sort(idx_in[:n_val]), np.sort(idx_in[n_val:])
    ood_idx = np.where(~in_mask)[0]
    pick = lambda ix: {k: obs[k][ix] for k in ("x", "t", "u")}
    return {"train": pick(train_idx), "val": pick(val_idx), "ood": pick(ood_idx), "t_cutoff": t_cutoff}


def _rmse(model, d: Dict[str, np.ndarray]) -> float:
    if len(d["u"]) == 0:
        return float("nan")
    with torch.no_grad():
        x = torch.tensor(d["x"], dtype=torch.float32).reshape(-1, 1)
        t = torch.tensor(d["t"], dtype=torch.float32).reshape(-1, 1)
        pred = model(x, t).numpy().ravel()
    return float(np.sqrt(np.mean((pred - d["u"]) ** 2)))


# ---------------------------------------------------------------- per-candidate validation
def validate_candidate(hyp: Hypothesis, splits: Dict, setup: Dict, pcfg: Dict, training_cfg: Dict,
                       gate: Dict, fallback_bounds: Dict, inactive_threshold: float,
                       model_dir: Optional[str] = None) -> Dict:
    """Inverse PINN (existing Phase 3 trainer, via Phase 4 adapter) on the TRAIN split only;
    collocation/BC/IC restricted to t <= t_cutoff so OOD prediction is genuine extrapolation."""
    x_min, x_max, t_min = setup["x_min"], setup["x_max"], setup["t_min"]
    tc = splits["t_cutoff"]
    seed = training_cfg.get("seed", 0)
    data = {k: torch.tensor(v, dtype=torch.float32).reshape(-1, 1) for k, v in splits["train"].items()}
    colloc = make_collocation_points(x_min, x_max, t_min, tc, training_cfg["n_collocation"], seed=seed)
    bc = make_bc_points(x_min, x_max, t_min, tc, training_cfg["n_bc"], seed=seed + 1)
    ic = make_ic_points(x_min, x_max, training_cfg["n_ic"], setup["ic_kind"], seed=seed + 2)

    adapted = hypothesis_to_pinn_config(hyp)
    trainer, history = build_and_train(
        adapted["hypothesis_cfg"], adapted["param_init"], adapted["trainable"],
        architecture_cfg=pcfg["architecture"], weights=pcfg["weights"],
        data=data, collocation=colloc, bc=bc, ic=ic, training_cfg=training_cfg, seed=seed,
    )
    final = history[-1]
    params = {k: float(v.item()) for k, v in trainer.params.items()}
    model = trainer.model

    physics_res = evaluate_physics_residual(model, adapted["hypothesis_cfg"]["fn"], trainer.params,
                                            x_min, x_max, t_min, tc)
    train_rmse = _rmse(model, splits["train"])
    val_rmse = _rmse(model, splits["val"])
    ood_obs_rmse = _rmse(model, splits["ood"])

    bounds = resolve_bounds(hyp, fallback_bounds)
    bound_check = check_fitted_bounds(params, bounds)
    checks = {
        "val_rmse": val_rmse <= gate["val_rmse_max"],
        "physics_residual": physics_res <= gate["physics_residual_max"],
        "bc_loss": final["bc"] <= gate["bc_loss_max"],
        "ic_loss": final["ic"] <= gate["ic_loss_max"],
        "physical_bounds": bound_check["passed"],
    }
    gate_passed = all(checks.values())

    contributions = parameter_contributions(model, hyp, params, (x_min, x_max), (t_min, tc))
    inactive = [p for p, v in contributions.items() if v < inactive_threshold]
    red_sig, red_eq = reduced_signature(hyp, inactive) if inactive else (canonical_signature(hyp), hyp.equation)

    n_train = len(splits["train"]["u"])
    k = len([p for p in hyp.parameters if p.trainable])
    result = {
        "id": hyp.id, "equation": hyp.equation, "params": params,
        "train_rmse": train_rmse, "val_rmse": val_rmse, "ood_obs_rmse": ood_obs_rmse,
        "physics_residual": physics_res, "data_loss": final["data"], "bc_loss": final["bc"], "ic_loss": final["ic"],
        "physical_bounds": bound_check, "checks": checks, "gate_passed": gate_passed,
        "failed_checks": [c for c, ok in checks.items() if not ok],
        "status": ("supported by the available observations and physics constraints" if gate_passed
                   else "rejected under the tested conditions"),
        "contributions": contributions, "inactive_parameters": inactive,
        "reduced_equation_if_inactive_terms_dropped": red_eq if inactive else None,
        "signature": canonical_signature(hyp), "reduced_signature": red_sig,
        "complexity": equation_complexity(hyp),
        "bic": bic(train_rmse ** 2, n_train, k), "n_train": n_train, "k_params": k,
        "decision_data": "held-out noisy observations only (no oracle)",
    }
    result["complexity_score"] = result["complexity"]["complexity_score"]
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, f"{hyp.id}.pt"))
    return result, model


def load_candidate_model(model_dir: str, cid: str, pcfg: Dict) -> PINN:
    m = PINN(hidden_dims=pcfg["architecture"]["hidden_dims"], activation=pcfg["architecture"]["activation"])
    m.load_state_dict(torch.load(os.path.join(model_dir, f"{cid}.pt")))
    m.eval()
    return m


# ---------------------------------------------------------------- conventional NN baseline
def train_data_only_mlp(splits: Dict, pcfg: Dict, epochs: int = 3000, lr: float = 1e-3, seed: int = 0) -> PINN:
    """Conventional neural network baseline: SAME architecture, data loss ONLY
    (no physics, no BC, no IC). Required comparison from the original brief."""
    torch.manual_seed(seed)
    m = PINN(hidden_dims=pcfg["architecture"]["hidden_dims"], activation=pcfg["architecture"]["activation"])
    x = torch.tensor(splits["train"]["x"], dtype=torch.float32).reshape(-1, 1)
    t = torch.tensor(splits["train"]["t"], dtype=torch.float32).reshape(-1, 1)
    u = torch.tensor(splits["train"]["u"], dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((m(x, t) - u) ** 2)
        loss.backward()
        opt.step()
    m.eval()
    return m


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
