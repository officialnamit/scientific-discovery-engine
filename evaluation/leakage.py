"""
Phase 7: leakage audit. Static check that decision-making functions never reference
ground-truth identifiers, plus a corpus scan for hidden benchmark values. The static
check ignores docstrings (which legitimately explain the no-oracle policy).
"""

import inspect

FORBIDDEN = ["truth", "D_true", "exact", "full.npz", "u_clean", "load_clean_reference"]


def ground_truth_references(fn) -> list:
    src = inspect.getsource(fn)
    body = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
    return sorted({w for w in FORBIDDEN if w in body})


def decision_functions():
    import evaluation.engine as eng
    import hypotheses.search as srch
    import validation.model_selection as ms
    return {
        "evaluation.engine.build_pool": eng.build_pool,
        "evaluation.engine.validate_and_select": eng.validate_and_select,
        "hypotheses.search.validate_candidate": srch.validate_candidate,
        "hypotheses.search.make_splits": srch.make_splits,
        "hypotheses.search.build_candidate_pool": srch.build_candidate_pool,
        "validation.model_selection.select_model": ms.select_model,
    }


def leakage_audit(res, datasets, cfg):
    from evaluation.adversarial import find_leaks
    static = {name: ground_truth_references(fn) for name, fn in decision_functions().items()}
    truths = sorted({v for d in datasets for v in d.truth.values()} | {cfg["diffusion"]["D_true"]})
    leaks = find_leaks(res.docs, truths)
    return {
        "decision_functions_referencing_ground_truth": {k: v for k, v in static.items() if v},
        "decision_functions_checked": list(static),
        "hidden_values_checked": truths,
        "corpus_documents_containing_hidden_values": leaks,
        "phase6_and_generalization_selection_data": "held-out noisy observations only (see hypotheses/search.py)",
        "passed": (not any(static.values())) and not leaks,
    }
