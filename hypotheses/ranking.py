"""
Phase 4: preliminary hypothesis ranking.

READ THIS BEFORE USING THE OUTPUT OF THIS MODULE FOR ANYTHING:
this ranking decides INVESTIGATION ORDER (which hypotheses are worth
spending PINN training time on first) -- it is explicitly NOT a
scientific-validity judgment. A hypothesis ranked #1 here can still be
"rejected under the tested conditions" by validation/scorer.py, and a
hypothesis ranked last could, in principle, turn out to be the one
PINN validation supports. Only Phase 3's PINN validation produces an
evidence-based accept/reject status.

Score components (weights are a stated config choice, not tuned to
produce a particular winner):
  - validity_score:      1.0 if hypotheses/validation.py found no hard errors, else 0.0
  - completeness_score:  did the LLM fill in rationale/mechanism/assumptions?
  - dimensional_score:   fraction of parameters with units specified
  - simplicity_score:    fewer free parameters scores higher (Occam's-razor-style prior)
  - llm_self_confidence: the LLM's own (subjective, unverified) confidence field
"""

from typing import Dict, List

from llm.schemas import Hypothesis

DEFAULT_WEIGHTS = {
    "validity": 0.35,
    "completeness": 0.15,
    "dimensional": 0.15,
    "simplicity": 0.15,
    "llm_confidence": 0.20,
}


def _completeness_score(h: Hypothesis) -> float:
    fields_present = [bool(h.rationale.strip()), bool(h.mechanism.strip()), len(h.assumptions) > 0]
    return sum(fields_present) / len(fields_present)


def _simplicity_score(h: Hypothesis) -> float:
    return 1.0 / (1 + len(h.parameters))


def rank_hypotheses(
    hypotheses: List[Hypothesis], validations: Dict[str, Dict], weights: Dict[str, float] = None
) -> List[Dict]:
    weights = weights or DEFAULT_WEIGHTS
    scored = []
    for h in hypotheses:
        v = validations.get(h.id, {"valid": False, "dimensional_completeness": 0.0})
        components = {
            "validity_score": 1.0 if v["valid"] else 0.0,
            "completeness_score": _completeness_score(h),
            "dimensional_score": v.get("dimensional_completeness", 0.0),
            "simplicity_score": _simplicity_score(h),
            "llm_confidence": h.confidence if h.confidence is not None else 0.5,
        }
        composite = (
            weights["validity"] * components["validity_score"]
            + weights["completeness"] * components["completeness_score"]
            + weights["dimensional"] * components["dimensional_score"]
            + weights["simplicity"] * components["simplicity_score"]
            + weights["llm_confidence"] * components["llm_confidence"]
        )
        scored.append({
            "id": h.id, "name": h.name, "equation": h.equation,
            "composite_score": composite, **components,
            "note": "investigation-order triage only -- NOT a scientific validity judgment",
        })
    scored.sort(key=lambda r: r["composite_score"], reverse=True)
    return scored
