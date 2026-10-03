"""
Phase 4: end-to-end hypothesis generation pipeline.

    scientific_problem
          |
    llm/prompts.py (problem representation -> LLM prompt)
          |
    LLMProvider.generate()  (mock or real, chosen by llm/get_provider)
          |
    JSON parsing + llm/schemas.py Pydantic validation  <-- prose that
          |                                                  doesn't match
          |                                                  the schema is
          |                                                  DROPPED here
    hypotheses/validation.py deterministic checks
          |
    hypotheses/validation.py deduplication
          |
    hypotheses/ranking.py preliminary triage ranking
          |
    candidate hypothesis set (ready for hypotheses/adapter.py -> Phase 3)

Nothing here decides scientific truth -- see llm/schemas.py and
hypotheses/ranking.py docstrings.
"""

import json
from typing import Dict, List, Optional

from pydantic import ValidationError

from llm.base import LLMProvider
from llm.prompts import SYSTEM_PROMPT, build_hypothesis_prompt, build_hypothesis_prompt_with_context
from llm.schemas import Hypothesis
from hypotheses.ranking import rank_hypotheses
from hypotheses.validation import deduplicate, validate_hypothesis


def _extract_json_array(raw: str) -> List[Dict]:
    """Strip markdown code fences if the model added them despite
    instructions not to, then json.loads. Raises json.JSONDecodeError
    (uncaught) if the result still isn't valid JSON -- the caller
    decides how to handle a totally malformed response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def generate_hypotheses(
    problem_description: str,
    provider: LLMProvider,
    n_hypotheses: int = 4,
    domain_hint: str = "",
    temperature: float = 0.7,
    context: Optional[Dict] = None,
) -> Dict:
    """
    Phase 5 adds the optional `context` parameter (Mode B, evidence-
    grounded): a plain dict, typically `ScientificContext.model_dump()`
    from rag/context.py, though hypotheses/pipeline.py has no import
    dependency on rag/ -- any dict with the expected keys works, and
    Phase 4 remains fully usable with context=None (Mode A,
    evidence-free), exactly as before Phase 5.
    """
    if context is not None:
        prompt = build_hypothesis_prompt_with_context(problem_description, context, n_hypotheses, domain_hint)
    else:
        prompt = build_hypothesis_prompt(problem_description, n_hypotheses, domain_hint)
    raw_response = provider.generate(prompt, system=SYSTEM_PROMPT, temperature=temperature)

    schema_errors = []
    try:
        raw_items = _extract_json_array(raw_response)
    except json.JSONDecodeError as e:
        # Total failure to parse ANY JSON -- return an empty candidate set
        # rather than letting raw prose leak downstream.
        return {
            "provider": provider.name,
            "mode": "evidence-grounded" if context is not None else "evidence-free",
            "problem_description": problem_description,
            "raw_response": raw_response,
            "schema_errors": [{"index": None, "error": f"Response was not valid JSON: {e}"}],
            "all_hypotheses": [],
            "validations": {},
            "valid_hypotheses": [],
            "invalid_hypotheses": [],
            "removed_duplicates": [],
            "ranking": [],
        }

    hypotheses: List[Hypothesis] = []
    for i, item in enumerate(raw_items):
        try:
            hypotheses.append(Hypothesis(**item))
        except ValidationError as e:
            schema_errors.append({"index": i, "error": str(e), "raw": item})

    validations = {h.id: validate_hypothesis(h) for h in hypotheses}
    schema_valid_and_parseable = [h for h in hypotheses if validations[h.id]["valid"]]
    invalid_hypotheses = [h for h in hypotheses if not validations[h.id]["valid"]]

    unique_hypotheses, removed_duplicates = deduplicate(schema_valid_and_parseable)
    ranking = rank_hypotheses(unique_hypotheses, validations)

    return {
        "provider": provider.name,
        "mode": "evidence-grounded" if context is not None else "evidence-free",
        "problem_description": problem_description,
        "raw_response": raw_response,
        "schema_errors": schema_errors,
        "all_hypotheses": hypotheses,
        "validations": validations,
        "valid_hypotheses": unique_hypotheses,
        "invalid_hypotheses": invalid_hypotheses,
        "removed_duplicates": removed_duplicates,
        "ranking": ranking,
    }
