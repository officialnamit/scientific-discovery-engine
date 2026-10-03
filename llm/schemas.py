"""
Phase 4: structured hypothesis schema.

This is the ONLY form in which a candidate hypothesis is allowed to
leave the LLM layer. Raw LLM prose is never passed downstream --
llm/prompts.py instructs the model to return JSON matching this shape,
and hypotheses/pipeline.py rejects anything that doesn't validate
against it (see "Do not allow arbitrary LLM prose to bypass schema
validation" in the Phase 4 constraints).

`confidence` is the LLM's own self-reported plausibility estimate. It
is explicitly NOT a scientific-validity score -- see the warning on
the field itself and README's "LLM vs PINN" section. Only the PINN
validation stage (Phase 3, via validation/scorer.py) produces an
evidence-based accept/reject status.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ParameterSpec(BaseModel):
    name: str = Field(..., description="Symbol used in the equation string, e.g. 'D'.")
    description: str = ""
    trainable: bool = True
    initial_value: Optional[float] = None
    units: Optional[str] = Field(
        default=None,
        description="Best-effort physical units, e.g. 'length^2/time'. Leave null if unknown "
                    "-- do NOT guess a unit just to fill the field (see Part 5 dimensional "
                    "metadata rule: don't pretend dimensional analysis is complete).",
    )
    # Phase 6: optional physical bounds (EXCLUSIVE: value must be > lower_bound and
    # < upper_bound). None = unconstrained on that side. Declared per parameter so
    # e.g. a diffusivity can require D > 0 while an advection velocity may take
    # either sign -- never one blanket constraint for every parameter.
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class Hypothesis(BaseModel):
    id: str
    name: str
    domain: str = "unspecified"
    equation: str = Field(..., description="e.g. 'u_t = D*u_xx' or 'u_t + c*u_x = 0'")
    latex: Optional[str] = None

    dependent_variables: List[str] = Field(default_factory=lambda: ["u"])
    independent_variables: List[str] = Field(default_factory=lambda: ["x", "t"])
    parameters: List[ParameterSpec] = Field(default_factory=list)

    required_derivatives: List[str] = Field(
        default_factory=list,
        description="As STATED by the LLM. hypotheses/validation.py cross-checks this "
                    "against what the equation string actually contains (via sympy) and "
                    "flags any mismatch -- this field is evidence, not ground truth.",
    )
    required_initial_conditions: bool = True
    required_boundary_conditions: bool = True

    assumptions: List[str] = Field(default_factory=list)
    mechanism: str = ""
    rationale: str = ""
    testable_predictions: List[str] = Field(default_factory=list)

    confidence: Optional[float] = Field(
        default=None,
        description="LLM's own self-reported plausibility in [0,1]. NOT a scientific "
                    "validity score -- only PINN validation (Phase 3) produces evidence-based "
                    "accept/reject. Do not sort final scientific conclusions by this alone.",
    )

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {v}")
        return v

    @field_validator("equation")
    @classmethod
    def _equation_has_equals(cls, v):
        if "=" not in v:
            raise ValueError(f"equation must contain '=' (lhs = rhs), got: {v!r}")
        return v


class HypothesisSet(BaseModel):
    """A validated, deduplicated collection of candidate hypotheses for
    one scientific problem, plus provenance metadata."""
    problem_description: str
    provider: str
    hypotheses: List[Hypothesis]
    raw_response: Optional[str] = Field(
        default=None, description="Raw provider output, kept for audit/debugging only."
    )
