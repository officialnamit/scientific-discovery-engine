"""
Phase 4: prompt templates.

Kept out of experiments/06_hypothesis_generation.py on purpose, per
the "Create dedicated prompts rather than embedding huge prompts
directly inside experiment scripts" constraint.

The user-facing prompt spells out the exact JSON schema fields from
llm/schemas.py.Hypothesis so the raw text response can be parsed and
schema-validated directly -- this is the mechanism that prevents
arbitrary LLM prose from bypassing validation (a response that isn't
valid JSON matching this shape is rejected by
hypotheses/pipeline.py before it ever reaches anything else).
"""

SYSTEM_PROMPT = """You are a scientific hypothesis generator assisting a physics discovery \
system. You propose PLAUSIBLE candidate governing equations for an observed scientific \
field, given only a description of the observations -- never the true governing equation.

Rules you must follow:
- Generate MULTIPLE competing, structurally DIFFERENT candidate mechanisms, not variations \
of the same equation with different coefficients.
- Do NOT assume you already know the "correct" answer. You are proposing hypotheses to be \
tested later by an independent physics-informed neural network (PINN) validator -- your \
job is breadth and scientific plausibility, not final judgment.
- Explicitly state the assumptions each hypothesis rests on.
- Distinguish established scientific facts (e.g. "diffusion is a well-known transport \
mechanism") from the hypothesis itself (i.e. "this specific field is undergoing diffusion").
- Do NOT fabricate citations, experimental evidence, or numeric results you were not given.
- Your self-reported "confidence" for each hypothesis is your own subjective plausibility \
estimate ONLY -- it is not, and must not be described as, a scientifically validated result.
- Return ONLY a JSON array of hypothesis objects. No prose before or after, no markdown \
code fences.

Each hypothesis object must have exactly these fields:
{
  "id": string (e.g. "H1"),
  "name": string,
  "domain": string,
  "equation": string (e.g. "u_t = D*u_xx" or "u_t + c*u_x = 0"),
  "latex": string,
  "dependent_variables": [string],
  "independent_variables": [string],
  "parameters": [{"name": string, "description": string, "trainable": bool,
                   "initial_value": number or null, "units": string or null}],
  "required_derivatives": [string] (e.g. ["u_t","u_xx"]),
  "required_initial_conditions": bool,
  "required_boundary_conditions": bool,
  "assumptions": [string],
  "mechanism": string,
  "rationale": string,
  "testable_predictions": [string],
  "confidence": number in [0,1] or null
}
"""


def build_hypothesis_prompt_with_context(
    problem_description: str, context: dict, n_hypotheses: int = 4, domain_hint: str = ""
) -> str:
    """
    Phase 5: evidence-grounded prompt (Mode B). `context` is a plain
    dict (NOT a specific class -- llm/ has no import dependency on
    rag/, so Phase 4 stays usable without RAG per the Phase 5
    constraint) with whatever subset of these keys is available:
    mechanisms, known_equations, variables, parameters, assumptions,
    sources.

    The retrieved material is explicitly framed as GENERAL DOCUMENTED
    BACKGROUND, not as the answer for this specific problem -- the
    model is still told to reason about which (if any) mechanism
    applies and to propose structurally diverse candidates, exactly as
    in Mode A.
    """
    domain_line = f"Scientific domain hint: {domain_hint}\n" if domain_hint else ""

    mechanisms = context.get("mechanisms") or []
    known_equations = context.get("known_equations") or []
    variables = context.get("variables") or []
    parameters = context.get("parameters") or []
    assumptions = context.get("assumptions") or []
    sources = context.get("sources") or []

    evidence_block = f"""The following is GENERAL DOCUMENTED SCIENTIFIC BACKGROUND retrieved from a \
reference corpus -- it describes well-known mechanisms IN GENERAL, not a claim about what is \
actually happening in THIS specific observed problem. You must still decide, from the \
observations themselves, which (if any) of these general mechanisms plausibly applies, and you \
are free to propose hypotheses not listed here at all.

Documented mechanisms mentioned in retrieved background material: {mechanisms}
General equation forms associated with those mechanisms (generic, no fitted parameter values): {known_equations}
Variables commonly involved: {variables}
Parameters commonly involved: {parameters}
Typical assumptions documented for these mechanisms: {assumptions}
Background sources: {sources}
"""

    return f"""{domain_line}Observed scientific problem:
{problem_description}

{evidence_block}

Generate {n_hypotheses} structurally different candidate governing equations (hypotheses) \
that could plausibly explain the OBSERVATIONS above, following the JSON schema and rules given \
in your system instructions. You may draw on the documented background material where it \
genuinely fits the observations, but do not simply restate it uncritically, and do not assume \
every documented mechanism applies just because it was retrieved. Prioritize mechanistic \
diversity over minor coefficient variations of the same equation."""


def build_hypothesis_prompt(problem_description: str, n_hypotheses: int = 4, domain_hint: str = "") -> str:
    """
    problem_description: the observational/scientific context to reason from.
    MUST describe observations, not state the governing equation (see
    Phase 4's "Current benchmark" constraint) -- this function does not
    enforce that itself (it can't know what's "the answer"), the
    experiment script is responsible for writing an honest prompt.
    """
    domain_line = f"Scientific domain hint: {domain_hint}\n" if domain_hint else ""
    return f"""{domain_line}Observed scientific problem:
{problem_description}

Generate {n_hypotheses} structurally different candidate governing equations (hypotheses) \
that could plausibly explain these observations, following the JSON schema and rules given \
in your system instructions. Prioritize mechanistic diversity (e.g. diffusive, advective, \
reactive, and combined mechanisms where relevant) over minor coefficient variations of the \
same equation."""
