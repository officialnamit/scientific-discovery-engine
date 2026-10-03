"""
Phase 4: deterministic mock LLM provider.

Used for:
  - all unit tests (no API key/network required, per the Phase 4
    constraints),
  - offline experiment runs when no real provider is configured.

IMPORTANT HONESTY NOTE: this is a CANNED, deterministic set of
candidate mechanisms for 1D scalar-field transport problems. It is NOT
doing real scientific reasoning about the prompt -- it returns the
same structurally-diverse set of hypotheses (diffusion, advection,
reaction-diffusion, advection-diffusion) regardless of input, with
generic (not dataset-specific) parameter guesses. It exists to make
the rest of the pipeline (schema validation, deduplication, ranking,
the Phase 3 adapter) fully testable without external dependencies.
The real demonstration of LLM-driven reasoning requires
llm/gemini_provider.py with a real API key -- see README's Phase 4
section for how to switch.
"""

import json

from llm.base import LLMProvider

_CANNED_HYPOTHESES = [
    {
        "id": "H1",
        "name": "Diffusion",
        "domain": "transport",
        "equation": "u_t = D*u_xx",
        "latex": "u_t = D u_{xx}",
        "dependent_variables": ["u"],
        "independent_variables": ["x", "t"],
        "parameters": [
            {"name": "D", "description": "diffusion coefficient", "trainable": True,
             "initial_value": 0.2, "units": "length^2/time", "lower_bound": 0.0}
        ],
        "required_derivatives": ["u_t", "u_xx"],
        "required_initial_conditions": True,
        "required_boundary_conditions": True,
        "assumptions": ["isotropic medium", "no advective transport", "no source/sink terms"],
        "mechanism": "diffusive transport",
        "rationale": "A field that spreads out and decays smoothly over time, with no net "
                     "directional drift, is consistent with a purely diffusive (random-walk) "
                     "transport mechanism.",
        "testable_predictions": ["spatial variance grows linearly in time",
                                  "field amplitude decays monotonically with no translation"],
        "confidence": 0.55,
    },
    {
        "id": "H2",
        "name": "Advection",
        "domain": "transport",
        "equation": "u_t + c*u_x = 0",
        "latex": "u_t + c\\,u_x = 0",
        "dependent_variables": ["u"],
        "independent_variables": ["x", "t"],
        "parameters": [
            {"name": "c", "description": "advection (transport) velocity", "trainable": True,
             "initial_value": 0.2, "units": "length/time"}
        ],
        "required_derivatives": ["u_t", "u_x"],
        "required_initial_conditions": True,
        "required_boundary_conditions": True,
        "assumptions": ["no diffusive spreading", "no source/sink terms", "constant transport velocity"],
        "mechanism": "advective (directional) transport",
        "rationale": "If the observed field is primarily being carried/translated through space "
                     "rather than spreading out, a pure advection mechanism is a plausible "
                     "structurally simple alternative to diffusion.",
        "testable_predictions": ["the field's spatial profile translates rigidly with time",
                                  "peak shape is preserved rather than flattening"],
        "confidence": 0.30,
    },
    {
        "id": "H3",
        "name": "Reaction-diffusion (logistic growth/decay)",
        "domain": "transport",
        "equation": "u_t = D*u_xx + r*u*(1 - u/K)",
        "latex": "u_t = D u_{xx} + r u (1 - u/K)",
        "dependent_variables": ["u"],
        "independent_variables": ["x", "t"],
        "parameters": [
            {"name": "D", "description": "diffusion coefficient", "trainable": True,
             "initial_value": 0.2, "units": "length^2/time", "lower_bound": 0.0},
            {"name": "r", "description": "local growth/decay rate", "trainable": True,
             "initial_value": 0.1, "units": "1/time"},
            {"name": "K", "description": "carrying capacity / saturation level", "trainable": True,
             "initial_value": 1.0, "units": "same as u", "lower_bound": 0.0},
        ],
        "required_derivatives": ["u_t", "u_xx"],
        "required_initial_conditions": True,
        "required_boundary_conditions": True,
        "assumptions": ["diffusive transport plus a local nonlinear source/sink term",
                        "logistic-form self-limiting reaction kinetics"],
        "mechanism": "diffusion combined with local nonlinear reaction kinetics",
        "rationale": "If the decay rate itself appears to depend on the local field value "
                     "(not just a constant-rate exponential decay), a reaction term alongside "
                     "diffusion is a plausible richer alternative worth testing.",
        "testable_predictions": ["decay rate varies with local amplitude, not just position",
                                  "field approaches a nonzero steady state if r>0"],
        "confidence": 0.35,
    },
    {
        "id": "H4",
        "name": "Advection-diffusion",
        "domain": "transport",
        "equation": "u_t + c*u_x = D*u_xx",
        "latex": "u_t + c\\,u_x = D u_{xx}",
        "dependent_variables": ["u"],
        "independent_variables": ["x", "t"],
        "parameters": [
            {"name": "c", "description": "advection velocity", "trainable": True,
             "initial_value": 0.1, "units": "length/time"},
            {"name": "D", "description": "diffusion coefficient", "trainable": True,
             "initial_value": 0.2, "units": "length^2/time", "lower_bound": 0.0},
        ],
        "required_derivatives": ["u_t", "u_x", "u_xx"],
        "required_initial_conditions": True,
        "required_boundary_conditions": True,
        "assumptions": ["both directional transport and diffusive spreading are present"],
        "mechanism": "combined advective and diffusive transport",
        "rationale": "A general combination of the two simplest linear transport mechanisms; "
                     "worth testing in case the field both drifts and spreads.",
        "testable_predictions": ["field both translates and spreads over time"],
        "confidence": 0.25,
    },
]


class MockLLMProvider(LLMProvider):
    """Deterministic, offline. Ignores prompt content except for
    cosmetic logging; always returns the same 4 structurally distinct
    candidate mechanisms as a JSON string."""

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        return json.dumps(_CANNED_HYPOTHESES)
