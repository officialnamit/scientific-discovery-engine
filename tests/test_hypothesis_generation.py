"""
Phase 4 tests. All run WITHOUT any external LLM API (MockLLMProvider
only), per the "Tests must work without an external LLM API" constraint.

Run:
    python tests/test_hypothesis_generation.py
or:
    python -m pytest tests/test_hypothesis_generation.py -v
"""

import json
import os
import sys

import torch
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypotheses.adapter import hypothesis_to_pinn_config
from hypotheses.equation_compiler import EquationParseError, compile_residual, parse_equation
from hypotheses.pipeline import generate_hypotheses
from hypotheses.validation import canonical_signature, deduplicate, validate_hypothesis
from llm.mock_provider import MockLLMProvider
from llm.schemas import Hypothesis, ParameterSpec


def _make_hypothesis(**overrides) -> Hypothesis:
    defaults = dict(
        id="H1", name="Diffusion", domain="transport",
        equation="u_t = D*u_xx",
        parameters=[ParameterSpec(name="D", trainable=True, initial_value=0.2, units="length^2/time")],
        required_derivatives=["u_t", "u_xx"],
        mechanism="diffusive transport", rationale="test",
    )
    defaults.update(overrides)
    return Hypothesis(**defaults)


# 1. Hypothesis schema validation
def test_hypothesis_schema_validation():
    h = _make_hypothesis()
    assert h.id == "H1"
    assert h.equation == "u_t = D*u_xx"
    print("[PASS] test_hypothesis_schema_validation")


def test_schema_rejects_bad_confidence():
    try:
        _make_hypothesis(confidence=1.5)
        assert False, "Expected ValidationError for confidence out of [0,1]"
    except ValidationError:
        pass
    print("[PASS] test_schema_rejects_bad_confidence")


def test_schema_rejects_equation_without_equals():
    try:
        _make_hypothesis(equation="D*u_xx")  # no '='
        assert False, "Expected ValidationError for equation missing '='"
    except ValidationError:
        pass
    print("[PASS] test_schema_rejects_equation_without_equals")


# 3. Required fields
def test_required_fields_enforced():
    try:
        Hypothesis(name="Diffusion", equation="u_t = D*u_xx")  # missing 'id'
        assert False, "Expected ValidationError for missing required field 'id'"
    except ValidationError:
        pass
    print("[PASS] test_required_fields_enforced")


# 4. Equation parsing (syntax + auto derivative detection)
def test_equation_parsing_diffusion():
    residual_expr, used_fields, used_params, _ = parse_equation("u_t = D*u_xx", ["D"])
    assert set(used_fields) == {"u_t", "u_xx"}
    assert used_params == ["D"]
    print("[PASS] test_equation_parsing_diffusion")


def test_equation_parsing_advection_diffusion():
    _, used_fields, used_params, _ = parse_equation("u_t + c*u_x = D*u_xx", ["c", "D"])
    assert set(used_fields) == {"u_t", "u_x", "u_xx"}
    assert set(used_params) == {"c", "D"}
    print("[PASS] test_equation_parsing_advection_diffusion")


def test_equation_parsing_reaction_diffusion():
    _, used_fields, used_params, _ = parse_equation("u_t = D*u_xx + r*u*(1-u/K)", ["D", "r", "K"])
    assert set(used_fields) == {"u", "u_t", "u_xx"}
    assert set(used_params) == {"D", "r", "K"}
    print("[PASS] test_equation_parsing_reaction_diffusion")


def test_equation_parsing_rejects_undeclared_symbol():
    try:
        parse_equation("u_t = D*u_xx + q*u", ["D"])  # 'q' not declared
        assert False, "Expected EquationParseError for undeclared symbol 'q'"
    except EquationParseError:
        pass
    print("[PASS] test_equation_parsing_rejects_undeclared_symbol")


# 2 & 5. Variable/parameter consistency + dimensional metadata
def test_invalid_hypothesis_rejected_by_deterministic_validation():
    h = _make_hypothesis(equation="u_t = D*u_xx + q*u")  # 'q' undeclared -> unparseable in our scope
    v = validate_hypothesis(h)
    assert v["valid"] is False
    assert len(v["errors"]) > 0
    print(f"[PASS] test_invalid_hypothesis_rejected_by_deterministic_validation: {v['errors']}")


def test_derivative_mismatch_warning():
    h = _make_hypothesis(required_derivatives=["u_t"])  # equation actually needs u_t AND u_xx
    v = validate_hypothesis(h)
    assert v["valid"] is True  # mismatch is a WARNING, not a hard error
    assert any("Derivative mismatch" in w for w in v["warnings"])
    print("[PASS] test_derivative_mismatch_warning")


def test_unused_parameter_warning():
    h = _make_hypothesis(
        equation="u_t = D*u_xx",
        parameters=[
            ParameterSpec(name="D", trainable=True),
            ParameterSpec(name="k", trainable=True),  # declared but never used
        ],
    )
    v = validate_hypothesis(h)
    assert v["valid"] is True
    assert any("never appear" in w for w in v["warnings"])
    print("[PASS] test_unused_parameter_warning")


def test_dimensional_completeness_partial():
    h = _make_hypothesis(parameters=[ParameterSpec(name="D", trainable=True, units=None)])
    v = validate_hypothesis(h)
    assert v["dimensional_completeness"] == 0.0
    assert any("Dimensional metadata incomplete" in w for w in v["warnings"])
    print("[PASS] test_dimensional_completeness_partial")


# 6. Duplicate detection
def test_duplicate_removal():
    h1 = _make_hypothesis(id="H1", name="Diffusion", equation="u_t = D*u_xx",
                           parameters=[ParameterSpec(name="D", trainable=True)])
    h2 = _make_hypothesis(id="H2", name="Diffusion (renamed coefficient)", equation="u_t = k*u_xx",
                           parameters=[ParameterSpec(name="k", trainable=True)])
    h3 = _make_hypothesis(id="H3", name="Advection", equation="u_t + c*u_x = 0",
                           parameters=[ParameterSpec(name="c", trainable=True)])
    unique, removed = deduplicate([h1, h2, h3])
    assert [h.id for h in unique] == ["H1", "H3"]
    assert len(removed) == 1 and removed[0]["id"] == "H2" and removed[0]["duplicate_of"] == "H1"
    print("[PASS] test_duplicate_removal")


def test_canonical_signature_sign_invariant():
    h_a = _make_hypothesis(id="A", equation="u_t = D*u_xx")
    h_b = _make_hypothesis(id="B", equation="D*u_xx = u_t")  # same equation, sides swapped
    assert canonical_signature(h_a) == canonical_signature(h_b)
    print("[PASS] test_canonical_signature_sign_invariant")


# 7. Hypothesis-to-PINN configuration conversion
def test_hypothesis_to_pinn_config():
    h = _make_hypothesis()
    adapted = hypothesis_to_pinn_config(h)
    assert adapted["param_init"] == {"D": 0.2}
    assert adapted["trainable"] == {"D": True}
    assert adapted["hypothesis_cfg"]["equation_str"] == "u_t = D*u_xx"

    # the compiled residual_fn must behave identically to calling compile_residual directly
    direct_fn, _, _, _ = compile_residual(h.equation, ["D"])
    derivs = {"u": torch.tensor([1.0]), "u_t": torch.tensor([0.5]),
              "u_x": torch.tensor([0.1]), "u_xx": torch.tensor([-2.0]), "u_xxx": torch.tensor([0.0])}
    params = {"D": torch.tensor(0.1)}
    assert torch.allclose(adapted["hypothesis_cfg"]["fn"](derivs, params), direct_fn(derivs, params))
    print("[PASS] test_hypothesis_to_pinn_config")


# 8. Mock generation
def test_mock_provider_generates_valid_json():
    provider = MockLLMProvider()
    raw = provider.generate("anything", system="anything")
    items = json.loads(raw)
    assert isinstance(items, list) and len(items) >= 3
    ids = {item["id"] for item in items}
    assert len(ids) == len(items), "Mock provider produced duplicate IDs"
    print(f"[PASS] test_mock_provider_generates_valid_json: {len(items)} hypotheses")


# 9. End-to-end Phase 4 pipeline
def test_end_to_end_pipeline_with_mock_provider():
    provider = MockLLMProvider()
    result = generate_hypotheses("A field decreases in amplitude and smooths out over time.", provider, n_hypotheses=4)

    assert len(result["schema_errors"]) == 0
    assert len(result["valid_hypotheses"]) >= 3
    names = {h.name for h in result["valid_hypotheses"]}
    assert len(names) == len(result["valid_hypotheses"]), "Expected structurally distinct hypotheses"

    reference = _make_hypothesis(id="REF")
    ref_sig = canonical_signature(reference)
    matches = [h for h in result["valid_hypotheses"] if canonical_signature(h) == ref_sig]
    assert len(matches) == 1, "Expected exactly one candidate structurally equal to true diffusion"

    assert len(result["ranking"]) == len(result["valid_hypotheses"])
    scores = [r["composite_score"] for r in result["ranking"]]
    assert scores == sorted(scores, reverse=True), "Ranking is not sorted descending"
    print(f"[PASS] test_end_to_end_pipeline_with_mock_provider: "
          f"{len(result['valid_hypotheses'])} candidates, diffusion match = {matches[0].id}")


# 10. Backward compatibility with all Phase 1-3 tests
def test_backward_compatibility_phase1_3():
    from tests.test_equation_discovery import (
        test_clean_diffusion_recovery,
        test_generality_on_advection,
        test_library_shapes_and_labels,
        test_stlsq_recovers_known_sparse_signal,
    )
    from tests.test_pinn import (
        test_autograd_derivatives_match_analytic,
        test_network_output_shape,
        test_residual_registry,
        test_trainer_smoke,
    )

    test_library_shapes_and_labels()
    test_stlsq_recovers_known_sparse_signal()
    test_generality_on_advection()
    test_clean_diffusion_recovery()
    test_network_output_shape()
    test_autograd_derivatives_match_analytic()
    test_residual_registry()
    test_trainer_smoke()
    print("[PASS] test_backward_compatibility_phase1_3")


# 11. Gemini provider: importable/constructible offline, clear error without a key
def test_gemini_provider_offline_behaviour():
    import warnings

    import pytest

    from llm import get_provider
    from llm.gemini_provider import DEFAULT_MODEL, GeminiProvider

    saved = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "GEMINI_MODEL")}
    try:
        provider = GeminiProvider()
        assert provider.model == DEFAULT_MODEL
        assert GeminiProvider(model="custom-model").model == "custom-model"
        os.environ["GEMINI_MODEL"] = "env-model"
        assert GeminiProvider().model == "env-model"
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            provider.generate("anything")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert isinstance(get_provider({"provider": "gemini"}), MockLLMProvider)
        os.environ["GEMINI_API_KEY"] = "dummy-test-key"
        assert isinstance(get_provider({"provider": "gemini", "model": "cfg-model"}), GeminiProvider)
        assert get_provider({"provider": "gemini", "model": "cfg-model"}).model == "cfg-model"
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    print("[PASS] test_gemini_provider_offline_behaviour")


if __name__ == "__main__":
    test_hypothesis_schema_validation()
    test_schema_rejects_bad_confidence()
    test_schema_rejects_equation_without_equals()
    test_required_fields_enforced()
    test_equation_parsing_diffusion()
    test_equation_parsing_advection_diffusion()
    test_equation_parsing_reaction_diffusion()
    test_equation_parsing_rejects_undeclared_symbol()
    test_invalid_hypothesis_rejected_by_deterministic_validation()
    test_derivative_mismatch_warning()
    test_unused_parameter_warning()
    test_dimensional_completeness_partial()
    test_duplicate_removal()
    test_canonical_signature_sign_invariant()
    test_hypothesis_to_pinn_config()
    test_mock_provider_generates_valid_json()
    test_end_to_end_pipeline_with_mock_provider()
    test_backward_compatibility_phase1_3()
    test_gemini_provider_offline_behaviour()
    print("\nAll Phase 4 tests passed.")
