from __future__ import annotations

import pytest

from tabu_lab.contracts import canonical_hash
from tabu_lab.mathspec import Mathematics
from tabu_lab.mathtex import render_model_tex
from tabu_lab.registry import ModelSpec, get_model_spec, model_spec_identity_payload


def _mathematics() -> Mathematics:
    return Mathematics.model_validate(
        {
            "schema_version": "1.0.0",
            "abstract": "A deterministic projection of the declared mathematics.",
            "unit_semantics": "Each carrier state represents one declared table cell.",
            "notation": [
                {
                    "id": "unit_state",
                    "symbol": "h_u",
                    "meaning": "state for Unit u",
                    "domain": r"\mathbb{R}^d",
                }
            ],
            "steps": [
                {
                    "id": "encode",
                    "title": "Encode Units",
                    "purpose": "Preserve source_truth % and role metadata.",
                    "equations": [
                        {
                            "id": "encode_unit",
                            "latex": r"h_u = \operatorname{OAttention}(x_u)",
                            "meaning": "Construct a Unit state without exposing target truth.",
                        }
                    ],
                    "invariants": ["Target truth is not a forward input."],
                }
            ],
            "invariants": [
                {
                    "id": "truth_isolation",
                    "statement": "Truth remains in the host-side sidecar.",
                    "evidence": "compiler contract test",
                }
            ],
        }
    )


def _spec_with_mathematics() -> ModelSpec:
    payload = get_model_spec("tabu.cell.base").model_dump(mode="python")
    payload["mathematics"] = _mathematics().model_dump(mode="python")
    return ModelSpec.model_validate(payload)


def test_math_projection_is_structured_and_deterministic() -> None:
    spec = _spec_with_mathematics()

    first = render_model_tex(spec)
    second = render_model_tex(spec)

    assert first == second
    assert first.count(r"\begin{equation}") == 1
    assert first.count(r"\subsection{Step ") == 1
    assert r"\subsection{truth\_isolation}" in first
    assert r"\operatorname{OAttention}" in first


def test_math_projection_fails_closed_without_structured_block() -> None:
    with pytest.raises(ValueError, match="no structured mathematics block"):
        render_model_tex(get_model_spec("tabu.cell.base"))


def test_math_projection_escapes_prose_but_preserves_formula_latex() -> None:
    rendered = render_model_tex(_spec_with_mathematics())

    assert r"source\_truth \%" in rendered
    assert r"\operatorname{OAttention}" in rendered
    assert r"\(\mathbb{R}^d\)" in rendered
    assert r"\textbackslash\{\}mathbb" not in rendered


def test_math_projection_uses_injective_equation_labels() -> None:
    payload = _mathematics().model_dump(mode="python")
    first = dict(payload["steps"][0])
    first_equation = dict(first["equations"][0])
    first_equation["id"] = "a_b"
    first["equations"] = (first_equation,)
    second = dict(first)
    second["id"] = "decode"
    second_equation = dict(first_equation)
    second_equation["id"] = "a-b"
    second["equations"] = (second_equation,)
    payload["steps"] = (first, second)

    spec_payload = get_model_spec("tabu.cell.base").model_dump(mode="python")
    spec_payload["mathematics"] = payload
    rendered = render_model_tex(ModelSpec.model_validate(spec_payload))

    assert r"\label{eq:id-615f62}" in rendered
    assert r"\label{eq:id-612d62}" in rendered


def test_mathematics_rejects_duplicate_equation_ids() -> None:
    payload = _mathematics().model_dump(mode="python")
    duplicate_step = dict(payload["steps"][0])
    duplicate_step["id"] = "decode"
    payload["steps"] = (*payload["steps"], duplicate_step)

    with pytest.raises(ValueError, match="equations ids must be unique"):
        Mathematics.model_validate(payload)


def test_existing_modelspec_identity_is_not_rewritten() -> None:
    spec = get_model_spec("tabu.cell.base", "0.2.0")
    legacy_payload = spec.model_dump(mode="json")
    legacy_payload.pop("mathematics")

    assert spec.mathematics is None
    assert model_spec_identity_payload(spec) == legacy_payload
    assert canonical_hash(model_spec_identity_payload(spec)) == canonical_hash(legacy_payload)


def test_populated_mathematics_is_identity_bound() -> None:
    spec = _spec_with_mathematics()

    assert model_spec_identity_payload(spec)["mathematics"] is not None
    assert canonical_hash(model_spec_identity_payload(spec)) != canonical_hash(
        model_spec_identity_payload(get_model_spec("tabu.cell.base"))
    )
