from __future__ import annotations

import pytest

from tabu_lab.mathspec import Mathematics
from tabu_lab.mathtex import render_model_tex
from tabu_lab.registry import ModelSpec, get_model_spec


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


def test_mathematics_rejects_duplicate_equation_ids() -> None:
    payload = _mathematics().model_dump(mode="python")
    duplicate_step = dict(payload["steps"][0])
    duplicate_step["id"] = "decode"
    payload["steps"] = (*payload["steps"], duplicate_step)

    with pytest.raises(ValueError, match="equations ids must be unique"):
        Mathematics.model_validate(payload)


def test_existing_modelspec_identity_is_not_rewritten() -> None:
    spec = get_model_spec("tabu.cell.base", "0.2.0")

    assert spec.mathematics is None
