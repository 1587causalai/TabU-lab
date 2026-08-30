from __future__ import annotations

import pytest

from tabu_lab.mathspec import Mathematics
from tabu_lab.mathtex import render_model_tex
from tabu_lab.registry import get_model_spec


def test_tabuf_math_projection_is_structured_and_deterministic() -> None:
    spec = get_model_spec("tabuf")

    first = render_model_tex(spec)
    second = render_model_tex(spec)

    assert first == second
    assert first.count(r"\begin{equation}") == 15
    assert first.count(r"\subsection{Step ") == 5
    assert r"\subsection{truth\_isolation}" in first
    assert r"\mathcal E_{\mathrm{in}}" in first
    assert r"buildable\_contract" in first


def test_math_projection_fails_closed_without_structured_block() -> None:
    spec = get_model_spec("tabul").model_copy(update={"mathematics": None})

    with pytest.raises(ValueError, match="no structured mathematics block"):
        render_model_tex(spec)


def test_math_projection_escapes_prose_but_preserves_formula_latex() -> None:
    rendered = render_model_tex(get_model_spec("tabuf"))

    assert r"\textbackslash{}" not in rendered
    assert r"\operatorname{OAttention}" in rendered


def test_mathematics_rejects_duplicate_equation_ids() -> None:
    mathematics = get_model_spec("tabuf").mathematics
    assert mathematics is not None
    payload = mathematics.model_dump()
    payload["steps"][1]["equations"][0]["id"] = payload["steps"][0]["equations"][0]["id"]

    with pytest.raises(ValueError, match="equations ids must be unique"):
        Mathematics.model_validate(payload)
