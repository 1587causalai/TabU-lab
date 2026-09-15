from pathlib import Path
from types import SimpleNamespace

import pytest

from tabu_lab.registry import _source_path


@pytest.mark.parametrize("name", ["TabUL", "TabU4Graph", "TabU4Rec", "TabU4Do"])
def test_retired_alias_resolves_and_existing_old_file_takes_precedence(
    tmp_path: Path, name: str
) -> None:
    factory = tmp_path / "latex/model-factory"
    generation = "table-cell-as-query-models" if name == "TabU-v2" else "first-generation-models"
    current = factory / generation / name / "main.tex"
    current.parent.mkdir(parents=True)
    current.write_text("model source")
    spec = SimpleNamespace(upstream=SimpleNamespace(path=f"../latex/model-factory/{name}/main.tex"))
    assert _source_path(spec, tmp_path / "lab") == current.resolve()
    old = factory / name / "main.tex"
    old.parent.mkdir()
    old.write_text("wrong legacy bytes must remain visible to validation")
    assert _source_path(spec, tmp_path / "lab") == old.resolve()


def test_unrelated_missing_source_is_not_redirected(tmp_path: Path) -> None:
    spec = SimpleNamespace(
        upstream=SimpleNamespace(path="../latex/model-factory/NewModel/main.tex")
    )
    assert _source_path(spec, tmp_path / "lab") == (
        tmp_path / "latex/model-factory/NewModel/main.tex"
    ).resolve()
