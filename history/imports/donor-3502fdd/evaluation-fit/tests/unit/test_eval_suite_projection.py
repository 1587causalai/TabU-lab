from __future__ import annotations

import json
from pathlib import Path

from tabu_lab.evaluation.foundry.loader import default_suite_directory, list_suite_ids

ROOT = Path(__file__).resolve().parents[2]


def test_packaged_suite_projection_is_byte_identical() -> None:
    canonical = ROOT / "evaluations" / "suites"
    packaged = ROOT / "src" / "tabu_lab" / "evaluation" / "foundry" / "suites"
    canonical_files = {path.name: path.read_bytes() for path in canonical.glob("*.yaml")}
    packaged_files = {path.name: path.read_bytes() for path in packaged.glob("*.yaml")}
    assert packaged_files == canonical_files

    manifest = json.loads((packaged / "projection-manifest.json").read_text())
    assert manifest["schema_version"] == "tabu.eval-suite-package-projection.v1"
    assert set(manifest["files"]) == set(canonical_files)


def test_repository_loader_prefers_canonical_suite_sources() -> None:
    assert default_suite_directory() == ROOT / "evaluations" / "suites"
    assert list_suite_ids() == (
        "graph-completion-micro-v0",
        "recsys-completion-micro-v0",
        "table-completion-micro-v0",
        "table-completion-micro-v1",
        "table-supervised-micro-v0",
        "table-supervised-micro-v1",
    )
