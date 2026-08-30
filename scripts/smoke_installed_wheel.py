#!/usr/bin/env python3
"""Verify that the installed wheel carries and builds TabUBase."""

from __future__ import annotations

import json
from importlib import resources

from tabu_lab.models import TabUCellBaseModel
from tabu_lab.registry import BuildStatus, build_model, get_model_spec


def main() -> None:
    spec = get_model_spec("tabu.cell.base", "0.2.0")
    result = build_model("tabu.cell.base", profile="completion.artificial_mask.v1")
    if result.status is not BuildStatus.READY or not result.executable:
        raise SystemExit(f"TabUBase wheel build failed: {result.status.value}: {result.detail}")
    if not isinstance(result.model, TabUCellBaseModel):
        raise SystemExit("registry returned the wrong executable type")
    if result.model.variant_ref.contract_version != spec.contract_version:
        raise SystemExit("runtime identity is not bound to the packaged ModelSpec")
    source_manifest = json.loads(
        resources.files("tabu_lab.specs")
        .joinpath("model-factory-source-manifest.json")
        .read_text(encoding="utf-8")
    )
    source = source_manifest["contracts"]["tabu.cell.base"]
    if source["entrypoint_sha256"] != spec.upstream.sha256:
        raise SystemExit("packaged source manifest does not match the packaged ModelSpec")
    if source["semantic_source_tree_sha256"] != spec.upstream.semantic_source_tree_sha256:
        raise SystemExit("packaged source tree identity does not match the packaged ModelSpec")
    print("PASS: installed wheel built tabu.cell.base@0.2.0 from its packaged ModelSpec")


if __name__ == "__main__":
    main()
