from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tabu_lab.contracts import canonical_hash
from tabu_lab.models import MODEL_BUILDERS, build_from_spec, build_model
from tabu_lab.models.tar import TabUTARModel, TARConfig
from tabu_lab.registry import (
    BuildStatus,
    get_model_spec,
    model_spec_identity_payload,
    validate_registry_source_parity,
)
from tabu_lab.registry import (
    build_model as build_registered,
)


def test_tar_builds_through_both_registry_boundaries():
    config = TARConfig(width=16, heads=4, ff_width=32, blocks=1)
    spec = get_model_spec("tabu.tar")
    model = build_from_spec(spec, config=config, device="meta")
    assert isinstance(model, TabUTARModel)
    assert model.contract_version == spec.contract_version
    assert model.model_spec_hash == canonical_hash(model_spec_identity_payload(spec))
    result = build_registered("tabu.tar", config=config, device="meta")
    assert result.status is BuildStatus.READY
    assert isinstance(result.model, TabUTARModel)
    assert build_model("tabu.tar", device="meta").config == TARConfig()


def test_tar_builder_is_protected_and_spec_cannot_be_substituted():
    with pytest.raises(ValueError, match="canonical model builder"):
        MODEL_BUILDERS.register("tabu.tar", lambda: None, replace=True)
    spec = get_model_spec("tabu.tar")
    altered = spec.model_copy(update={"display_name": "unregistered variant"})
    with pytest.raises(ValueError, match="exactly match"):
        build_from_spec(altered, device="meta")


def test_tar_manifest_parity_and_source_binding():
    validate_registry_source_parity()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "specs/model-factory-source-manifest.json").read_text())
    entry = manifest["contracts"]["tabu.tar"]
    spec = get_model_spec("tabu.tar")
    assert spec.upstream.sha256 == entry["entrypoint_sha256"]
    assert spec.upstream.semantic_source_tree_sha256 == canonical_hash(
        entry["semantic_source_closure"]
    )
    assert spec.upstream.source_manifest.endswith("#contracts/tabu.tar")


def test_tar_cli_preserves_program_and_tabur():
    from tabu_lab.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["program", "validate"]).command == "program"
    assert parser.parse_args(["tar", "inspect"]).tar_command == "inspect"
    assert parser.parse_args([
        "tabur", "optimize", "--prereg", "x", "--output-root", "y"
    ]).command == "tabur"


def test_checked_in_tar_design_matches_the_bound_source():
    root = Path(__file__).resolve().parents[2]
    spec = get_model_spec("tabu.tar")
    source = (root / "docs/design/tar-model-design.tex").read_bytes()
    assert hashlib.sha256(source).hexdigest() == spec.upstream.sha256
    defaults = json.loads((root / "docs/design/tar-defaults.json").read_text())
    assert TARConfig.from_design_defaults(defaults) == TARConfig()
