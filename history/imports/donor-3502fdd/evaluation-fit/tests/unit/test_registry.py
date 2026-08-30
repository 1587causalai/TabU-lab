from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabu_lab.registry import (
    BuildStatus,
    IssueSeverity,
    ModelNotFoundError,
    ModelVersionNotFoundError,
    RegistryValidationError,
    build_model,
    clear_registry_cache,
    get_model_spec,
    instantiate_model,
    list_model_versions,
    list_models,
    validate_model_spec,
    validate_registry,
    validate_registry_source_parity,
)

EXPECTED_IDS = {
    "tabu.cell.base",
    "tabu.cell.column",
    "tabu.cell.rec",
    "tabu.cell.row",
    "tabu.cell.row_column",
    "tabufl",
    "tabul",
    "tabuf",
    "tabu4rec",
    "tabu4graph",
    "tabu4do",
    "tabu.unit_row",
    "tabu.unit_pair",
}
BUILDABLE_IDS = EXPECTED_IDS - {
    "tabu4do",
    "tabu.cell.rec",
}


def test_registry_contains_legacy_and_table_cell_contracts_in_stable_order() -> None:
    specs = list_models()
    ids = [spec.contract_id for spec in specs]
    assert ids == sorted(EXPECTED_IDS)
    assert len(ids) == len(EXPECTED_IDS)
    assert {
        spec.maturity.stage.value
        for spec in specs
        if spec.contract_id
        not in {
            "tabu4do",
            "tabu.cell.rec",
        }
    } == {"experimental"}
    assert {
        spec.maturity.stage.value
        for spec in specs
        if spec.contract_id == "tabu.cell.rec"
    } == {"design_open"}
    assert get_model_spec("tabu4do").maturity.stage.value == "design_open"
    assert {
        spec.maturity.evidence.value
        for spec in specs
        if spec.contract_id
        not in {
            "tabu4do",
            "tabu.cell.rec",
        }
    } == {"specified"}


def test_registry_is_structurally_valid_without_owner_source_mount() -> None:
    report = validate_registry(verify_upstream=False)
    assert report.ok, report.model_dump(mode="json")
    assert report.issues == ()


def test_public_and_packaged_manifests_are_byte_identical() -> None:
    validate_registry_source_parity()


def test_tabu4rec_current_alias_and_exact_history_are_identical() -> None:
    current = get_model_spec("tabu4rec")
    exact = get_model_spec("tabu4rec", "0.2.0")

    assert exact == current
    assert list_model_versions("tabu4rec") == (exact,)


def test_missing_exact_version_is_typed_error() -> None:
    with pytest.raises(ModelVersionNotFoundError) as caught:
        get_model_spec("tabu4rec", "9.9.9")

    assert caught.value.contract_id == "tabu4rec"
    assert caught.value.contract_version == "9.9.9"
    assert caught.value.available == ("0.2.0",)


def test_source_parity_rejects_missing_or_changed_history(tmp_path: Path) -> None:
    public = tmp_path / "public"
    packaged = tmp_path / "packaged"
    (public / "tabu4rec").mkdir(parents=True)
    (packaged / "tabu4rec").mkdir(parents=True)
    payload = Path(__file__).resolve().parents[2] / "specs/models/tabu4rec.yaml"
    (public / "tabu4rec/0.2.0.yaml").write_bytes(payload.read_bytes())

    with pytest.raises(RegistryValidationError, match="path mismatch"):
        validate_registry_source_parity(public_dir=public, packaged_dir=packaged)

    (packaged / "tabu4rec/0.2.0.yaml").write_text("changed", encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="byte mismatch"):
        validate_registry_source_parity(public_dir=public, packaged_dir=packaged)


def test_packaged_history_rejects_filename_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tabu_lab.registry as registry

    current = get_model_spec("tabu4rec").model_dump(mode="json")
    (tmp_path / "tabu4rec").mkdir()
    (tmp_path / "tabu4rec.yaml").write_text(json.dumps(current), encoding="utf-8")
    (tmp_path / "tabu4rec/0.1.0.yaml").write_text(json.dumps(current), encoding="utf-8")
    monkeypatch.setattr(registry, "_model_resource_dir", lambda: tmp_path)
    clear_registry_cache()
    try:
        with pytest.raises(RegistryValidationError, match="filename version"):
            list_models()
    finally:
        clear_registry_cache()


def test_packaged_history_rejects_duplicate_contract_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tabu_lab.registry as registry

    current = get_model_spec("tabu4rec").model_dump(mode="json")
    (tmp_path / "tabu4rec").mkdir()
    (tmp_path / "tabu4rec.yaml").write_text(json.dumps(current), encoding="utf-8")
    history = json.dumps(current)
    (tmp_path / "tabu4rec/0.2.0.yaml").write_text(history, encoding="utf-8")
    (tmp_path / "tabu4rec/0.2.0.yml").write_text(history, encoding="utf-8")
    monkeypatch.setattr(registry, "_model_resource_dir", lambda: tmp_path)
    clear_registry_cache()
    try:
        with pytest.raises(RegistryValidationError, match="duplicate historical"):
            list_models()
    finally:
        clear_registry_cache()


def test_exact_lookup_keeps_history_when_current_alias_advances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tabu_lab.registry as registry

    old = get_model_spec("tabu4rec").model_dump(mode="json")
    current = {**old, "contract_version": "0.3.0"}
    (tmp_path / "tabu4rec").mkdir()
    (tmp_path / "tabu4rec.yaml").write_text(json.dumps(current), encoding="utf-8")
    (tmp_path / "tabu4rec/0.2.0.yaml").write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(registry, "_model_resource_dir", lambda: tmp_path)
    clear_registry_cache()
    try:
        assert get_model_spec("tabu4rec").contract_version == "0.3.0"
        assert get_model_spec("tabu4rec", "0.2.0").contract_version == "0.2.0"
        assert [spec.contract_version for spec in list_model_versions("tabu4rec")] == [
            "0.2.0",
            "0.3.0",
        ]
    finally:
        clear_registry_cache()


def test_current_alias_must_match_same_version_history_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tabu_lab.registry as registry

    current = get_model_spec("tabu4rec").model_dump(mode="json")
    (tmp_path / "tabu4rec").mkdir()
    (tmp_path / "tabu4rec.yaml").write_text(json.dumps(current), encoding="utf-8")
    (tmp_path / "tabu4rec/0.2.0.yaml").write_text(
        json.dumps(current, indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "_model_resource_dir", lambda: tmp_path)
    clear_registry_cache()
    try:
        with pytest.raises(RegistryValidationError, match="current alias and immutable history"):
            list_models()
    finally:
        clear_registry_cache()


def test_published_schema_constrains_sha256_shape() -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "schemas" / "model-spec.schema.json").read_text())
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "https://research.wehub.us/schemas/model-spec.schema.json"
    sha_schema = schema["$defs"]["UpstreamSource"]["properties"]["sha256"]
    assert sha_schema["pattern"] == "^[0-9a-f]{64}$"


def test_published_schema_matches_runtime_model() -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "schemas" / "model-spec.schema.json").read_text())

    from tabu_lab.registry import ModelSpec

    assert schema == ModelSpec.model_json_schema()


@pytest.mark.parametrize("contract_id", ["tabuf", "tabu.unit_row", "tabu.unit_pair"])
def test_frozen_v0_rejects_natural_missing_targets(contract_id: str) -> None:
    spec = get_model_spec(contract_id)
    assert "artificial_mask_completion" in spec.capabilities
    assert "natural_missing_completion" not in spec.capabilities
    assert "typed unsupported" in spec.interfaces["infer"]


def test_missing_upstream_is_warning_not_invalid(tmp_path: Path) -> None:
    report = validate_model_spec(get_model_spec("tabuf"), source_root=tmp_path)
    assert report.ok
    assert [issue.severity for issue in report.issues] == [IssueSeverity.WARNING]
    assert report.issues[0].code == "upstream_source_unavailable"


def test_present_upstream_hash_mismatch_is_invalid(tmp_path: Path) -> None:
    spec = get_model_spec("tabuf")
    source = (tmp_path / spec.upstream.path).resolve()
    source.parent.mkdir(parents=True)
    source.write_text("changed", encoding="utf-8")
    report = validate_model_spec(spec, source_root=tmp_path)
    assert not report.ok
    assert report.issues[0].code == "upstream_hash_mismatch"


@pytest.mark.parametrize("promoted", ["supported", "evidence-backed"])
def test_public_maturity_promotion_without_gate_refs_is_invalid(promoted: str) -> None:
    payload = get_model_spec("tabuf").model_dump(mode="json")
    payload["maturity"]["stage"] = promoted
    payload["maturity"]["implementation"] = promoted

    report = validate_model_spec(payload, verify_upstream=False)

    assert not report.ok
    assert report.issues[0].code == "invalid_model_spec"
    assert "maturity requires" in report.issues[0].message


def test_evidence_backed_promotion_requires_accepted_claim_hash() -> None:
    payload = get_model_spec("tabuf").model_dump(mode="json")
    payload["maturity"]["stage"] = "evidence-backed"
    payload["maturity"]["implementation"] = "supported"
    payload["maturity"]["evidence"] = "evidence-backed"
    payload["maturity_evidence"] = {
        "gate1_receipt_hash": "1" * 64,
        "independent_review_report_hash": "2" * 64,
        "gong_approval_hash": "3" * 64,
    }

    missing_claim = validate_model_spec(payload, verify_upstream=False)
    assert not missing_claim.ok
    assert "accepted claim hash" in missing_claim.issues[0].message

    payload["maturity_evidence"]["accepted_claim_hash"] = "4" * 64
    complete = validate_model_spec(payload, verify_upstream=False)
    assert complete.ok, complete.model_dump(mode="json")


def test_tabu4do_build_is_typed_design_open() -> None:
    result = build_model("tabu4do")
    assert result.status is BuildStatus.DESIGN_OPEN
    assert result.ok
    assert result.model is None
    assert result.detail
    instantiated = instantiate_model("tabu4do")
    assert instantiated.status is BuildStatus.DESIGN_OPEN
    assert instantiated.model is None


@pytest.mark.parametrize("contract_id", sorted(BUILDABLE_IDS))
def test_buildable_contracts_route_to_lazy_model_builder(contract_id: str) -> None:
    result = build_model(contract_id)
    assert result.status is BuildStatus.READY, result.detail
    assert result.ok
    assert result.model is not None


def test_unknown_contract_is_typed_error() -> None:
    try:
        get_model_spec("not-a-model")
    except ModelNotFoundError as exc:
        assert exc.contract_id == "not-a-model"
        assert set(exc.available) == EXPECTED_IDS
    else:  # pragma: no cover
        raise AssertionError("expected ModelNotFoundError")
