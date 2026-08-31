from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tabu_lab.catalog import (
    CatalogBuildError,
    CatalogEntry,
    CatalogIndex,
    build_catalog,
    render_catalog_html,
    render_catalog_json,
)
from tabu_lab.contracts import canonical_hash

ROOT = Path(__file__).resolve().parents[2]


def test_current_catalog_indexes_consolidated_model_sources() -> None:
    catalog = build_catalog(ROOT)

    assert [entry.entry_id for entry in catalog.entries] == [
        "model_contract:tabu.cell.base@0.2.0",
        "model_contract:tabu.query.base@0.1.0",
        "model_contract:tabu.query.column@0.1.0",
        "model_contract:tabu.query.row@0.2.0",
        "model_contract:tabu.query.row_column@0.1.0",
    ]
    assert catalog.formal_receipt_count == 0
    assert catalog.accepted_claim_count == 0


def test_catalog_json_and_html_are_deterministic_bounded_projections() -> None:
    catalog = build_catalog(ROOT)

    assert render_catalog_json(catalog) == render_catalog_json(catalog)
    html = render_catalog_html(catalog)
    assert "catalog projection; not evidence or claim acceptance" in html
    assert "Formal receipts: 0;\naccepted claims: 0" in html
    assert "tabu.cell.base@0.2.0" in html
    assert "tabu.query.base@0.1.0" in html
    assert "tabu.query.row@0.2.0" in html
    assert "tabu.query.column@0.1.0" in html
    assert "tabu.query.row_column@0.1.0" in html


def test_catalog_rejects_tampered_source_tree_hash() -> None:
    payload = build_catalog(ROOT).model_dump(mode="python")
    payload["source_tree_hash"] = "0" * 64

    with pytest.raises(ValidationError, match="does not match entries"):
        CatalogIndex.model_validate(payload)


def test_catalog_rejects_wrapper_payload_identity_mismatch() -> None:
    payload = build_catalog(ROOT).entries[0].model_dump(mode="python")
    payload["object_id"] = "tabu.cell.other"

    with pytest.raises(ValidationError, match="wrapper identity"):
        CatalogEntry.model_validate(payload)


def test_catalog_fails_closed_when_no_modelspec_sources_exist(tmp_path: Path) -> None:
    (tmp_path / "specs" / "models").mkdir(parents=True)
    (tmp_path / "src" / "tabu_lab" / "specs" / "models").mkdir(parents=True)

    with pytest.raises(CatalogBuildError, match="no canonical ModelSpec sources"):
        build_catalog(tmp_path)


def test_catalog_index_rejects_empty_entries() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        CatalogIndex.model_validate(
            {
                "entries": [],
                "source_tree_hash": canonical_hash(()),
            }
        )


def test_catalog_requires_public_and_packaged_modelspec_parity(tmp_path: Path) -> None:
    public = tmp_path / "specs" / "models"
    packaged = tmp_path / "src" / "tabu_lab" / "specs" / "models"
    public.mkdir(parents=True)
    packaged.mkdir(parents=True)
    source = ROOT / "specs" / "models" / "tabu.cell.base.yaml"
    public.joinpath(source.name).write_bytes(source.read_bytes())
    packaged.joinpath(source.name).write_text("contract_id: changed\n", encoding="utf-8")

    with pytest.raises(CatalogBuildError, match="sources differ"):
        build_catalog(tmp_path)


def test_checked_catalog_projections_match_current_sources() -> None:
    catalog = build_catalog(ROOT)

    assert (ROOT / "catalog.json").read_text(encoding="utf-8") == render_catalog_json(catalog)
    assert (ROOT / "site/public/catalog.json").read_text(encoding="utf-8") == render_catalog_json(
        catalog
    )
    assert (ROOT / "site/public/models/index.html").read_text(
        encoding="utf-8"
    ) == render_catalog_html(catalog)
