from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tabu_lab.catalog import (
    CatalogEntry,
    CatalogIndex,
    CatalogObjectKind,
    CatalogSourceRevision,
)
from tabu_lab.contracts.canonical import canonical_hash
from tabu_lab.publication import (
    PUBLIC_MARKER,
    check_public_projection,
    render_public_projection,
    write_public_projection,
)


def _catalog(*, exact_revision: bool = False) -> CatalogIndex:
    data = {
        "schema_version": "tabu.test-model.v1",
        "contract_id": "tabuf",
        "maturity": "specified",
    }
    entry = CatalogEntry(
        kind=CatalogObjectKind.MODEL_CONTRACT,
        object_id="tabuf",
        object_schema_version="tabu.test-model.v1",
        object_hash=canonical_hash(data),
        source_hash="1" * 64,
        source_path="specs/models/tabuf.yaml",
        status="specified",
        data=data,
    )
    source_tree_hash = canonical_hash(
        {
            "schema": "tabu.catalog-source-tree.v1",
            "sources": [
                {
                    "kind": entry.kind.value,
                    "object_id": entry.object_id,
                    "source_hash": entry.source_hash,
                    "source_path": entry.source_path,
                }
            ],
            "lineage": [],
        }
    )
    source_revision = (
        CatalogSourceRevision(
            repository_uri="https://github.com/1587causalai/TabU-lab",
            commit="a" * 40,
            catalog_source_tree_hash=source_tree_hash,
        )
        if exact_revision
        else None
    )
    return CatalogIndex(
        source_tree_hash=source_tree_hash,
        source_revision=source_revision,
        entries=(entry,),
    )


def test_public_projection_is_deterministic_and_catalog_driven(tmp_path) -> None:
    catalog = _catalog()
    first = render_public_projection(catalog)
    second = render_public_projection(catalog)
    assert first == second
    assert first["catalog.json"] == second["catalog.json"]
    model_page = first["models/tabuf/index.html"].decode()
    assert PUBLIC_MARKER in model_page
    assert catalog.show("tabuf").object_hash in model_page
    assert "catalog projection; not independent evidence" in model_page
    assert "/Users/" not in model_page
    assert "blob/main" not in model_page
    assert "Exact public revision not recorded" in model_page
    assert "<code>specs/models/tabuf.yaml</code>" in model_page
    assert "github.com/1587causalai/TabU-lab/blob/" not in model_page

    write_public_projection(catalog, tmp_path)
    assert check_public_projection(catalog, tmp_path) == ()
    (tmp_path / "models" / "tabuf" / "index.html").write_text("stale", encoding="utf-8")
    assert check_public_projection(catalog, tmp_path) == (
        "stale generated public file: models/tabuf/index.html",
    )


def test_model_page_separates_contract_from_trained_artifacts() -> None:
    files = render_public_projection(_catalog())
    page = files["models/tabuf/index.html"].decode()
    assert "Canonical record" in page
    assert "Trained artifacts" in page
    assert "No cataloged objects" in page


def test_public_source_link_uses_cataloged_exact_revision() -> None:
    catalog = _catalog(exact_revision=True)
    files = render_public_projection(catalog)
    page = files["models/tabuf/index.html"].decode()
    expected = (
        "https://github.com/1587causalai/TabU-lab/blob/"
        f"{'a' * 40}/specs/models/tabuf.yaml"
    )
    assert f'href="{expected}"' in page
    assert "blob/main" not in page
    assert f"Git commit <code>{'a' * 40}</code>" in page
    projection = json.loads(files["research-projection.json"])
    assert projection["catalog_source_revision"] == catalog.source_revision.model_dump(
        mode="json"
    )


def test_catalog_source_revision_fails_closed_on_unbound_or_mutable_identity() -> None:
    catalog = _catalog()
    with pytest.raises(ValidationError, match="does not bind source_tree_hash"):
        CatalogIndex(
            source_tree_hash=catalog.source_tree_hash,
            source_revision=CatalogSourceRevision(
                repository_uri="https://github.com/1587causalai/TabU-lab",
                commit="b" * 40,
                catalog_source_tree_hash="2" * 64,
            ),
            entries=catalog.entries,
        )
    with pytest.raises(ValidationError, match="exactly one GitHub repository"):
        CatalogSourceRevision(
            repository_uri="https://github.com/1587causalai/TabU-lab/blob/main",
            commit="b" * 40,
            catalog_source_tree_hash=catalog.source_tree_hash,
        )
