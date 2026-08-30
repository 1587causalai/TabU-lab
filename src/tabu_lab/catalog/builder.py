"""Deterministic catalog construction from current canonical ModelSpecs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from tabu_lab.contracts.canonical import canonical_hash, canonical_json
from tabu_lab.registry import ModelSpec, model_spec_identity_payload

from .models import CatalogEntry, CatalogIndex, CatalogObjectKind


class CatalogBuildError(ValueError):
    """A canonical catalog source cannot be loaded or bound safely."""


def _model_sources(root: Path) -> tuple[Path, ...]:
    directory = root / "specs" / "models"
    if not directory.is_dir():
        raise CatalogBuildError("specs/models is missing")
    sources = tuple(
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in {".yaml", ".yml"}
    )
    if not sources:
        raise CatalogBuildError("specs/models contains no canonical ModelSpec sources")
    return sources


def _require_packaged_parity(root: Path, public_sources: tuple[Path, ...]) -> None:
    packaged = root / "src" / "tabu_lab" / "specs" / "models"
    if not packaged.is_dir():
        raise CatalogBuildError("packaged ModelSpec directory is missing")
    public = {path.name: path.read_bytes() for path in public_sources}
    installed = {
        path.name: path.read_bytes()
        for path in sorted(packaged.iterdir())
        if path.is_file() and path.suffix in {".yaml", ".yml"}
    }
    if public != installed:
        raise CatalogBuildError("public and packaged ModelSpec sources differ")


def build_catalog(repository: str | Path) -> CatalogIndex:
    root = Path(repository).resolve()
    sources = _model_sources(root)
    _require_packaged_parity(root, sources)
    entries: list[CatalogEntry] = []
    for path in sources:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            spec = ModelSpec.model_validate(raw)
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            raise CatalogBuildError(f"invalid ModelSpec source {path.name}: {exc}") from exc
        data = model_spec_identity_payload(spec)
        entries.append(
            CatalogEntry(
                kind=CatalogObjectKind.MODEL_CONTRACT,
                object_id=spec.contract_id,
                version=spec.contract_version,
                source_path=path.relative_to(root).as_posix(),
                object_hash=canonical_hash(data),
                data=data,
            )
        )
    ordered = tuple(sorted(entries, key=lambda entry: entry.entry_id))
    return CatalogIndex(
        entries=ordered,
        source_tree_hash=canonical_hash(
            tuple(entry.model_dump(mode="python") for entry in ordered)
        ),
    )


def render_catalog_json(catalog: CatalogIndex) -> str:
    return canonical_json(catalog.model_dump(mode="python")) + "\n"


def load_catalog(path: str | Path) -> CatalogIndex:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return CatalogIndex.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise CatalogBuildError(f"invalid catalog: {exc}") from exc


def check_catalog(repository: str | Path, path: str | Path) -> bool:
    expected = render_catalog_json(build_catalog(repository))
    try:
        return Path(path).read_text(encoding="utf-8") == expected
    except OSError:
        return False


__all__ = [
    "CatalogBuildError",
    "build_catalog",
    "check_catalog",
    "load_catalog",
    "render_catalog_json",
]
