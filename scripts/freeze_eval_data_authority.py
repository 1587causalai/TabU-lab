"""Fail-closed integrity checker for private evaluation-data freeze bundles.

The checker deliberately verifies only the candidate freeze contract.  It does
not promote a dataset, issue a receipt, or make a scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "eval-data-authority-freeze.schema.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FREEZE_ID = re.compile(r"^eval-data-freeze-[0-9a-f]{64}$")
_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ROLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MAX_FILE_SIZE = 128 * 1024 * 1024


class FreezeValidationError(ValueError):
    """Raised when a candidate freeze does not satisfy its checked-in schema."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeValidationError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise FreezeValidationError(f"{label} must be a JSON object")
    return payload


def _schema_contract() -> dict[str, Any]:
    schema = _load_json(SCHEMA_PATH, label="freeze schema")
    if schema.get("$id") != "https://research.wehub.us/schemas/eval-data-authority-freeze.schema.json":
        raise FreezeValidationError("checked-in freeze schema has an unexpected $id")
    properties = schema.get("properties")
    definitions = schema.get("$defs")
    if not isinstance(properties, dict) or not isinstance(definitions, dict):
        raise FreezeValidationError("checked-in freeze schema is missing properties or $defs")
    source_properties = definitions.get("FreezeSourceRef", {}).get("properties")
    output_properties = definitions.get("FreezeOutputRef", {}).get("properties")
    if not isinstance(source_properties, dict) or not isinstance(output_properties, dict):
        raise FreezeValidationError("checked-in freeze schema is missing source/output definitions")
    return {
        "root_keys": set(properties),
        "root_required": set(schema.get("required", ())),
        "root_schema_version": properties.get("schema_version", {}).get("const"),
        "source_keys": set(source_properties),
        "source_required": set(definitions["FreezeSourceRef"].get("required", ())),
        "source_schema_version": source_properties.get("schema_version", {}).get("const"),
        "output_keys": set(output_properties),
        "output_required": set(definitions["FreezeOutputRef"].get("required", ())),
        "output_schema_version": output_properties.get("schema_version", {}).get("const"),
    }


def _reject_unknown_keys(value: dict[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise FreezeValidationError(f"{label} has unsupported keys: {unknown}")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FreezeValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_non_empty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FreezeValidationError(f"{label} must be a non-empty string")
    return value


def _require_relative_path(value: Any, *, label: str) -> str:
    path = _require_non_empty_string(value, label=label)
    if "\\" in path:
        raise FreezeValidationError(f"{label} must use POSIX separators")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise FreezeValidationError(f"{label} must be a normalized relative path")
    return path


def _require_size(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FreezeValidationError(f"{label} must be an integer")
    if not 0 < value <= _MAX_FILE_SIZE:
        raise FreezeValidationError(f"{label} must be in (0, {_MAX_FILE_SIZE}] bytes")
    return value


def _file_identity(path: Path, *, label: str) -> tuple[str, int]:
    if not path.is_file():
        raise FreezeValidationError(f"{label} does not exist as a file: {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise FreezeValidationError(f"cannot read {label}: {path}") from exc
    return digest.hexdigest(), size


def _validate_ref(
    value: Any,
    *,
    label: str,
    keys: set[str],
    required: set[str],
    schema_version: str | None,
    source: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FreezeValidationError(f"{label} must be an object")
    _reject_unknown_keys(value, keys, label=label)
    missing = sorted(required - set(value))
    if missing:
        raise FreezeValidationError(f"{label} is missing required keys: {missing}")
    if value.get("schema_version", schema_version) != schema_version:
        raise FreezeValidationError(f"{label}/schema_version is invalid")
    role = _require_non_empty_string(value.get("role"), label=f"{label}/role")
    if _ROLE.fullmatch(role) is None:
        raise FreezeValidationError(f"{label}/role is invalid")
    _require_sha256(value.get("sha256"), label=f"{label}/sha256")
    _require_size(value.get("size_bytes"), label=f"{label}/size_bytes")
    _require_non_empty_string(value.get("media_type"), label=f"{label}/media_type")
    if source:
        retained_name = _require_non_empty_string(
            value.get("retained_name"), label=f"{label}/retained_name"
        )
        if "/" in retained_name or "\\" in retained_name:
            raise FreezeValidationError(f"{label}/retained_name must be path-free")
        if value.get("provenance_status", "retained_candidate_unreviewed") != (
            "retained_candidate_unreviewed"
        ):
            raise FreezeValidationError(f"{label}/provenance_status is invalid")
    else:
        _require_relative_path(value.get("relative_path"), label=f"{label}/relative_path")
        content_sha256 = value.get("content_sha256")
        if content_sha256 is not None:
            _require_sha256(content_sha256, label=f"{label}/content_sha256")
    return value


def _locate_source(root: Path, retained_name: str) -> Path:
    candidates = [
        root / "sources" / retained_name,
        root / "source_inputs" / retained_name,
        root / "inputs" / retained_name,
        root / retained_name,
    ]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        existing = [candidate for candidate in root.rglob(retained_name) if candidate.is_file()]
    if len(existing) != 1:
        raise FreezeValidationError(
            f"source {retained_name!r} must resolve to exactly one retained file, "
            f"found {len(existing)}"
        )
    return existing[0]


def _manifest_path(root: Path, manifest_name: str) -> Path:
    requested = root / manifest_name
    if requested.is_file():
        return requested
    if manifest_name == "manifest.json":
        fallback = root / "freeze-manifest.json"
        if fallback.is_file():
            return fallback
    raise FreezeValidationError(f"freeze manifest does not exist: {requested}")


def check_freeze(output_root: Path, *, manifest_name: str = "manifest.json") -> dict[str, Any]:
    """Validate one private freeze root and return a non-promoting summary."""

    root = output_root.expanduser().resolve()
    if not root.is_dir():
        raise FreezeValidationError(f"freeze output root is not a directory: {root}")
    contract = _schema_contract()
    manifest_path = _manifest_path(root, manifest_name)
    manifest = _load_json(manifest_path, label="freeze manifest")
    _reject_unknown_keys(manifest, contract["root_keys"], label="freeze manifest")
    missing = sorted(contract["root_required"] - set(manifest))
    if missing:
        raise FreezeValidationError(f"freeze manifest is missing required keys: {missing}")
    if manifest.get("schema_version", contract["root_schema_version"]) != contract[
        "root_schema_version"
    ]:
        raise FreezeValidationError("freeze manifest/schema_version is invalid")
    if manifest.get("authority_status", "self_consistent_unreviewed") != (
        "self_consistent_unreviewed"
    ):
        raise FreezeValidationError("freeze manifest/authority_status must remain unreviewed")
    if manifest.get("network_access", False) is not False:
        raise FreezeValidationError("freeze manifest/network_access must be false")
    if manifest.get("publication_eligible", False) is not False:
        raise FreezeValidationError("freeze manifest/publication_eligible must be false")
    for field in ("blockers", "review_ids"):
        values = manifest.get(field)
        if values is not None and (
            not isinstance(values, list) or any(not isinstance(item, str) for item in values)
        ):
            raise FreezeValidationError(f"freeze manifest/{field} must be a string list")
    if manifest.get("exporter_sha256") is not None:
        _require_sha256(manifest["exporter_sha256"], label="exporter_sha256")
    freeze_id = _require_non_empty_string(manifest.get("freeze_id"), label="freeze_id")
    if _FREEZE_ID.fullmatch(freeze_id) is None:
        raise FreezeValidationError("freeze_id is invalid")
    dataset_id = _require_non_empty_string(manifest.get("dataset_id"), label="dataset_id")
    if _DATASET_ID.fullmatch(dataset_id) is None:
        raise FreezeValidationError("dataset_id is invalid")
    _require_non_empty_string(manifest.get("source_version"), label="source_version")
    decisions = manifest.get("decisions")
    if not isinstance(decisions, dict):
        raise FreezeValidationError("decisions must be an object")

    sources = manifest.get("source_inputs")
    outputs = manifest.get("outputs")
    if not isinstance(sources, list) or not sources:
        raise FreezeValidationError("source_inputs must be a non-empty list")
    if not isinstance(outputs, list) or len(outputs) < 2:
        raise FreezeValidationError("outputs must contain at least two files")

    source_paths: list[Path] = []
    source_names: set[str] = set()
    for index, item in enumerate(sources):
        value = _validate_ref(
            item,
            label=f"source_inputs/{index}",
            keys=contract["source_keys"],
            required=contract["source_required"],
            schema_version=contract["source_schema_version"],
            source=True,
        )
        retained_name = str(value["retained_name"])
        if retained_name in source_names:
            raise FreezeValidationError(f"duplicate retained source name: {retained_name}")
        source_names.add(retained_name)
        path = _locate_source(root, retained_name)
        path = path.resolve()
        if root not in path.parents:
            raise FreezeValidationError(f"source_inputs/{index} escapes freeze root: {path}")
        digest, size = _file_identity(path, label=f"source_inputs/{index}")
        if digest != value["sha256"] or size != value["size_bytes"]:
            raise FreezeValidationError(f"source_inputs/{index} hash or size mismatch: {path}")
        source_paths.append(path)

    output_paths: list[Path] = []
    output_names: set[str] = set()
    for index, item in enumerate(outputs):
        value = _validate_ref(
            item,
            label=f"outputs/{index}",
            keys=contract["output_keys"],
            required=contract["output_required"],
            schema_version=contract["output_schema_version"],
            source=False,
        )
        relative_path = str(value["relative_path"])
        if relative_path in output_names:
            raise FreezeValidationError(f"duplicate output path: {relative_path}")
        output_names.add(relative_path)
        path = (root / relative_path).resolve()
        if root not in path.parents:
            raise FreezeValidationError(f"outputs/{index}/relative_path escapes freeze root")
        digest, size = _file_identity(path, label=f"outputs/{index}")
        if digest != value["sha256"] or size != value["size_bytes"]:
            raise FreezeValidationError(f"outputs/{index} hash or size mismatch: {path}")
        output_paths.append(path)

    overlap = set(source_paths) & set(output_paths)
    if overlap:
        raise FreezeValidationError(f"source and output refs overlap: {sorted(map(str, overlap))}")
    return {
        "manifest_path": str(manifest_path),
        "freeze_id": freeze_id,
        "dataset_id": dataset_id,
        "source_count": len(source_paths),
        "output_count": len(output_paths),
        "authority_status": "self_consistent_unreviewed",
        "publication_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="check one private freeze bundle")
    check.add_argument("--output-root", type=Path, required=True)
    check.add_argument("--manifest-name", default="manifest.json")
    args = parser.parse_args(argv)
    try:
        result = check_freeze(args.output_root, manifest_name=args.manifest_name)
    except FreezeValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: freeze integrity; "
        f"dataset={result['dataset_id']} sources={result['source_count']} "
        f"outputs={result['output_count']} authority={result['authority_status']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the review command
    raise SystemExit(main())
