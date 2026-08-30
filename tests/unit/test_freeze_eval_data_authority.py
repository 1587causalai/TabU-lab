from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "freeze_eval_data_authority.py"
_SPEC = importlib.util.spec_from_file_location("freeze_eval_data_authority", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
FreezeValidationError = _MODULE.FreezeValidationError
check_freeze = _MODULE.check_freeze


def _ref(path: Path, *, role: str, relative_path: str | None = None) -> dict[str, object]:
    data = path.read_bytes()
    result: dict[str, object] = {
        "role": role,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "media_type": "application/octet-stream",
    }
    if relative_path is None:
        result["retained_name"] = path.name
    else:
        result["relative_path"] = relative_path
    return result


def _write_bundle(root: Path) -> Path:
    sources = root / "sources"
    outputs = root / "outputs"
    sources.mkdir(parents=True)
    outputs.mkdir()
    source = sources / "source.bin"
    output_a = outputs / "features.bin"
    output_b = outputs / "labels.bin"
    source.write_bytes(b"source")
    output_a.write_bytes(b"features")
    output_b.write_bytes(b"labels")
    manifest = {
        "schema_version": "tabu.eval-data-authority-freeze.v1",
        "freeze_id": "eval-data-freeze-" + "a" * 64,
        "dataset_id": "fixture-dataset",
        "source_version": "fixture-v1",
        "network_access": False,
        "publication_eligible": False,
        "source_inputs": [
            _ref(source, role="raw-source"),
        ],
        "decisions": {"fixture": True},
        "outputs": [
            _ref(output_a, role="features", relative_path="outputs/features.bin"),
            _ref(output_b, role="labels", relative_path="outputs/labels.bin"),
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_check_freeze_validates_schema_bound_file_identity(tmp_path: Path) -> None:
    _write_bundle(tmp_path)

    result = check_freeze(tmp_path)

    assert result["dataset_id"] == "fixture-dataset"
    assert result["source_count"] == 1
    assert result["output_count"] == 2
    assert result["publication_eligible"] is False


def test_check_freeze_rejects_changed_retained_bytes(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "sources" / "source.bin").write_bytes(b"changed")

    with pytest.raises(FreezeValidationError, match="source_inputs/0 hash or size mismatch"):
        check_freeze(tmp_path)


def test_check_freeze_rejects_unknown_schema_keys(tmp_path: Path) -> None:
    manifest_path = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unbound"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FreezeValidationError, match="unsupported keys"):
        check_freeze(tmp_path)


def test_check_freeze_cli_matches_review_packet_command(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    script = Path(__file__).parents[2] / "scripts" / "freeze_eval_data_authority.py"

    result = subprocess.run(
        [sys.executable, str(script), "check", "--output-root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PASS: freeze integrity" in result.stdout
