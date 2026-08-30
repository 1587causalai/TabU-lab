from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from tabu_lab.experiments.preregistration import build_f0_preregistration
from tabu_lab.experiments.s1_registry import list_s1_registrations

_REPOSITORY = Path(__file__).resolve().parents[2]
_SCRIPT = runpy.run_path(str(_REPOSITORY / "scripts" / "build_fit_first_assets.py"))
_write_preregistration_once: Callable[[Path, Any], None] = _SCRIPT["_write_preregistration_once"]
_build_s1_assets: Callable[[Path], None] = _SCRIPT["build_s1_assets"]


def test_preregistration_asset_is_create_once_and_semantically_idempotent(
    tmp_path: Path,
) -> None:
    spec = build_f0_preregistration("tabuf", device="cpu")
    directory = tmp_path / spec.experiment_id
    _write_preregistration_once(directory, spec)
    path = directory / "preregistration.yaml"
    original = path.read_bytes()

    # Older YAML may omit a field whose typed default is part of the same
    # experiment.  Rebuilding validates it but must not rewrite its bytes.
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["execution"].pop("evidence_mode")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    equivalent = path.read_bytes()
    _write_preregistration_once(directory, spec)
    assert path.read_bytes() == equivalent

    changed = spec.model_copy(update={"experiment_id": f"{spec.experiment_id}-forged-in-place"})
    with pytest.raises(RuntimeError, match="refusing to rewrite immutable preregistration"):
        _write_preregistration_once(directory, changed)
    assert path.read_bytes() == equivalent
    assert path.read_bytes() != original


def test_s1_asset_builder_writes_nine_cuda_specs_without_touching_f0(tmp_path: Path) -> None:
    sentinel = tmp_path / "experiments" / "fit-first" / "F0" / "sentinel.yaml"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"f0-must-remain-byte-identical\n")
    before = sentinel.read_bytes()

    _build_s1_assets(tmp_path)

    paths = tuple(
        sorted((tmp_path / "experiments" / "fit-first" / "S1").glob("*/preregistration.yaml"))
    )
    registrations = list_s1_registrations()
    assert tuple(path.parent.name for path in paths) == tuple(
        registration.experiment_id for registration in registrations
    )
    assert len(paths) == 9
    assert sentinel.read_bytes() == before
    for path in paths:
        spec = _SCRIPT["FitExperimentSpec"].model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        assert spec.stage.value == "S1"
        assert spec.execution.device.value == "cuda"
        assert spec.execution.device_index == 0
