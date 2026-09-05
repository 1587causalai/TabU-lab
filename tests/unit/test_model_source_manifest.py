from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def test_check_rehashes_upstream_even_when_a_registered_binding_exists(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/build_model_source_manifest.py"
    spec = importlib.util.spec_from_file_location("source_manifest_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = tmp_path / "factory"
    factory.mkdir()
    source = factory / "main.tex"
    source.write_text("original source")
    outputs = (tmp_path / "public.json", tmp_path / "packaged.json")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "FACTORY", factory)
    monkeypatch.setattr(module, "OUTPUTS", outputs)
    monkeypatch.setattr(module, "ENTRYPOINTS", {"tabu.v2.tabur": "main.tex"})
    payload = module.build_payload(refresh_existing=True)
    for output in outputs:
        output.write_text(module.serialize(payload))
    monkeypatch.setattr(sys, "argv", [str(script), "--check", "--contract", "tabu.v2.tabur"])
    module.main()
    source.write_text("changed upstream source")
    with pytest.raises(SystemExit, match="stale"):
        module.main()
    monkeypatch.setattr(sys, "argv", [str(script), "--check"])
    with pytest.raises(SystemExit, match="stale"):
        module.main()
    # A failed verification cannot silently rebind the checked-in contract.
    assert json.loads(outputs[0].read_text()) == payload
