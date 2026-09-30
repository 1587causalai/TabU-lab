"""The replay archive must reconstruct exact bytes and refuse unsafe destinations."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "experiments/history/v6-20260930/archive.py"
SPEC = importlib.util.spec_from_file_location("v6_archive", SCRIPT)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def fixture_archive(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    (root / "objects").mkdir()
    content = b"historical experiment source\n"
    digest = hashlib.sha256(content).hexdigest()
    obj = f"objects/{digest}.py"
    (root / obj).write_bytes(content)
    manifest = {"schema": "tabu.v6.archive.collection.v1", "root": "/original/experiments",
                "entries": [{"path": "run/source/model.py", "kind": "archived", "object": obj,
                             "bytes": len(content), "sha256": digest},
                            {"path": "last/source", "kind": "symlink", "target": "../next/source"},
                            {"path": "next/source", "kind": "symlink", "target": "../run/source"},
                            {"path": "run/data", "kind": "symlink", "target": "/external/data"}]}
    (root / "local.json").write_text(json.dumps(manifest))
    return root, content, obj


def test_byte_exact_materialization_and_external_locator(tmp_path):
    root, content, _ = fixture_archive(tmp_path)
    dest = tmp_path / "restored"
    result = archive.materialize(root, "local", dest)
    assert result["copied_files"] == 1
    assert result["restored_links"] == 2
    assert (dest / "next/source/model.py").read_bytes() == content
    assert (dest / "last/source/model.py").read_bytes() == content
    assert not (dest / "run/data").exists()
    receipt = json.loads((dest / "archive-materialization.json").read_text())
    assert receipt["external_artifacts"][0]["target"] == "/external/data"
    with pytest.raises(FileExistsError):
        archive.materialize(root, "local", dest)


def test_corruption_is_rejected_before_creating_destination(tmp_path):
    root, content, obj = fixture_archive(tmp_path)
    (root / obj).write_bytes(b"x" * len(content))
    with pytest.raises(ValueError, match="hash mismatch"):
        archive.materialize(root, "local", tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "objects/../../outside"])
def test_paths_cannot_escape_archive(tmp_path, relative):
    with pytest.raises(ValueError, match="unsafe"):
        archive.safe_path(tmp_path, relative)
