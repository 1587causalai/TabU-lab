#!/usr/bin/env python3
"""Verify or materialize byte-preserved V6 research files; never launch jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
from pathlib import Path, PurePosixPath

HOSTS = ("local", "dgx2", "dustinstudio", "gongqian-mini")
BASE = Path(__file__).resolve().parent


def safe_path(base: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts:
        raise ValueError(f"unsafe archive path: {relative}")
    result = base.joinpath(*parts.parts)
    if not result.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"archive path escapes root: {relative}")
    return result


def load_manifest(base: Path, host: str):
    if host not in HOSTS:
        raise ValueError("unknown host")
    manifest = json.loads((base / f"{host}.json").read_text())
    if manifest["schema"] != "tabu.v6.archive.collection.v1":
        raise ValueError("unknown archive schema")
    return manifest


def verify(base: Path, hosts=HOSTS):
    checked = set()
    references = 0
    for host in hosts:
        seen = set()
        for entry in load_manifest(base, host)["entries"]:
            safe_path(base, entry["path"])
            if entry["path"] in seen:
                raise ValueError(f"duplicate archive path: {entry['path']}")
            seen.add(entry["path"])
            if entry["kind"] != "archived":
                continue
            key = entry["object"]
            path = safe_path(base, key)
            if path.stat().st_size != entry["bytes"]:
                raise ValueError(f"size mismatch: {host}/{entry['path']}")
            identity = (key, entry["sha256"])
            if identity not in checked:
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"hash mismatch: {host}/{entry['path']}")
                checked.add(identity)
            references += 1
    return {"verified_objects": len(checked), "archived_file_references": references}


def materialize(base: Path, host: str, destination: Path):
    """Rebuild a host's entire collected tree in a new directory, including dependencies."""
    verify(base, (host,))
    manifest = load_manifest(base, host)
    # Refuse an existing destination, so an active experiment can never be overwritten.
    destination.mkdir(parents=True, exist_ok=False)
    archived = [e for e in manifest["entries"] if e["kind"] == "archived"]
    for entry in archived:
        target = safe_path(destination, entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(safe_path(base, entry["object"]).read_bytes())
    restored_links = []
    external = [e for e in manifest["entries"] if e["kind"] not in {"archived", "symlink"}]
    pending = [e for e in manifest["entries"] if e["kind"] == "symlink"]
    while pending:
        unresolved = []
        for entry in pending:
            original = posixpath.join(manifest["root"], entry["path"])
            target = posixpath.normpath(
                posixpath.join(posixpath.dirname(original), entry["target"])
            )
            prefix = manifest["root"].rstrip("/") + "/"
            relative = target[len(prefix):] if target.startswith(prefix) else None
            if relative and safe_path(destination, relative).exists():
                link = safe_path(destination, entry["path"])
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(os.path.relpath(safe_path(destination, relative), link.parent))
                restored_links.append(entry["path"])
                continue
            unresolved.append(entry)
        if len(unresolved) == len(pending):
            break
        pending = unresolved
    external.extend(unresolved if pending else [])
    receipt = {
        "host": host, "original_root": manifest["root"],
        "copied_files": len(archived), "restored_internal_links": restored_links,
        "external_artifacts": external,
        "warning": "Historical absolute paths are unchanged. Restore and verify external data/"
                   "checkpoints and adapt paths in a NEW run configuration before execution. "
                   "No training, deployment, observer or remote command was executed.",
    }
    (destination / "archive-materialization.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"copied_files": len(archived), "restored_links": len(restored_links),
            "external_artifacts": len(external), "destination": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    restore = sub.add_parser("materialize")
    restore.add_argument("--host", choices=HOSTS, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = (verify(BASE) if args.command == "verify"
              else materialize(BASE, args.host, args.destination))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
