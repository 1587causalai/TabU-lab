#!/usr/bin/env python3
"""Hash readonly TabU model-factory source closures without vendoring contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (
    ROOT / "specs" / "model-factory-source-manifest.json",
    ROOT / "src" / "tabu_lab" / "specs" / "model-factory-source-manifest.json",
)
ENTRYPOINTS = {
    "tabu.cell.base": "table-cell-as-unit-models/TabUBase/main.tex",
    "tabu.query.base": "table-cell-as-query-models/TabUBase/main.tex",
    "tabu.query.row": "table-cell-as-query-models/TabUR/main.tex",
    "tabu.query.column": "table-cell-as-query-models/TabUC/main.tex",
    "tabu.query.row_column": "table-cell-as-query-models/TabURC/main.tex",
}
INPUT_PATTERN = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHIC_PATTERN = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")


def resolve_factory() -> Path:
    direct = ROOT.parent / "latex" / "model-factory"
    if direct.is_dir():
        return direct
    common_dir = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    common_path = Path(common_dir)
    common = (ROOT / common_path).resolve() if not common_path.is_absolute() else common_path
    worktree_sibling = common.resolve().parent.parent / "latex" / "model-factory"
    return worktree_sibling


FACTORY = resolve_factory()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logical(path: Path) -> str:
    return path.resolve().relative_to(FACTORY.resolve()).as_posix()


def resolve_reference(parent: Path, reference: str, *, tex: bool) -> Path:
    candidate = parent / reference
    if tex and not candidate.suffix:
        candidate = candidate.with_suffix(".tex")
    if candidate.is_file():
        return candidate
    if not tex and not candidate.suffix:
        for suffix in (".pdf", ".png", ".jpg", ".jpeg", ".svg"):
            resolved = candidate.with_suffix(suffix)
            if resolved.is_file():
                return resolved
    raise FileNotFoundError(f"unresolved model-factory include: {candidate}")


def source_closure(entrypoint: Path) -> dict[str, str]:
    pending = [entrypoint]
    seen: set[Path] = set()
    factory = FACTORY.resolve()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        if factory not in path.parents:
            raise ValueError(f"include escapes model-factory: {path}")
        seen.add(path)
        if path.suffix != ".tex":
            continue
        text = path.read_text(encoding="utf-8")
        pending.extend(
            resolve_reference(path.parent, reference, tex=True)
            for reference in INPUT_PATTERN.findall(text)
        )
        pending.extend(
            resolve_reference(path.parent, reference, tex=False)
            for reference in GRAPHIC_PATTERN.findall(text)
        )
    return {logical(path): sha256(path) for path in sorted(seen)}


def content_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_payload() -> dict[str, object]:
    if not FACTORY.is_dir():
        raise FileNotFoundError(
            "readonly model-factory source is unavailable; the checked manifest remains usable"
        )
    contracts: dict[str, object] = {}
    for contract_id, entrypoint_name in ENTRYPOINTS.items():
        entrypoint = FACTORY / entrypoint_name
        semantic_sources = source_closure(entrypoint)
        contracts[contract_id] = {
            "entrypoint": entrypoint_name,
            "entrypoint_sha256": sha256(entrypoint),
            "semantic_source_closure": semantic_sources,
            "semantic_source_tree_sha256": content_hash(semantic_sources),
        }
    return {
        "schema_version": "tabu-lab.model-factory-source-manifest.v1",
        "observed_at": "2026-08-30",
        "scope": "TabU query-family entrypoints plus recursive TeX and graphics include closures",
        "contracts": contracts,
    }


def serialize(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--if-available",
        action="store_true",
        help="skip source rehashing when the owner workspace is not mounted",
    )
    args = parser.parse_args()
    if args.if_available and not FACTORY.is_dir():
        print("SKIP: readonly model-factory source is unavailable")
        return
    expected = serialize(build_payload())
    if args.check:
        stale = [
            path.relative_to(ROOT).as_posix()
            for path in OUTPUTS
            if not path.exists() or path.read_text(encoding="utf-8") != expected
        ]
        if stale:
            raise SystemExit(
                f"TabUBase source closure manifests are stale: {', '.join(stale)}; "
                "run `python scripts/build_model_source_manifest.py`"
            )
        print("PASS: TabUBase source closure manifest is current")
        return
    for path in OUTPUTS:
        path.write_text(expected, encoding="utf-8")
        print(f"WROTE: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
