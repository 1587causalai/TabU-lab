#!/usr/bin/env python3
"""Hash the readonly model-factory source closures without vendoring their contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTORY = ROOT.parent / "latex" / "model-factory"
OUTPUT = ROOT / "specs" / "model-factory-source-manifest.json"
CONTRACTS = {
    "tabufl": "TabUFL/main.tex",
    "tabul": "TabUL/main.tex",
    "tabuf": "TabUF/main.tex",
    "tabu4rec": "TabU4Rec/main.tex",
    "tabu4graph": "TabU4Graph/main.tex",
    "tabu4do": "TabU4Do/main.tex",
    "tabu.unit_row": "TabU/unit-as-row.tex",
    "tabu.unit_pair": "TabU/unit-as-cell.tex",
    "tabu.cell.base": "table-cell-as-unit-models/TabUBase/main.tex",
    "tabu.cell.row": "table-cell-as-unit-models/TabUR/main.tex",
    "tabu.cell.column": "table-cell-as-unit-models/TabUC/main.tex",
    "tabu.cell.row_column": "table-cell-as-unit-models/TabURC/main.tex",
    "tabu.cell.rec": "table-cell-as-unit-models/TabU4Rec/main.tex",
}
INPUT_PATTERN = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHIC_PATTERN = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")


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
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        if FACTORY.resolve() not in path.parents:
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
    for contract_id, relative_entrypoint in CONTRACTS.items():
        entrypoint = FACTORY / relative_entrypoint
        semantic_sources = source_closure(entrypoint)
        folder = entrypoint.parent
        readme = folder / "README.md"
        build_config = folder / ".latexmkrc"
        context_files = {
            logical(readme): sha256(readme),
            logical(build_config): sha256(build_config),
        }
        # The nested table-cell family README carries the non-aliasing
        # boundaries between Base/R/C/RC and axis-B Rec.  Bind it into every
        # member's source context so a changed family map cannot silently
        # leave the model manifests looking current.
        family_readme = folder.parent / "README.md"
        if folder.parent.name == "table-cell-as-unit-models" and family_readme.is_file():
            context_files[logical(family_readme)] = sha256(family_readme)
        projection = entrypoint.with_suffix(".pdf")
        contracts[contract_id] = {
            "entrypoint": logical(entrypoint),
            "entrypoint_sha256": sha256(entrypoint),
            "semantic_source_closure": semantic_sources,
            "semantic_source_tree_sha256": content_hash(semantic_sources),
            "context": context_files,
            "pdf_projection": {
                "path": logical(projection),
                "sha256": sha256(projection),
            },
        }
    return {
        "schema_version": "tabu-lab.model-factory-source-manifest.v1",
        "observed_at": "2026-08-28",
        "scope": {
            "semantic": "entrypoint plus recursive TeX and graphics include closure",
            "context": "model README and build configuration; not semantic authority",
            "projection": "compiled PDF; never semantic or implementation authority",
        },
        "factory_index": {
            "path": "README.md",
            "sha256": sha256(FACTORY / "README.md"),
        },
        "scoped_git_candidate": {
            "branch": "codex/tabu-model-factory-contracts",
            "commit": "53dbd4b18c8390517a7e248a493471f34a2e1a39",
            "state": "local_review_candidate_not_merged_or_pushed",
        },
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
        actual = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if actual != expected:
            raise SystemExit(
                "specs/model-factory-source-manifest.json is stale; run "
                "`python scripts/build_model_source_manifest.py`"
            )
        print("PASS: model-factory source closure manifest is current")
        return
    OUTPUT.write_text(expected, encoding="utf-8")
    print(f"WROTE: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
