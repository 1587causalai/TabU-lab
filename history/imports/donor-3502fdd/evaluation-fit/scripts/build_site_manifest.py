#!/usr/bin/env python3
"""Build or verify the deterministic source-to-projection hash manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "site" / "projection-manifest.json"
BINDINGS = (
    (
        "docs/blog/introducing-tabu-lab.md",
        "site/public/blog/introducing-tabu-lab/index.html",
    ),
    (
        "docs/blog/introducing-tabu-lab.zh.md",
        "site/public/zh/blog/introducing-tabu-lab/index.html",
    ),
    (
        "docs/blog/configure-tabu-model-with-yaml.md",
        "site/public/blog/configure-tabu-model-with-yaml/index.html",
    ),
    (
        "docs/blog/configure-tabu-model-with-yaml.zh.md",
        "site/public/zh/blog/configure-tabu-model-with-yaml/index.html",
    ),
    ("docs/blog/README.md", "site/public/blog/index.html"),
    ("docs/blog/README.md", "site/public/zh/blog/index.html"),
    ("EVIDENCE_LEDGER.md", "site/public/agent.json"),
    ("catalog.json", "site/public/catalog.json"),
)
STATIC_PUBLIC_FILES = (
    "site/public/index.html",
    "site/public/zh/index.html",
    "site/public/styles.css",
    "site/public/blog.css",
    "site/public/app.js",
    "site/public/assets/tabu-mark.svg",
    "site/public/research.css",
    "site/public/research-projection.json",
)


def public_files() -> tuple[str, ...]:
    generated_roots = ("models", "experiments", "runs", "evaluations", "lineage")
    generated = [
        path.relative_to(ROOT).as_posix()
        for name in generated_roots
        for path in sorted((ROOT / "site" / "public" / name).rglob("*"))
        if path.is_file()
    ]
    return tuple(sorted((*STATIC_PUBLIC_FILES, *generated)))


def digest(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def payload() -> dict[str, object]:
    return {
        "schema_version": "tabu-lab.site-projection.v1",
        "generated_by": "scripts/build_site_manifest.py",
        "bindings": [
            {
                "source": source,
                "source_sha256": digest(source),
                "projection": projection,
                "projection_sha256": digest(projection),
            }
            for source, projection in BINDINGS
        ],
        "public_files": {
            relative_path: digest(relative_path) for relative_path in public_files()
        },
    }


def serialize(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail rather than update when the checked-in manifest differs",
    )
    args = parser.parse_args()
    expected = serialize(payload())
    if args.check:
        actual = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
        if actual != expected:
            raise SystemExit(
                "site/projection-manifest.json is stale; run "
                "`python scripts/build_site_manifest.py`"
            )
        print("PASS: source/projection hash manifest is current")
        return
    MANIFEST.write_text(expected, encoding="utf-8")
    print(f"WROTE: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
