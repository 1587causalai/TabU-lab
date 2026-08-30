#!/usr/bin/env python3
"""Build or verify catalog-driven public research routes."""

from __future__ import annotations

import argparse
from pathlib import Path

from tabu_lab.catalog import build_catalog, load_catalog
from tabu_lab.publication import check_public_projection, write_public_projection

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    catalog_path = ROOT / "catalog.json"
    public_root = ROOT / "site" / "public"
    if args.check:
        catalog = load_catalog(catalog_path)
        rebuilt = build_catalog(ROOT, source_revision=catalog.source_revision)
        if rebuilt != catalog:
            raise SystemExit("catalog.json is stale; run `tabu-lab catalog build`")
        issues = check_public_projection(catalog, public_root)
        if issues:
            raise SystemExit("\n".join(issues))
        print("PASS: catalog and public research projection are current")
        return 0

    source_revision = (
        load_catalog(catalog_path).source_revision if catalog_path.is_file() else None
    )
    catalog = build_catalog(
        ROOT,
        output_path=catalog_path,
        source_revision=source_revision,
    )
    written = write_public_projection(catalog, public_root)
    print(f"WROTE: catalog.json and {len(written)} public research files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
