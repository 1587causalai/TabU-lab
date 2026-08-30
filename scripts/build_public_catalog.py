#!/usr/bin/env python3
"""Build or verify the bounded current-source catalog projections."""

from __future__ import annotations

import argparse
from pathlib import Path

from tabu_lab.catalog import build_catalog, render_catalog_html, render_catalog_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {
    ROOT / "catalog.json": lambda catalog: render_catalog_json(catalog),
    ROOT / "site" / "public" / "catalog.json": lambda catalog: render_catalog_json(catalog),
    ROOT / "site" / "public" / "models" / "index.html": render_catalog_html,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    catalog = build_catalog(ROOT)
    stale: list[str] = []
    for path, render in OUTPUTS.items():
        expected = render(catalog)
        if arguments.check:
            actual = path.read_text(encoding="utf-8") if path.exists() else ""
            if actual != expected:
                stale.append(path.relative_to(ROOT).as_posix())
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")
        print(f"WROTE: {path.relative_to(ROOT)}")
    if stale:
        raise SystemExit("stale catalog projections: " + ", ".join(stale))
    if arguments.check:
        print("PASS: catalog projections are current")


if __name__ == "__main__":
    main()
