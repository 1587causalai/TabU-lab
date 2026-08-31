#!/usr/bin/env python3
"""Run the evolution-program CLI from a portable checked-out source archive."""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

def _main() -> int:
    from tabu_lab.cli import main

    return main(["program", "run", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(_main())
