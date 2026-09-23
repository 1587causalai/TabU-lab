"""Prepare a tiny single-table V5.4 fixture without starting training.

Reuse the deterministic synthetic table generator; only old_a participates in
this fixture's single fit stage. Validation/test rows remain reserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curriculum_v53_fixture import create_fixture as create_tables


def create_fixture(output_root: Path, *, size="small", episode_kind="supervised_row") -> Path:
    if size not in ("nano", "small") or episode_kind not in ("supervised_row", "random_cell"):
        raise ValueError("fixture requires nano/small and supervised_row/random_cell")
    path = create_tables(output_root)
    spec = json.loads(path.read_text())
    recipe = {"fraction": 0.25}  # Fixture budget, not a production training prescription.
    if episode_kind != "supervised_row":
        recipe["kind"] = episode_kind
    spec.update(
        schema="tabu.curriculum.v54.v1",
        experiment_id=f"v54-{size}-{episode_kind}-local-fixture",
        description="Two local steps on one deterministic synthetic table; no capability claim.",
        model={"size": size},
        tables=spec["tables"][:1],
        probes=[{"name": "fixed_fit", "cohorts": ["old"], "partition": "train",
                 "purpose": "fit", "recipe": dict(recipe), "masks": 1}],
        stages=[{
            "name": "single_table", "question": "Does the versioned local execution path work?",
            "max_updates": 2, "max_seconds": 120.0,
            "sampling": [{"cohort": "old", "episodes": 1}],
            "recipe": {"synthetic": recipe}, "optimizer": "adamw",
            "evaluate_every": 2, "checkpoint_every": 1, "probes": ["fixed_fit"],
        }],
    )
    spec["seeds"]["model"] = 54
    path.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--size", choices=("nano", "small"), default="small")
    parser.add_argument("--episode-kind", choices=("supervised_row", "random_cell"),
                        default="supervised_row")
    args = parser.parse_args()
    print(create_fixture(args.output_root, size=args.size, episode_kind=args.episode_kind))


if __name__ == "__main__":
    main()
