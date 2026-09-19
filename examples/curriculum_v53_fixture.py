"""Create a tiny, entirely synthetic curriculum fixture in a new directory.

No network, training, random global state, or external dataset is involved.
The new_a table's ``kind=real`` exercises recipe dispatch only; its provenance
is explicitly synthetic in both the table file and the manifest description.
"""

# Chinese questions in the generated manifest intentionally use Chinese punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _write_json(path: Path, value) -> str:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def create_fixture(output_root: Path) -> Path:
    """Return the manifest path; existing output directories are never reused."""
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_dir = output / "data"
    data_dir.mkdir()
    entries = []
    for number, name in enumerate(("old_a", "old_b", "new_a")):
        values = []
        for row in range(12):
            x = (row - 5.5) / 5.5
            category, rank = row % 2, row % 3
            y = (0.7 + 0.2 * number) * x + 0.1 * math.sin(3 * x + number)
            y += 0.15 * category - 0.1 * rank + 0.05 * number
            values.append([x, category, rank, y])
        table = {
            "name": name,
            "provenance": "deterministic_synthetic_fixture; not a real-world dataset",
            "values": values,
            "features": [
                {"kind": "numeric", "domain": []},
                {"kind": "nominal", "domain": ["a", "b"]},
                {"kind": "ordinal", "domain": ["middle", "high", "low"],
                 "order": [2, 0, 1]},
                {"kind": "numeric", "domain": []},
            ],
            "splits": {"train": list(range(8)), "validation": [8, 9], "test": [10, 11]},
        }
        relative = f"data/{name}.json"
        digest = _write_json(output / relative, table)
        entries.append({
            "id": name, "path": relative, "sha256": digest,
            "cohort": "old" if number < 2 else "new_simulated",
            "kind": "synthetic" if number < 2 else "real",
            "role": "train", "target_column": 3, "window_rows": None,
        })

    random_cell = {"kind": "random_cell", "fraction": 0.25,
                   "numeric_query_guard": {"kind": "none"}}
    supervised = {"kind": "supervised_row", "fraction": 0.25}

    def probe(name, cohorts, partition, purpose, recipe, masks=1):
        return {"name": name, "cohorts": cohorts, "partition": partition,
                "purpose": purpose, "recipe": recipe, "masks": masks}

    def stage(name, question, updates, cohort, kind, recipe, probes):
        return {
            "name": name, "question": question, "max_updates": updates,
            "max_seconds": 120.0, "sampling": [{"cohort": cohort, "episodes": 2}],
            "recipe": {kind: recipe}, "optimizer": "adamw",
            "evaluate_every": updates, "checkpoint_every": 1, "probes": probes,
        }

    manifest = {
        "schema": "tabu.curriculum.v53.v1",
        "experiment_id": "v53-local-curriculum-fixture",
        "description": (
            "全部三表均为本脚本生成的微型合成数据。new_a 的 kind=real 仅演示按 kind "
            "选择 recipe，不是现实数据证据。共 10 次更新用于本地流程检查；无性能门槛或能力声明。"
        ),
        "seeds": {"model": 53, "order": 101, "masks": 103, "codes": 107,
                  "windows": 109, "evaluation": 113},
        "model": {"backbone": {"width": 128, "layers": 1, "heads": 4,
                               "slots": 4, "ff_width": 24, "kind": "inducing"},
                  "unit_layers": 0, "regression_width": 4, "center_chunk_size": 4},
        "optimizer": {"learning_rate": 0.0001, "grad_clip": 1.0},
        "tables": entries,
        "probes": [
            probe("old_fit", ["old"], "train", "fit", random_cell, masks=2),
            probe("old_retention", ["old"], "train", "retention", random_cell, masks=2),
            probe("new_fit", ["new_simulated"], "train", "fit", random_cell, masks=2),
            probe("validation", ["old", "new_simulated"], "validation", "validation",
                  supervised),
            probe("final_test", ["old", "new_simulated"], "test", "final_test", supervised),
        ],
        "stages": [
            stage("fit", "同一模型能否同时优化两张固定训练表？", 4,
                  "old", "synthetic", random_cell, ["old_fit", "new_fit", "validation"]),
            stage("continual", "只学习新表后，新表拟合与旧表保留如何变化？", 3,
                  "new_simulated", "real", random_cell,
                  ["new_fit", "old_retention", "validation"]),
            stage("adapt", "同一新表改为监督行任务后，目标列拟合如何变化？", 3,
                  "new_simulated", "real", supervised,
                  ["new_fit", "old_retention", "validation"]),
        ],
    }
    path = output / "manifest.json"
    _write_json(path, manifest)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="new directory; refuses existing paths")
    args = parser.parse_args()
    print(create_fixture(args.output_root))


if __name__ == "__main__":
    main()
