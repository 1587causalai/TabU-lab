"""Prepare matched DGX2 H8 V5.5/V6 manifests for immutable seed2 data.

Both consumers load the same V5.5 plan to preserve the model configuration,
sampling recipe, optimizer and episode seeds.  V6 changes only model forward
and training loss in its separate runner.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT.parents[1] / "v6-sparse-relevance-600-20260927" / "manifests"
ARMS = ("signal6", "full32")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        output.write("\n")


def main() -> None:
    generated = json.loads((ROOT / "data-receipt.json").read_text())
    assert generated["generator_seed"] == 2026092702
    assert generated["relevant_columns"] == [2, 9, 11, 14, 16, 27]
    provenance = {}
    for arm in ARMS:
        source = TEMPLATES / f"{arm}.json"
        template = json.loads(source.read_text())
        manifest = copy.deepcopy(template)
        data = ROOT / "data" / f"{arm}.json"
        expected = generated["data"][arm]["sha256"]
        if sha256(data) != expected:
            raise ValueError(f"seed2 {arm} data changed after generation")
        manifest["experiment_id"] = f"sparse-relevance-seed2-{arm}-20260927"
        manifest["description"] = "Independent teacher seed and relevant-column placement; paired V5.5/V6 DGX2 H8 fit"
        assert len(manifest["tables"]) == 1
        table = manifest["tables"][0]
        table.update(id=arm, path=f"../data/{arm}.json", sha256=expected,
                     target_column=generated["data"][arm]["target_column"])
        manifest["stages"][0]["question"] = "Does target broadcasting improve sparse-relevance fitting?"
        assert manifest["model"]["size"] == "small"
        assert manifest["model"]["backbone"]["heads"] == 8
        assert manifest["model"]["unit_layers"] == 3
        assert manifest["model"]["numeric_scaling"] == "zscore"
        assert manifest["stages"][0]["max_updates"] == 600
        assert manifest["stages"][0]["objective"] == {"kind": "squared"}
        output = ROOT / "manifests" / f"{arm}.json"
        write_new(output, manifest)
        provenance[arm] = dict(data_sha256=expected, manifest_sha256=sha256(output),
                               template_sha256=sha256(source),
                               target_column=table["target_column"])
    a = json.loads((ROOT / "manifests/signal6.json").read_text())
    b = json.loads((ROOT / "manifests/full32.json").read_text())
    for field in ("model", "optimizer", "seeds", "stages"):
        if a[field] != b[field]:
            raise ValueError(f"seed2 arms differ in {field}")
    receipt = dict(schema="tabu.sparse-relevance.seed2-manifests.v1",
                   generator_seed=generated["generator_seed"],
                   model="Small H8 / Unit3, DGX2 parent weights-only",
                   training="synthetic_fit, 204 rows, 1/3 Query, squared loss",
                   evaluation="8 fixed episodes; 136 support + 68 train/test Query per arm",
                   arms=provenance)
    write_new(ROOT / "manifest-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
