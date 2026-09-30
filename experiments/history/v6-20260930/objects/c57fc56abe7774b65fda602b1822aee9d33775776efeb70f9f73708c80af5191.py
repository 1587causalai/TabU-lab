"""Freeze actual evaluator addresses for all 618 tables before training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from tabu_lab.curriculum_v53.data import build_episode, load_table


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifests/candidate.json"
OLD = ROOT.parent / "triad90b-mini-20260926"
OLD_BANK = OLD / "assets/evidence/ordinal-bank/actual-evaluation-bank-addresses.json"
OUTPUT = ROOT / "evidence/actual-evaluation-bank-addresses.json"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> None:
    torch.set_num_threads(1)
    manifest = json.loads(MANIFEST.read_text())
    old = json.loads((OLD / "numeric/manifests/candidate.json").read_text())
    old_bank = json.loads(OLD_BANK.read_text())
    assert manifest["tables"][:498] == old["tables"]
    assert manifest["probes"][:5] == old["probes"]
    assert manifest["seeds"] == old["seeds"]
    assert old_bank["table_count"] == 498 and old_bank["mask_count"] == 998
    probes = {p["cohorts"][0]: p for p in manifest["probes"]}
    assert set(probes) == {"ordinal100", "nominal100", "tanh100", "old120", "new120", "real78"}
    rows = []
    for entry in manifest["tables"]:
        table = load_table(entry, MANIFEST.parent)
        assert table.digest == entry["sha256"] and table.train_rows == 204
        probe = probes[entry["cohort"]]
        seeds = dict(manifest["seeds"])
        seeds["evaluation"] = int.from_bytes(hashlib.sha256(
            f"{seeds['evaluation']}/{probe['name']}".encode()).digest()[:8], "little")
        masks = []
        for index in range(probe["masks"]):
            _, _, _, info = build_episode(table, probe["recipe"], index, seeds, "cpu",
                                         evaluation=True, partition=probe["partition"],
                                         epsilon=manifest["model"].get("epsilon", 1e-6),
                                         codec_version=manifest["model"]["codec_version"])
            queries = info["query_row_ids"]
            row_ids = info["row_ids"]
            assert len(row_ids) == 204 and len(queries) == 51
            support = sorted(set(row_ids) - set(queries))
            assert len(support) == 153
            masks.append({
                "index": info["index"], "mask_seed": info["mask_seed"],
                "code_seed": info["code_seed"], "window_seed": info["window_seed"],
                "row_ids": row_ids, "query_addresses": info["query_addresses"],
                "query_count": info["query_count"], "query_rows": queries,
                "support_count": len(support), "support_rows": support,
                "target_column": table.target_column, "evaluation": info["evaluation"],
                "partition": info["partition"],
            })
        rows.append([table.name, probe["name"], masks])
    assert len(rows) == 618 and sum(len(r[2]) for r in rows) == 1238
    for rebuilt, original in zip(rows[:498], old_bank["tables"], strict=True):
        assert rebuilt == original, rebuilt[0]
    old_projection = [[name, probe, [mask["query_addresses"] for mask in masks]]
                      for name, probe, masks in rows[:498]]
    assert digest(old_projection) == "baf6614adb3a55fada76a5577e6aeafeabb1446be181664947d17afa54632171"
    projection = [[name, probe, [mask["query_addresses"] for mask in masks]]
                  for name, probe, masks in rows]
    frozen = {
        "schema": "tabu.mini.joint618.actual-evaluation-bank.v1",
        "manifest_file_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "source_data_identity_sha256": json.loads((ROOT / "plan/plan.json").read_text())["identity"]["sha256"],
        "table_count": 618, "mask_count": 1238,
        "old498_query_bank_sha256": digest(old_projection),
        "query_address_bank_sha256": digest(projection),
        "tables": rows,
    }
    OUTPUT.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in frozen.items() if k != "tables"}, indent=2))


if __name__ == "__main__":
    main()
