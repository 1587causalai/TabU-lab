"""Paired fixed-Query old618 retention at the pre/mid/end checkpoints."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from tabu_lab.curriculum_v53 import evaluation, runner
from tabu_lab.curriculum_v53.artifacts import atomic_json, load_checkpoint, sha256
from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.models.restoration._dtype import execution_dtype


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema="tabu.openml12.old618-retention.v1", outcome="started",
                   host=args.host, label=args.label, checkpoint=str(Path(args.checkpoint).resolve()),
                   bank=str(Path(args.bank).resolve()), expected_checkpoint_sha256=args.expected_sha,
                   source_script_sha256=sha256(__file__))
    atomic_json(output / "started.json", receipt)
    native_build = evaluation.build_episode
    try:
        runtime = runner.configure_runtime(args.device)
        current = load_v55_plan(args.current_manifest)
        old = load_v55_plan(args.old_manifest)
        payload, checkpoint_sha = load_checkpoint(args.checkpoint)
        assert checkpoint_sha == args.expected_sha and payload["identity"] == current.identity
        assert current.spec["stages"][0]["objective"] == {"kind": "squared"}
        assert payload["model_config"] == old.config.as_dict() == current.config.as_dict()
        current_entries = {e["id"]: e for e in current.spec["tables"]}
        assert len(old.tables) == 618 and len(current.tables) == 630
        for entry in old.spec["tables"]:
            assert current_entries[entry["id"]]["sha256"] == entry["sha256"]
        bank_sha = sha256(args.bank)
        assert bank_sha == "e791488f67179a67ae2057440100144b34edbccb9c900f9633e2004d26284f41"
        bank = json.loads(Path(args.bank).read_text())
        assert bank["table_count"] == len(bank["tables"]) == 618 and bank["masks"] == 1238
        frozen = {entry["table"]: entry for entry in bank["tables"]}
        assert set(frozen) == {t.name for t in old.tables}
        checks = {"masks": 0, "addresses": 0}

        def checked_build(*pos, **kw):
            inputs, request, truth, info = native_build(*pos, **kw)
            table, index = pos[0], pos[2]
            expected = frozen[table.name]["masks"][index]
            assert frozen[table.name]["probe"] in {p["name"] for p in old.spec["probes"]
                                                      if table.cohort in p["cohorts"]}
            for key in ("mask_seed", "code_seed", "window_seed"):
                assert info[key] == expected[key], (table.name, index, key)
            assert info["query_addresses"] == expected["query_addresses"]
            assert info["query_row_ids"] == expected["query_rows"]
            supports = sorted(set(info["row_ids"]) - set(info["query_row_ids"]))
            assert supports == expected["support_rows"]
            checks["masks"] += 1
            checks["addresses"] += len(info["query_addresses"])
            return inputs, request, truth, info

        evaluation.build_episode = checked_build
        checkpoint_update = payload["state"]["update"]
        selected_probes = old.spec["probes"]
        if args.smoke:
            old = replace(old, tables=(old.tables[0],))
            selected_probes = [p for p in old.spec["probes"]
                               if old.tables[0].cohort in p["cohorts"]]
        expected_masks = sum(p["masks"] * sum(t.cohort in p["cohorts"] for t in old.tables)
                             for p in selected_probes)
        model = make_model(old).to(device=args.device, dtype=execution_dtype(args.device))
        model.load_state_dict(payload["model"], strict=True)
        model.eval().requires_grad_(False)
        del payload
        receipt.update(outcome="running", runtime=runtime, bank_sha256=bank_sha,
                       checkpoint_sha256=checkpoint_sha,
                       checkpoint_update=checkpoint_update,
                       probes={})
        atomic_json(output / "progress.json", receipt)
        for probe in selected_probes:
            report = evaluation.evaluate_probe(model, old, probe, args.device)
            receipt["probes"][probe["name"]] = report
            atomic_json(output / "progress.json", receipt)
            print(json.dumps(dict(probe=probe["name"], tables=report["tables"],
                                  seconds=report["seconds"], macro=report["macro"])), flush=True)
        assert checks["masks"] == expected_masks
        assert sha256(args.checkpoint) == checkpoint_sha
        receipt.update(outcome="completed", bank_checked_masks=checks["masks"],
                       bank_checked_query_addresses=checks["addresses"],
                       claim_boundary="fixed-Query training-row fit on old618"
                       if not args.smoke else "one-table evaluator smoke only")
        atomic_json(output / "terminal.json", receipt)
        print(json.dumps(dict(outcome="completed", host=args.host, label=args.label,
                              masks=checks["masks"])), flush=True)
    except Exception as error:
        receipt.update(outcome="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(output / "terminal.json", receipt)
        raise
    finally:
        evaluation.build_episode = native_build


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("host", "label", "current-manifest", "old-manifest", "checkpoint",
                "expected-sha", "bank", "device", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
