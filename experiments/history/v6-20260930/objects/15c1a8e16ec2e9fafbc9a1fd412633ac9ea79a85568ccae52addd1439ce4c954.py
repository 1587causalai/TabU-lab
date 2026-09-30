"""Evaluate V6 retention on the frozen 1,238-mask old618 Query bank.

This is read-only, target-column inference.  Repeated predictions of the same
Query address are averaged at the metric level before each table is scored,
matching the earlier V5.5 old618 evaluator.  The input masks, seeds, support
rows, Query addresses and target codecs all come from the frozen V5.5 bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from tabu_lab.curriculum_v53.artifacts import atomic_json, sha256
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration.contracts import RestorationRequest
from tabu_lab.models.restoration_v6 import V6Model


FROZEN_BANK_SHA256 = "e791488f67179a67ae2057440100144b34edbccb9c900f9633e2004d26284f41"


def _identity_digest(identity: dict) -> str:
    unsigned = {key: value for key, value in identity.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _table_metrics(table, address_scores: dict[int, dict]) -> dict:
    """Same per-address, equal-table numeric R2 / discrete metric units as V5.5."""
    if not address_scores:
        raise ValueError(f"{table.name}: fixed bank produced no Query addresses")
    items = list(address_scores.values())
    if table.schema[table.target_column].kind == "numeric":
        truth = [item["truth"] for item in items]
        center = _mean(truth)
        variance = _mean([(value - center) ** 2 for value in truth])
        mse = _mean([item["sse"] / item["visits"] for item in items])
        return {"r2": 1 - mse / variance if variance and variance > 0 else None,
                "mse": mse, "unique_query_rows": len(items),
                "query_exposures": sum(item["visits"] for item in items)}
    accuracy = _mean([item["correct"] / item["visits"] for item in items])
    result = {"accuracy": accuracy, "unique_query_rows": len(items),
              "query_exposures": sum(item["visits"] for item in items)}
    if table.schema[table.target_column].kind == "ordinal":
        result["rank_mae"] = _mean([item["rank_ae"] / item["visits"] for item in items])
    return result


def _macro(rows: list[dict]) -> dict:
    numeric = [row["metrics"]["r2"] for row in rows if row["target_kind"] == "numeric"]
    nominal = [row["metrics"]["accuracy"] for row in rows if row["target_kind"] == "nominal"]
    ordinal = [row["metrics"] for row in rows if row["target_kind"] == "ordinal"]
    if any(value is None for value in numeric):
        raise ValueError("numeric R2 undefined for a fixed-bank table")
    return {
        "numeric": {"tables": len(numeric), "r2": _mean(numeric)},
        "nominal": {"tables": len(nominal), "accuracy": _mean(nominal)},
        "ordinal": {"tables": len(ordinal),
                    "accuracy": _mean([item["accuracy"] for item in ordinal]),
                    "rank_mae": _mean([item["rank_mae"] for item in ordinal])},
    }


def _load_checkpoint(path: Path, expected_sha: str, current_plan, device: str):
    digest = sha256(path)
    if digest != expected_sha:
        raise ValueError("V6 checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "tabu.v6.weights-only-checkpoint.v1" or payload.get("purpose") != "training":
        raise ValueError("expected a V6 training checkpoint")
    identity = payload["identity"]
    if identity.get("sha256") != _identity_digest(identity):
        raise ValueError("V6 checkpoint identity digest mismatch")
    if identity.get("parent_identity_sha256") != current_plan.identity["sha256"]:
        raise ValueError("V6 parent identity differs from current OpenML12 plan")
    if identity.get("parent_manifest_sha256") != sha256(current_plan.path):
        raise ValueError("V6 parent manifest bytes differ")
    if payload["model_config"] != current_plan.config.as_dict() or identity["model_config"] != payload["model_config"]:
        raise ValueError("V6 checkpoint model config differs")
    if identity.get("device") != device:
        raise ValueError("V6 checkpoint was trained on a different device")
    return payload, digest


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "tabu.v6.old618-fixed-query-retention.v1",
        "outcome": "started", "host": args.host, "label": args.label,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "expected_checkpoint_sha256": args.expected_sha,
        "bank": str(Path(args.bank).resolve()),
        "evaluator_sha256": sha256(__file__), "optimizer_updates": 0,
        "probes": {},
    }
    atomic_json(output / "started.json", report)
    try:
        started = time.monotonic()
        runtime = configure_runtime(args.device)
        current = load_v55_plan(args.current_manifest)
        old = load_v55_plan(args.old_manifest)
        payload, checkpoint_sha = _load_checkpoint(
            Path(args.checkpoint), args.expected_sha, current, args.device
        )
        current_entries = {entry["id"]: entry for entry in current.spec["tables"]}
        if len(old.tables) != 618 or len(current.tables) != 630:
            raise ValueError("expected frozen old618 and OpenML12+old618 manifests")
        for entry in old.spec["tables"]:
            if current_entries[entry["id"]]["sha256"] != entry["sha256"]:
                raise ValueError(f"old table changed in current manifest: {entry['id']}")

        bank_sha = sha256(args.bank)
        if bank_sha != FROZEN_BANK_SHA256:
            raise ValueError("old618 fixed Query bank SHA mismatch")
        bank = json.loads(Path(args.bank).read_text())
        if bank.get("table_count") != 618 or bank.get("masks") != 1238:
            raise ValueError("old618 fixed Query bank count mismatch")
        bank_by_name = {entry["table"]: entry for entry in bank["tables"]}
        if len(bank_by_name) != 618 or set(bank_by_name) != {table.name for table in old.tables}:
            raise ValueError("old618 fixed Query bank table names differ")
        probe_by_name = {probe["name"]: probe for probe in old.spec["probes"]}
        selected = list(old.tables)
        if args.smoke:
            selected = selected[:1]
        if args.only_table:
            names = set(args.only_table)
            selected = [table for table in selected if table.name in names]
            if {table.name for table in selected} != names:
                raise ValueError("requested old618 table is absent")
        if not selected:
            raise ValueError("no old618 table selected")

        model = V6Model(current.config).to(device=args.device, dtype=execution_dtype(args.device))
        model.load_state_dict(payload["model"], strict=True)
        model.eval().requires_grad_(False)
        model_before = _model_digest(model)
        report.update(outcome="running", runtime=runtime, bank_sha256=bank_sha,
                      checkpoint_sha256=checkpoint_sha,
                      checkpoint_update=payload["state"]["update"],
                      v6_identity_sha256=payload["identity"]["sha256"],
                      model_state_sha256=model_before)
        atomic_json(output / "resolved.json", report)
        del payload

        checked_masks = checked_addresses = 0
        table_reports = []
        with torch.inference_mode():
            for table in selected:
                tick = time.monotonic()
                frozen = bank_by_name[table.name]
                probe = probe_by_name[frozen["probe"]]
                if table.cohort not in probe["cohorts"] or probe["partition"] != "train":
                    raise ValueError(f"{table.name}: bank probe does not match table")
                if len(frozen["masks"]) != probe["masks"]:
                    raise ValueError(f"{table.name}: frozen mask count differs")
                target = table.target_column
                schema = table.schema[target]
                ranks = None
                if schema.kind == "ordinal":
                    ranks = {label: position / max(schema.domain_size - 1, 1)
                             for position, label in enumerate(schema.order or range(schema.domain_size))}
                address_scores: dict[int, dict] = {}
                for index, expected in enumerate(frozen["masks"]):
                    if expected["mask_index"] != index:
                        raise ValueError(f"{table.name}: fixed mask order differs")
                    seeds = dict(old.spec["seeds"])
                    seeds["evaluation"] = int.from_bytes(hashlib.sha256(
                        f"{seeds['evaluation']}/{probe['name']}".encode()
                    ).digest()[:8], "little")
                    inputs, _full_request, truth, info = build_episode(
                        table, probe["recipe"], index, seeds, args.device,
                        evaluation=True, partition="train", epsilon=old.config.epsilon,
                        codec_version=old.config.codec_version,
                    )
                    for key in ("mask_seed", "code_seed", "window_seed"):
                        if info[key] != expected[key]:
                            raise ValueError(f"{table.name} mask {index}: {key} differs")
                    support_rows = sorted(set(info["row_ids"]) - set(info["query_row_ids"]))
                    if (info["query_addresses"] != expected["query_addresses"]
                            or info["query_row_ids"] != expected["query_rows"]
                            or support_rows != expected["support_rows"]):
                        raise ValueError(f"{table.name} mask {index}: Query/support addresses differ")
                    if bool(inputs.query[:, :target].any()) or bool(inputs.query[:, target + 1:].any()):
                        raise ValueError("V6 old618 evaluation requires one target Query column")
                    request = RestorationRequest(inputs.query.nonzero())
                    prediction = model(inputs, request)
                    if (len(prediction.columns) != 1 or prediction.columns[0].column != target
                            or prediction.columns[0].result.status != "ok"
                            or prediction.columns[0].decoded is None):
                        raise ValueError(f"{table.name} mask {index}: V6 target readout failed")
                    column = prediction.columns[0]
                    positions = column.target_indices
                    local_rows = request.targets[positions, 0].cpu().tolist()
                    raw = truth.values[target][request.targets[positions, 0]].cpu().tolist()
                    decoded = column.decoded.cpu().tolist()
                    if len(decoded) != len(local_rows):
                        raise ValueError("V6 target prediction count differs")
                    for local, actual, value in zip(local_rows, raw, decoded, strict=True):
                        if not math.isfinite(value):
                            raise FloatingPointError("nonfinite V6 old618 prediction")
                        address = int(info["row_ids"][local])
                        item = address_scores.setdefault(address, dict(
                            visits=0, truth=float(actual), sse=0.0, correct=0.0, rank_ae=0.0
                        ))
                        if item["truth"] != float(actual):
                            raise ValueError("same fixed Query address has inconsistent truth")
                        item["visits"] += 1
                        if schema.kind == "numeric":
                            item["sse"] += (float(value) - float(actual)) ** 2
                        else:
                            item["correct"] += int(value == actual)
                            if ranks is not None:
                                item["rank_ae"] += abs(ranks[int(value)] - ranks[int(actual)])
                    checked_masks += 1
                    checked_addresses += len(info["query_addresses"])
                    del inputs, _full_request, truth, prediction, column, request

                measured = _table_metrics(table, address_scores)
                table_reports.append({"table": table.name, "cohort": table.cohort,
                                      "target_kind": schema.kind, "target_column": target,
                                      "probe": probe["name"], "masks": len(frozen["masks"]),
                                      "metrics": measured, "seconds": time.monotonic() - tick})
                report["probes"].setdefault(probe["name"], {"by_table": []})["by_table"].append(
                    table_reports[-1]
                )
                atomic_json(output / "progress.json", report)
                print(json.dumps({"table": table.name, "kind": schema.kind,
                                  "metrics": measured, "seconds": table_reports[-1]["seconds"]}),
                      flush=True)

        expected_masks = sum(len(bank_by_name[table.name]["masks"]) for table in selected)
        if checked_masks != expected_masks:
            raise ValueError("fixed Query mask coverage incomplete")
        if not args.smoke and not args.only_table:
            kinds = {kind: sum(row["target_kind"] == kind for row in table_reports)
                     for kind in ("numeric", "nominal", "ordinal")}
            if (len(table_reports), checked_masks, kinds) != (
                618, 1238, {"numeric": 292, "nominal": 186, "ordinal": 140}
            ):
                raise ValueError("old618 table, mask, or target-type coverage differs")
        if _model_digest(model) != model_before or sha256(args.checkpoint) != checkpoint_sha:
            raise ValueError("read-only evaluation changed model or checkpoint")
        for name, probe_report in report["probes"].items():
            probe_report["tables"] = len(probe_report["by_table"])
            probe_report["macro"] = _macro(probe_report["by_table"])
        cohort_names = sorted({row["cohort"] for row in table_reports})
        report.update(
            outcome="completed", bank_checked_masks=checked_masks,
            bank_checked_query_addresses=checked_addresses,
            tables=len(table_reports), macro=_macro(table_reports),
            by_cohort={name: _macro([row for row in table_reports if row["cohort"] == name])
                       for name in cohort_names},
            aggregation="per-address mean over repeated masks; equal addresses per table; equal tables per type",
            ordinal_rank_metric="absolute difference of decoded declared rank divided by domain_size-1",
            claim_boundary=("one-table evaluator smoke only" if args.smoke or args.only_table
                            else "old618 training-row fixed-Query retention"),
            seconds=time.monotonic() - started,
        )
        atomic_json(output / "terminal.json", report)
        print(json.dumps({"outcome": "completed", "tables": report["tables"],
                          "masks": checked_masks, "macro": report["macro"]}), flush=True)
    except Exception as error:
        report.update(outcome="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(output / "terminal.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("host", "label", "current-manifest", "old-manifest", "checkpoint",
                 "expected-sha", "bank", "device", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only-table", nargs="+")
    main(parser.parse_args())
