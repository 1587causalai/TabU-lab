"""Frozen probes with per-address, per-column and equal-table reporting."""

from __future__ import annotations

import hashlib
import math
import time

import torch

from tabu_lab.models.restoration_v53 import prepare_episode, score_prepared_episode
from tabu_lab.models.restoration_v53.readout import normalized_weights

from .artifacts import restore_rng, rng_state
from .data import build_episode


class BudgetExhausted(RuntimeError):
    """No additional work is admitted after the recorded wall budget."""


def mean(values):
    values = [value for value in values if value is not None]
    return math.fsum(values) / len(values) if values else None


def _finish(cells, columns):
    result = {}
    per_column = {}
    for state in ("retained", "query"):
        groups = {}
        for (label, column, _row), item in cells.items():
            if label != state:
                continue
            groups.setdefault(column, []).append(
                {key: value / item["visits"] for key, value in item.items() if key != "visits"}
            )
        for column, items in groups.items():
            numeric = columns[column].kind == "numeric"
            mse = mean([item["sse"] for item in items]) if numeric else None
            truth = [item["truth"] for item in items] if numeric else []
            variance = mean([(value - mean(truth)) ** 2 for value in truth]) if truth else None
            per_column[f"{state}/{column}"] = {
                "kind": columns[column].kind,
                "unique_targets": len(items),
                "numeric_mse": mse,
                "numeric_mae": mean([item["ae"] for item in items]) if numeric else None,
                "numeric_r2": 1 - mse / variance if variance and variance > 0 else None,
                "numeric_normalized_mse": mean([item["encoding"] for item in items])
                if numeric
                else None,
                "discrete_accuracy": mean([item["correct"] for item in items])
                if not numeric
                else None,
                "discrete_encoding_mse": mean([item["encoding"] for item in items])
                if not numeric
                else None,
                "ordinal_rank_mae": mean([item["rank_ae"] for item in items])
                if columns[column].kind == "ordinal" else None,
            }
        selected = [value for key, value in per_column.items() if key.startswith(state + "/")]
        for metric in (
            "numeric_mse",
            "numeric_mae",
            "numeric_r2",
            "numeric_normalized_mse",
            "discrete_accuracy",
            "discrete_encoding_mse",
            "ordinal_rank_mae",
        ):
            result[f"{state}_{metric}"] = mean([item[metric] for item in selected])
        result[f"{state}_unique_targets"] = sum(item["unique_targets"] for item in selected)
    return result, per_column


@torch.no_grad()
def evaluate_probe(model, plan, probe, device, *, deadline=None):
    """Evaluation consumes no training RNG and never updates model parameters.

    Averaging happens over repeat measurements of an address, then columns,
    then tables. It does not ensemble decoded predictions across masks/codecs.
    The default loss scale is used independently of stage training loss weights.
    """
    saved_rng, was_training = rng_state(), model.training
    started = time.monotonic()
    model.eval()
    tables = [table for table in plan.tables if table.cohort in probe["cohorts"]]
    reports = []
    try:
        for table in tables:
            cells, exposures, ess, maximum = {}, 0, [], []
            query_addresses = set()
            for index in range(probe["masks"]):
                if deadline is not None and time.monotonic() >= deadline:
                    raise BudgetExhausted(f"probe {probe['name']} exceeded its time budget")
                seeds = dict(plan.spec["seeds"])
                # Named bank identity stays fixed across stages and optimizer updates.
                seeds["evaluation"] = int.from_bytes(
                    hashlib.sha256(f"{seeds['evaluation']}/{probe['name']}".encode()).digest()[:8],
                    "little",
                )
                inputs, request, truth, info = build_episode(
                    table,
                    probe["recipe"],
                    index,
                    seeds,
                    device,
                    evaluation=True,
                    partition=probe["partition"],
                    epsilon=plan.config.epsilon,
                    codec_version=plan.config.codec_version,
                )
                score = score_prepared_episode(
                    model, prepare_episode(model, inputs, request, truth), decode=True
                )
                exposures += int(inputs.query.sum())
                for column in score.output.columns:
                    if column.result.status != "ok":
                        raise ValueError(f"probe {probe['name']}: {column.result.status}")
                    positions = column.target_indices
                    rows = request.targets[positions, 0]
                    expected = truth.values[column.column][rows]
                    spec = table.schema[column.column]
                    if spec.kind == "numeric":
                        # NMSE measures z-coordinate error. Raw sparse directions
                        # multiply TRAINING loss by 4, not this evaluation metric.
                        scale = score.output.facts[column.column].answers.scalar.scale
                        errors = ((column.decoded - expected.to(column.decoded.dtype)) / scale).square()
                    else:
                        errors = score.per_target[positions]
                    errors = errors.detach().cpu().tolist()
                    predicted = column.decoded.detach().cpu().tolist()
                    raw = expected.detach().cpu().tolist()
                    states = truth.states[rows, column.column].cpu().tolist()
                    rank = ({label: index / max(spec.domain_size - 1, 1)
                             for index, label in enumerate(spec.order or range(spec.domain_size))}
                            if spec.kind == "ordinal" else None)
                    for j, row in enumerate(rows.cpu().tolist()):
                        label = "query" if states[j] == 1 else "retained"
                        address = int(info["row_ids"][row])
                        key = (label, column.column, address)
                        item = cells.setdefault(
                            key,
                            dict(visits=0, encoding=0.0, sse=0.0, ae=0.0, correct=0.0,
                                 truth=0.0, rank_ae=0.0),
                        )
                        numeric = table.schema[column.column].kind == "numeric"
                        item["visits"] += 1
                        item["encoding"] += errors[j]
                        if numeric:
                            item["sse"] += (predicted[j] - raw[j]) ** 2
                            item["ae"] += abs(predicted[j] - raw[j])
                            item["truth"] += raw[j]
                        else:
                            item["correct"] += int(predicted[j] == raw[j])
                            if rank is not None:
                                item["rank_ae"] += abs(rank[predicted[j]] - rank[raw[j]])
                        if label == "query":
                            query_addresses.add((address, column.column))
                    query_rows = rows[truth.states[rows, column.column] == 1]
                    supports = score.output.facts[column.column].rows
                    for chunk in query_rows.split(32):
                        if not len(chunk):
                            continue
                        weights = normalized_weights(
                            score.output.units[chunk],
                            score.output.units[supports],
                            plan.config.bandwidth,
                        )
                        ess.extend((1 / weights.square().sum(-1)).cpu().tolist())
                        maximum.extend(weights.max(-1).values.cpu().tolist())
            metrics, per_column = _finish(cells, table.schema)
            reports.append(
                {
                    "table": table.name,
                    "cohort": table.cohort,
                    "kind": table.kind,
                    "metrics": metrics,
                    "by_column": per_column,
                    "query_exposures": exposures,
                    "unique_query_cells": len(query_addresses),
                    "query_rows": sorted({address[0] for address in query_addresses}),
                    "kernel": {
                        "query_ess_mean": mean(ess),
                        "query_ess_min": min(ess, default=None),
                        "query_max_weight_mean": mean(maximum),
                    },
                }
            )
        keys = reports[0]["metrics"] if reports else {}
        macro = {key: mean([report["metrics"][key] for report in reports]) for key in keys}
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetExhausted(f"probe {probe['name']} exceeded its time budget")
        return {
            "name": probe["name"],
            "partition": probe["partition"],
            "purpose": probe["purpose"],
            "complete": True,
            "tables": len(reports),
            "masks": probe["masks"],
            "macro": macro,
            "by_table": reports,
            "aggregation": "mean squared errors per address; equal columns; equal tables",
            "codec_version": plan.config.codec_version,
            "numeric_normalization": {
                "zscore": "each episode's forward-visible mean/population-std codec",
                "median_half_iqr": "each episode's forward-visible median/half-IQR codec",
            }[plan.config.numeric_scaling],
            "numeric_normalized_mse": "squared scalar-coordinate error; excludes code norm",
            "ordinal_rank_metric": "absolute error of decoded normalized declared rank",
            "claim_boundary": "training-row masked fit"
            if probe["partition"] == "train"
            else "supervised heldout rows; heldout predictors are visible (transductive)",
            "seconds": time.monotonic() - started,
        }
    finally:
        model.train(was_training)
        restore_rng(saved_rng)
