"""Frozen probes with per-address, per-column and equal-table reporting."""

from __future__ import annotations

import hashlib
import math
import time

import torch

from tabu_lab.models.restoration_v53 import prepare_episode, score_prepared_episode
from tabu_lab.models.restoration_v53.readout import normalized_weights_and_logits

from .artifacts import restore_rng, rng_state
from .data import build_episode


class BudgetExhausted(RuntimeError):
    """No additional work is admitted after the recorded wall budget."""


def mean(values):
    values = [value for value in values if value is not None]
    return math.fsum(values) / len(values) if values else None


_KERNEL_PATHS = ("query_to_support", "ll_centers_to_support")
_KERNEL_METRICS = ("ess", "relative_ess", "support_count", "max_weight", "logit_gap")


def _empty_kernel_samples():
    return {name: [] for name in _KERNEL_METRICS}


def _extend_kernel_samples(destination, source):
    for name in _KERNEL_METRICS:
        destination[name].extend(source[name])


def _distribution(values):
    """A compact, deterministic empirical distribution; quantiles interpolate linearly."""
    if not values:
        return {key: 0 if key == "count" else None
                for key in ("count", "min", "p05", "median", "p95", "max", "mean")}
    ordered = sorted(values)
    if not all(math.isfinite(value) for value in ordered):
        raise FloatingPointError("nonfinite kernel diagnostic")

    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        left = math.floor(position)
        return ordered[left] + (position - left) * (ordered[math.ceil(position)] - ordered[left])

    return {
        "count": len(ordered), "min": ordered[0], "p05": quantile(0.05),
        "median": quantile(0.5), "p95": quantile(0.95), "max": ordered[-1],
        "mean": math.fsum(ordered) / len(ordered),
    }


def _summarize_kernel_samples(samples):
    result = {name: _distribution(samples[name]) for name in _KERNEL_METRICS}
    result["center_count"] = result["ess"]["count"]
    return result


@torch.no_grad()
def _kernel_samples(units, support_rows, query_rows, bandwidth, *, chunk_size=32,
                    deadline=None):
    """Measure the same Gaussian/softmax weights used by the LL and query readout.

    All Unit rows are LL fitting centers. Query centers are the requested query
    target rows, including repeated requests if present. The top-two logit gap
    stays informative when FP32 softmax rounds the second weight down to zero.
    It is undefined when a column has just one support.
    """
    if not len(support_rows):
        raise ValueError("kernel diagnostics require visible supports")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("kernel diagnostic chunk_size must be a positive integer")
    samples = {path: _empty_kernel_samples() for path in _KERNEL_PATHS}
    sources = units[support_rows]
    requested = query_rows.cpu().tolist()
    count = len(support_rows)
    for start in range(0, len(units), chunk_size):
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetExhausted("kernel diagnostics exceeded probe time budget")
        centers = units[start:start + chunk_size]
        weights, logits = normalized_weights_and_logits(centers, sources, bandwidth)
        ess = weights.square().sum(-1).reciprocal()
        maximum = weights.max(-1).values
        components = [ess, ess / count, torch.full_like(ess, count), maximum]
        if count > 1:
            best = logits.topk(2, dim=-1).values
            components.append(best[:, 0] - best[:, 1])
        rows = torch.stack(components, dim=-1).cpu().tolist()
        query_indices = [row - start for row in requested if start <= row < start + len(rows)]
        for path, indices in (("ll_centers_to_support", range(len(rows))),
                              ("query_to_support", query_indices)):
            target = samples[path]
            for index in indices:
                values = rows[index]
                for metric, value in zip(_KERNEL_METRICS, values, strict=False):
                    target[metric].append(value)
    return samples


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
            cells, exposures = {}, 0
            kernel_by_column = {path: {} for path in _KERNEL_PATHS}
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
                        residual = column.decoded - expected.to(column.decoded.dtype)
                        errors = (residual / scale).square()
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
                    column_samples = _kernel_samples(
                        score.output.units, supports, query_rows, plan.config.bandwidth,
                        deadline=deadline,
                    )
                    for path in _KERNEL_PATHS:
                        target = kernel_by_column[path].setdefault(
                            str(column.column), _empty_kernel_samples(),
                        )
                        _extend_kernel_samples(target, column_samples[path])
            metrics, per_column = _finish(cells, table.schema)
            kernel = {
                "mode": "gaussian_unit_softmax",
                "bandwidth": float(plan.config.bandwidth),
            }
            for path in _KERNEL_PATHS:
                aggregate = _empty_kernel_samples()
                for values in kernel_by_column[path].values():
                    _extend_kernel_samples(aggregate, values)
                kernel[path] = {
                    **_summarize_kernel_samples(aggregate),
                    "by_column": {
                        column: _summarize_kernel_samples(values)
                        for column, values in sorted(kernel_by_column[path].items())
                    },
                }
            # Preserve the historical flat Query fields for existing readers.
            query_kernel = kernel["query_to_support"]
            kernel.update({
                "query_ess_mean": query_kernel["ess"]["mean"],
                "query_ess_min": query_kernel["ess"]["min"],
                "query_max_weight_mean": query_kernel["max_weight"]["mean"],
            })
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
                    "kernel": kernel,
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
