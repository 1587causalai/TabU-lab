"""Three-stage Small-128 restoration curriculum.

This is a restoration-model runner, not a reimplementation of legacy TAR.  It
reuses the typed restoration episode/scorer and gives the training process the
historical TAR data schedule: synthetic random-cell episodes in the first two
stages, followed by supervised-row episodes for real tables with synthetic
replay.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from tabu_lab.models.restoration import (
    ColumnSchema,
    LossConfig,
    RestorationConfig,
    RestorationModel,
    make_episode,
    prepare_episode,
    score_prepared_episode,
)
from tabu_lab.restoration_joint_fit import _distribution, _seed
from tabu_lab.restoration_masking import global_query_mask, validate_numeric_query_guard
from tabu_lab.restoration_optimizers import OptimizerConfig, adamw, switch_to_muon
from tabu_lab.tar_data import validate_full_dataset

SCHEMA = "tabu.restoration.curriculum-fit.v1"
CHECKPOINT_SCHEMA = "tabu.restoration.curriculum-fit-checkpoint.v1"
SYNTHETIC = "synthetic"
REAL = "real"


def _configure_runtime(device):
    if device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_mapping(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"mapping expected: {path}")
    return dict(value)


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 if zero else value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _loss_config(spec):
    """Resolve opt-in state means without changing archived all-cell recipes."""
    if "objective" not in spec:
        return LossConfig()
    objective = spec["objective"]
    if (not isinstance(objective, dict)
            or set(objective) != {"kind", "retained_weight", "query_weight"}
            or objective["kind"] != "state_weighted"):
        raise ValueError("objective must declare state_weighted, retained_weight and query_weight")
    retained = _positive(objective["retained_weight"], "retained_weight")
    query = _positive(objective["query_weight"], "query_weight")
    if not math.isclose(retained + query, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("retained_weight and query_weight must sum to one")
    return LossConfig((retained, query, 0.0, 0.0))


def _numeric_query_guard(spec):
    # Frozen recipes without an explicit policy retain their recorded cell guard.
    return validate_numeric_query_guard(spec.get("numeric_query_guard", {
        "kind": "median_half_iqr", "max_abs_robust_z": 4.0,
    }))


def _loss_terms(score, prepared, loss_config):
    """Log state means and their actual contributions under the active objective."""
    errors = score.per_target.detach()
    means = errors.new_zeros(2)
    contributions = errors.new_zeros(2)
    for type_mask in prepared.scoring.types:
        count = type_mask.sum().clamp_min(1)
        for state in (0, 1):
            selected = type_mask & prepared.scoring.state_masks[state]
            state_count = selected.sum().clamp_min(1)
            mean = (errors[selected] / state_count).sum()
            means[state] += mean
            contributions[state] += (
                mean * loss_config.state_weights[state]
                if loss_config.state_weights is not None
                else (errors[selected] / count).sum()
            )
    retained, query, retained_part, query_part = torch.cat((means, contributions)).cpu().tolist()
    return {"retained_loss": retained, "query_loss": query,
            "retained_loss_contribution": retained_part, "query_loss_contribution": query_part}


@dataclass(frozen=True)
class CurriculumTable:
    name: str
    cohort: str
    kind: str
    values: tuple[torch.Tensor, ...]
    schema: tuple[ColumnSchema, ...]
    row_ids: tuple[int, ...]
    reserved_rows: int
    windowed: bool = False

    @property
    def width(self):
        return len(self.schema)

    @property
    def train_rows(self):
        return len(self.row_ids)


def _schema(name: str, features: list[dict] | None, values: list[list]) -> tuple[ColumnSchema, ...]:
    width = len(values[0])
    if features is None:
        features = [{"kind": "numeric", "domain": []} for _ in range(width)]
    if not isinstance(features, list) or len(features) != width:
        raise ValueError(f"{name}: feature schema width mismatch")
    result = []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict) or feature.get("kind") not in (
            "numeric",
            "nominal",
            "ordinal",
        ):
            raise ValueError(f"{name}: unsupported feature kind at column {index}")
        kind = feature["kind"]
        if kind == "numeric":
            result.append(ColumnSchema(f"{name}/column-{index}", kind))
            continue
        domain = feature.get("domain")
        if not isinstance(domain, list) or not domain:
            raise ValueError(f"{name}: discrete column needs a declared domain")
        order = feature.get("order")
        result.append(
            ColumnSchema(
                f"{name}/column-{index}",
                kind,
                len(domain),
                tuple(order) if order is not None else None,
            )
        )
    return tuple(result)


def load_table(path: Path, name: str, cohort: str, kind: str, *, windowed=False) -> CurriculumTable:
    data = _load_mapping(path)
    values = data.get("values")
    if not isinstance(values, list) or not values or not isinstance(values[0], list):
        raise ValueError(f"{name}: values must be a nonempty matrix")
    width = len(values[0])
    if any(not isinstance(row, list) or len(row) != width for row in values):
        raise ValueError(f"{name}: values must be rectangular")
    if "splits" not in data:
        raise ValueError(f"{name}: explicit train/test split required")
    coverage = validate_full_dataset(data, len(values))
    schema = _schema(name, data.get("features"), values)
    columns = []
    for column, spec in enumerate(schema):
        raw = [row[column] for row in values]
        if spec.kind == "numeric":
            if any(
                isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
                for v in raw
            ):
                raise ValueError(f"{name}: numeric values must be finite")
            columns.append(
                torch.tensor([raw[i] for i in data["splits"]["train"]], dtype=torch.float64)
            )
        else:
            if any(type(v) is not int or not 0 <= v < spec.domain_size for v in raw):
                raise ValueError(f"{name}: discrete value outside declared domain")
            columns.append(
                torch.tensor([raw[i] for i in data["splits"]["train"]], dtype=torch.long)
            )
    return CurriculumTable(
        name,
        cohort,
        kind,
        tuple(columns),
        schema,
        tuple(data["splits"]["train"]),
        coverage["test_rows"],
        windowed,
    )


def _window(table: CurriculumTable, index: int, seed: int, *, evaluation=False):
    """Addressed <=204-row windows used by the historical nine-table panel."""
    n = table.train_rows
    size = min(n, 204)
    query_size = math.ceil(size / 3)
    count = math.ceil(n / query_size)
    round_id = 0 if evaluation else index // count
    slot = index if evaluation else index % count
    generator = torch.Generator().manual_seed(_seed(seed, table.name, round_id, "query_order"))
    order = torch.randperm(n, generator=generator).tolist()
    query = [order[(slot * query_size + j) % n] for j in range(query_size)]
    selected = set(query)
    generator = torch.Generator().manual_seed(_seed(seed, table.name, index, "context_rows"))
    context = [j for j in torch.randperm(n, generator=generator).tolist() if j not in selected][
        : size - query_size
    ]
    indices = context + query
    return tuple(table.values[column][indices] for column in range(table.width)), tuple(
        table.row_ids[index] for index in indices
    )


def _as_plan(table: CurriculumTable, *, index: int, seed: int, evaluation=False):
    values, row_ids = (
        (table.values, table.row_ids)
        if not table.windowed
        else _window(table, index, seed, evaluation=evaluation)
    )
    return table, values, row_ids


def _random_cell_episode(table, values, row_ids, fraction, seed, config, device,
                         numeric_query_guard=None):
    plan = _PlanView(table.name, values, table.schema, row_ids)
    query, info = global_query_mask(
        plan,
        fraction,
        seed,
        numeric_query_guard=numeric_query_guard if numeric_query_guard is not None else {
            "kind": "median_half_iqr",
            "max_abs_robust_z": 4.0,
        },
        numeric_scale_floor=config.encoder.epsilon,
    )
    return _episode(plan, query, _seed(seed, "code"), device), info


def _supervised_row_episode(table, values, row_ids, fraction, seed, device):
    plan = _PlanView(table.name, values, table.schema, row_ids)
    rows = plan.train_rows
    hidden = max(1, min(rows - 2, math.ceil(rows * fraction)))
    order = list(range(rows))
    random.Random(seed).shuffle(order)
    query = torch.zeros(rows, plan.width, dtype=torch.bool)
    query[order[:hidden], plan.width - 1] = True
    return _episode(plan, query, _seed(seed, "code"), device), {
        "mask_mode": "supervised_row",
        "query_rows": hidden,
        "query_count": hidden,
        "query_coverage": hidden / rows,
        "tail_guard_enabled": False,
    }


@dataclass(frozen=True)
class _PlanView:
    name: str
    values: tuple[torch.Tensor, ...]
    schema: tuple[ColumnSchema, ...]
    row_ids: tuple[int, ...]

    @property
    def train_rows(self):
        return len(self.row_ids)

    @property
    def width(self):
        return len(self.schema)


def _episode(table, query, code_seed, device):
    values = tuple(value.to(device) for value in table.values)
    query = query.to(device)
    return make_episode(table.schema, values, torch.ones_like(query), query, code_seed=code_seed)


def _episode_for(
    table: CurriculumTable,
    index: int,
    stage: dict,
    seeds: dict,
    config,
    device,
    *,
    evaluation=False,
    numeric_query_guard=None,
):
    _, values, row_ids = _as_plan(table, index=index, seed=seeds["windows"], evaluation=evaluation)
    mask_mode = stage.get(
        "synthetic_mask" if table.kind == SYNTHETIC else "real_mask", stage["mask"]
    )
    seed = _seed(
        seeds["masks"], stage["name"], table.name, index, "eval" if evaluation else "train"
    )
    fraction = stage.get(
        "synthetic_mask_fraction" if table.kind == SYNTHETIC else "real_mask_fraction",
        stage.get("mask_fraction", 1 / 3),
    )
    if mask_mode == "random_cell":
        return _random_cell_episode(table, values, row_ids, fraction, seed, config, device,
                                    numeric_query_guard=numeric_query_guard)
    if mask_mode == "supervised_row":
        return _supervised_row_episode(table, values, row_ids, fraction, seed, device)
    raise ValueError(f"unsupported mask mode {mask_mode}")


def _metric_add(acc, state, kind, encoding, numeric, correct):
    item = acc[state]
    item["count"] += 1
    item["encoding_sse"] += float(encoding)
    if kind == "numeric":
        item["numeric_count"] += 1
        item["numeric_sse"] += float(numeric)
    else:
        item["discrete_count"] += 1
        item["discrete_correct"] += int(correct)


def _finish_metrics(acc):
    result = {}
    for state, item in acc.items():
        out = {
            "count": item["count"],
            "encoding_mse": item["encoding_sse"] / item["count"] if item["count"] else None,
            "numeric_count": item["numeric_count"],
            "numeric_mse": item["numeric_sse"] / item["numeric_count"]
            if item["numeric_count"]
            else None,
            "discrete_count": item["discrete_count"],
            "discrete_accuracy": (
                item["discrete_correct"] / item["discrete_count"]
                if item["discrete_count"]
                else None
            ),
        }
        out["discrete_error_rate"] = (
            1 - out["discrete_accuracy"] if out["discrete_accuracy"] is not None else None
        )
        result[state] = out
    return result


def _new_acc():
    return {
        "retained": {
            "count": 0, "encoding_sse": 0.0, "numeric_count": 0,
            "numeric_sse": 0.0, "discrete_count": 0, "discrete_correct": 0,
        },
        "query": {
            "count": 0, "encoding_sse": 0.0, "numeric_count": 0,
            "numeric_sse": 0.0, "discrete_count": 0, "discrete_correct": 0,
        },
    }


def evaluate(model, tables, stage, seeds, config, device, *, masks=8, deadline=None,
             loss_config=None, numeric_query_guard=None):
    model.eval()
    by_table = {}
    by_type = {}
    by_source = {}
    losses = []
    coverage = Counter()
    complete = True
    with torch.no_grad():
        for table in tables:
            acc = _new_acc()
            for index in range(masks):
                if deadline is not None and time.monotonic() >= deadline:
                    complete = False
                    break
                episode, info = _episode_for(
                    table, index, stage, seeds, config, device, evaluation=True,
                    numeric_query_guard=numeric_query_guard,
                )
                score = score_prepared_episode(
                    model, prepare_episode(model, *episode), loss_config=loss_config,
                    decode=True, report=True
                )
                if not bool(torch.isfinite(score.loss)):
                    raise FloatingPointError("nonfinite evaluation loss")
                losses.append(float(score.loss.detach().cpu()))
                targets = episode[1].targets
                states = episode[2].states[targets[:, 0], targets[:, 1]]
                for prediction in score.output.columns:
                    positions = prediction.target_indices
                    kind = table.schema[prediction.column].kind
                    for local, position in enumerate(positions.tolist()):
                        state = "query" if int(states[position]) == 1 else "retained"
                        encoded = score.per_target[position]
                        actual = episode[2].values[prediction.column][targets[position, 0]]
                        if kind == "numeric":
                            error = (prediction.decoded[local] - actual).square()
                            _metric_add(acc, state, kind, encoded, error, False)
                            _metric_add(by_type.setdefault(kind, _new_acc()), state, kind,
                                        encoded, error, False)
                            _metric_add(by_source.setdefault(table.cohort, _new_acc()), state,
                                        kind, encoded, error, False)
                        else:
                            correct = prediction.decoded[local] == actual
                            _metric_add(
                                acc, state, kind, encoded, 0.0, correct
                            )
                            _metric_add(by_type.setdefault(kind, _new_acc()), state, kind,
                                        encoded, 0.0, correct)
                            _metric_add(by_source.setdefault(table.cohort, _new_acc()), state,
                                        kind, encoded, 0.0, correct)
                coverage["query_cells"] += int(
                    episode[1].targets.shape[0] and (episode[2].states == 1).sum()
                )
                coverage["total_cells"] += int((episode[2].states >= 0).sum())
                coverage["protected_numeric_tail_cells"] += int(
                    info.get("protected_numeric_tail_cells", 0)
                )
                coverage["query_numeric_tail_cells"] += int(info.get("query_numeric_tail_cells", 0))
                for key in ("protected_numeric_columns", "protected_numeric_column_cells",
                            "query_protected_numeric_column_cells"):
                    if key in info:
                        coverage[key] += int(info[key])
            if not complete:
                break
            by_table[table.name] = _finish_metrics(acc)
    by_state = {}
    for state in ("retained", "query"):
        merged = {
            "count": 0,
            "encoding_sse": 0.0,
            "numeric_count": 0,
            "numeric_sse": 0.0,
            "discrete_count": 0,
            "discrete_correct": 0,
        }
        for item in by_table.values():
            values = item[state]
            merged["count"] += values["count"]
            merged["encoding_sse"] += (
                values["encoding_mse"] * values["count"] if values["count"] else 0
            )
            merged["numeric_count"] += values["numeric_count"]
            merged["numeric_sse"] += (
                values["numeric_mse"] * values["numeric_count"] if values["numeric_count"] else 0
            )
            merged["discrete_count"] += values["discrete_count"]
            merged["discrete_correct"] += (
                values["discrete_accuracy"] * values["discrete_count"]
                if values["discrete_count"]
                else 0
            )
        by_state[state] = _finish_metrics({state: merged})[state]
    by_type = {kind: _finish_metrics(values) for kind, values in by_type.items()}
    by_source = {source: _finish_metrics(values) for source, values in by_source.items()}
    coverage["query_fraction"] = (
        coverage["query_cells"] / coverage["total_cells"]
        if coverage["total_cells"] else None
    )
    return {
        "loss": sum(losses) / len(losses) if losses else None,
        "by_state": by_state,
        "by_type": by_type,
        "by_source": by_source,
        "by_table": by_table,
        "coverage": dict(coverage),
        "complete": complete,
        "evaluation_masks": masks,
        "scope": "fixed masks on training rows; reserved rows excluded",
    }


def _schedule(stage, tables, seeds):
    name = stage["name"]
    rng_seed = seeds["order"]
    if name == "old120":
        old = [table for table in tables if table.cohort == "old120"]
        for round_index in range(stage["max_updates"] // len(old)):
            order = list(old)
            random.Random(f"{rng_seed}/{name}/{round_index}").shuffle(order)
            yield from ((table, round_index, position) for position, table in enumerate(order))
    elif name == "recent120_replay":
        new = [table for table in tables if table.cohort == "recent120"]
        old = [table for table in tables if table.cohort == "old120"]
        for round_index in range(stage["max_updates"] // 150):
            selected = old[(round_index * 30) % len(old) : ((round_index * 30) % len(old)) + 30]
            if len(selected) < 30:
                selected += old[: 30 - len(selected)]
            entries = new + selected
            random.Random(f"{rng_seed}/{name}/{round_index}").shuffle(entries)
            yield from ((table, round_index, position) for position, table in enumerate(entries))
    elif name == "openml12_mixed":
        real = [table for table in tables if table.kind == REAL]
        synthetic = [table for table in tables if table.kind == SYNTHETIC]
        if len(real) != 12 or len(synthetic) != 240:
            raise ValueError("openml12_mixed requires exactly 12 real and 240 synthetic tables")
        blocks = stage["max_updates"] // 15
        replay = list(synthetic)
        random.Random(f"{rng_seed}/{name}/replay").shuffle(replay)
        for block in range(blocks):
            entries = real + [replay[(block * 3 + offset) % len(replay)] for offset in range(3)]
            random.Random(f"{rng_seed}/{name}/block/{block}").shuffle(entries)
            # One reported cycle is 1,200 updates = 80 fifteen-step blocks.
            # ``position`` is therefore the position within that 1,200-step
            # cycle, while the block index remains recoverable for real-table
            # window addressing below.
            cycle = block // 80
            offset = (block % 80) * 15
            yield from ((table, cycle, offset + position) for position, table in enumerate(entries))
    else:
        raise ValueError(f"unsupported curriculum stage {name}")


def _episode_index(stage_name, cycle, position):
    if stage_name == "openml12_mixed":
        return cycle * 80 + position // 15
    return cycle


def _identity(spec, prereg: Path, tables, source_digest):
    return {
        "schema": SCHEMA,
        "preregistration_sha256": _hash(prereg),
        "source": source_digest,
        "model": spec["model"],
        "objective": spec.get("objective", {"kind": "type_mean_all_observed"}),
        "numeric_query_guard": _numeric_query_guard(spec),
        "table_digests": spec.get("table_digests", {}),
        "table_names": [table.name for table in tables],
    }


def _source_digest():
    root = Path(__file__).parent
    paths = [Path(__file__), root / "cli.py", root / "restoration_masking.py",
             root / "restoration_optimizers.py", root / "observers" / "restoration.py"]
    paths.extend(sorted((root / "models" / "restoration").glob("*.py")))
    files = {str(path.relative_to(root)): _hash(path) for path in paths}
    return {"files": files, "sha256": _digest(files)}


def _finite_state(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_state(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_finite_state(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _checkpoint(
    path,
    model,
    optimizer,
    identity,
    stage_index,
    update,
    cursor,
    elapsed,
    *,
    stage_name,
    stage_elapsed=0.0,
    cycle_losses=(),
    cycle_source_losses=None,
):
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "identity": identity,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "optimizer_kind": "mixed" if hasattr(optimizer, "muon") else "adamw",
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "stage_index": stage_index,
        "stage_name": stage_name,
        "update": update,
        "cursor": cursor,
        "stage_elapsed_seconds": stage_elapsed,
        "elapsed_seconds": elapsed,
        "cycle_losses": list(cycle_losses),
        "cycle_source_losses": {
            key: list(values) for key, values in (cycle_source_losses or {}).items()
        },
    }
    if not _finite_state(state["model"]) or not _finite_state(state["optimizer"]):
        raise FloatingPointError("nonfinite model checkpoint")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_checkpoint(path, model, optimizer, identity):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != CHECKPOINT_SCHEMA or state.get("identity") != identity:
        raise ValueError("curriculum checkpoint identity drift")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng"):
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    return state


class WallLimit(RuntimeError):
    pass


def _optimizer_config(spec):
    item = spec["optimizer"]
    return OptimizerConfig(
        float(item["learning_rate"]),
        float(item["weight_decay"]),
        tuple(item["betas"]),
        float(item["eps"]),
        float(item["grad_clip"]),
        float(item.get("muon_momentum", 0.95)),
        bool(item.get("muon_nesterov", True)),
        int(item.get("muon_ns_steps", 5)),
        item.get("muon_adjust_lr_fn", "match_rms_adamw"),
    )


def prepare_plan(preregistration: Path, corpus_root: Path, device="cpu"):
    preregistration = Path(preregistration).resolve()
    spec = _load_mapping(preregistration)
    if spec.get("schema") != SCHEMA or spec.get("status") != "local_unissued":
        raise ValueError(f"preregistration schema/status must be {SCHEMA}/local_unissued")
    if device not in ("cpu", "cuda:0"):
        raise ValueError("device must be cpu or cuda:0")
    config = RestorationConfig.from_dict(spec["model"])
    loss_config = _loss_config(spec)
    numeric_query_guard = _numeric_query_guard(spec)
    seeds = spec.get("seeds")
    if not isinstance(seeds, dict) or set(seeds) != {"model", "order", "masks", "codes", "windows"}:
        raise ValueError("seeds must declare model, order, masks, codes and windows")
    for name, value in seeds.items():
        _integer(value, f"seeds.{name}", minimum=0)
    stages = spec.get("stages")
    if not isinstance(stages, list) or [stage.get("name") for stage in stages] != [
        "old120",
        "recent120_replay",
        "openml12_mixed",
    ]:
        raise ValueError("stages must declare old120, recent120_replay, openml12_mixed")
    tables = []
    digest_map = {}
    items = list(spec.get("tables", []))
    # The checked-in recipe can name the four frozen historical corpus roots
    # instead of repeating hundreds of file names.  Their preregistrations are
    # read only to obtain the already frozen dataset digests and row counts.
    if not items and isinstance(spec.get("corpora"), dict):
        for cohort, corpus_path in spec["corpora"].items():
            root = Path(corpus_path)
            if not root.is_absolute():
                root = Path(corpus_root) / root
            root = root.resolve()
            corpus_spec = _load_mapping(root / "preregistration.yaml")
            datasets = corpus_spec.get("datasets")
            expected_rows = corpus_spec.get("expected_rows", {})
            if not isinstance(datasets, dict):
                raise ValueError(f"{cohort}: frozen corpus has no datasets")
            for name, digest in datasets.items():
                items.append(
                    {
                        "name": f"{cohort}/{name}",
                        "cohort": cohort,
                        "kind": REAL if cohort in ("old3", "new9") else SYNTHETIC,
                        "windowed": cohort == "new9",
                        "path": str(root / "data" / f"{name}.json"),
                        "sha256": digest,
                        "expected_rows": expected_rows.get(name),
                    }
                )
    for item in items:
        path = Path(item["path"])
        if not path.is_absolute():
            path = Path(corpus_root) / path
        path = path.resolve()
        expected = item.get("sha256")
        if expected and _hash(path) != expected:
            raise ValueError(f"dataset digest mismatch: {item['name']}")
        table = load_table(
            path,
            item["name"],
            item["cohort"],
            item["kind"],
            windowed=bool(item.get("windowed", False)),
        )
        if item.get("expected_rows") is not None and len(table.values[0]) != item["expected_rows"]:
            # ``load_table`` stores the training split; this check catches a
            # stale corpus recipe without changing its frozen split.
            data = _load_mapping(path)
            if len(data["values"]) != item["expected_rows"]:
                raise ValueError(f"row count mismatch: {item['name']}")
        tables.append(table)
        digest_map[table.name] = _hash(path)
    if not tables:
        raise ValueError("curriculum requires at least one table")
    counts = Counter((table.cohort, table.kind) for table in tables)
    if counts[('old120', SYNTHETIC)] != 120 or counts[('recent120', SYNTHETIC)] != 120:
        raise ValueError("curriculum requires 120 old120 and 120 recent120 synthetic tables")
    if counts[('old3', REAL)] != 3 or counts[('new9', REAL)] != 9:
        raise ValueError("curriculum requires 3 old3 and 9 new9 real tables")
    for stage, cycle_size in zip(stages, (120, 150, 1200), strict=True):
        _positive(stage.get("max_seconds"), f"stage {stage['name']} max_seconds")
        updates = _integer(stage.get("max_updates"), f"stage {stage['name']} max_updates")
        if updates % cycle_size:
            raise ValueError(f"stage {stage['name']} update budget must contain complete cycles")
        if stage.get("cycle_updates", cycle_size) != cycle_size:
            raise ValueError(f"stage {stage['name']} cycle_updates mismatch")
    opt = _optimizer_config(spec)
    identity = _identity(
        dict(spec, model=config.as_dict()), preregistration, tables,
        {"source": _source_digest(), "datasets": digest_map},
    )
    summary = {
        "status": "local_unissued",
        "outcome": "planned",
        "execution_started": False,
        "stage_names": [stage["name"] for stage in stages],
        "table_count": len(tables),
        "model": config.as_dict(),
        "optimizer": spec["optimizer"],
        "objective": spec.get("objective", {"kind": "type_mean_all_observed"}),
        "numeric_query_guard": numeric_query_guard,
        "device": device,
        "total_seconds": sum(stage["max_seconds"] for stage in stages),
        "stages": [
            {key: value for key, value in stage.items() if key != "tables"} for stage in stages
        ],
    }
    return dict(
        spec,
        _stages=stages,
        _tables=tables,
        _config=config,
        _optimizer=opt,
        _loss_config=loss_config,
        _numeric_query_guard=numeric_query_guard,
        _identity=identity,
        _summary=summary,
    )


def run_curriculum_fit(args, observer=None):
    prereg = Path(args.preregistration)
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError("output-root must be a new attempt directory")
    plan = prepare_plan(prereg, Path(args.corpus_root), args.device)
    if not args.execute:
        return plan["_summary"] | {"identity": plan["_identity"]}
    _configure_runtime(args.device)
    output.mkdir(parents=True, exist_ok=False)
    seeds = plan.get(
        "seeds", {"model": 1729, "order": 1730, "masks": 1731, "codes": 1732, "windows": 1733}
    )
    torch.manual_seed(seeds["model"])
    model = RestorationModel(plan["_config"]).to(device=args.device, dtype=torch.float64)
    cfg = plan["_optimizer"]
    loss_config = plan["_loss_config"]
    numeric_query_guard = plan["_numeric_query_guard"]
    optimizer = adamw(model, cfg)
    identity = plan["_identity"]
    stage_index = update = cursor = 0
    stage_elapsed_prior = 0.0
    prior_elapsed = 0.0
    resume_cycle_losses = []
    resume_cycle_source_losses = {}
    resume = getattr(args, "resume_checkpoint", None)
    if resume:
        raw = torch.load(Path(resume), map_location="cpu", weights_only=True)
        if raw.get("schema") != CHECKPOINT_SCHEMA or raw.get("identity") != identity:
            raise ValueError("curriculum checkpoint identity drift")
        if raw.get("optimizer_kind") == "mixed":
            optimizer, _ = switch_to_muon(optimizer, model, cfg)
        state = _load_checkpoint(Path(resume), model, optimizer, identity)
        stage_index, update, cursor, stage_elapsed_prior, prior_elapsed = (
            state["stage_index"],
            state["update"],
            state["cursor"],
            state.get("stage_elapsed_seconds", 0.0),
            state["elapsed_seconds"],
        )
        resume_cycle_losses = [float(value) for value in state.get("cycle_losses", [])]
        resume_cycle_source_losses = {
            str(key): [float(value) for value in values]
            for key, values in state.get("cycle_source_losses", {}).items()
        }
    started = time.monotonic()
    receipt = {
        "schema": SCHEMA,
        "status": "local_unissued",
        "outcome": "started",
        "execution_started": True,
        "identity": identity,
        "stage": stage_index,
        "update": update,
        "elapsed_seconds": prior_elapsed,
    }
    if args.device == "cuda:0":
        receipt["cuda_device"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    signal_previous = signal.signal(
        signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    resume_stage_index, resume_cursor = stage_index, cursor

    def elapsed():
        return prior_elapsed + time.monotonic() - started

    def emit(event, **payload):
        if observer is not None:
            observer(
                {
                    "event": event,
                    "stage_index": stage_index,
                    "update": update,
                    "elapsed_seconds": elapsed(),
                    **payload,
                }
            )

    try:
        def _write(path, value):
            path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        _write(
            output / "resolved.json", {"preregistration": plan["_summary"], "identity": identity}
        )
        _write(output / "started.json", receipt)
        for index in range(stage_index, len(plan["_stages"])):
            stage = plan["_stages"][index]
            stage_index = index
            if index == 1 and not hasattr(optimizer, "muon"):
                optimizer, partition = switch_to_muon(optimizer, model, cfg)
                _write(output / "stage-2-optimizer-partition.json", partition)
            stage_start = time.monotonic()
            stage_budget = float(stage["max_seconds"])
            skip_updates = resume_cursor if index == resume_stage_index else 0
            stage_elapsed = stage_elapsed_prior if index == resume_stage_index else 0.0
            cycle_size = {"old120": 120, "recent120_replay": 150, "openml12_mixed": 1200}[
                stage["name"]
            ]
            cycle_losses = list(resume_cycle_losses) if index == resume_stage_index else []
            cycle_source_losses = (
                {key: list(values) for key, values in resume_cycle_source_losses.items()}
                if index == resume_stage_index else {}
            )
            resume_cycle_losses = []
            resume_cycle_source_losses = {}
            emit("phase", stage=stage["name"], config=plan["_summary"], identity=identity)
            tables = plan["_tables"]

            def stage_tables(stage_index=index, stage_tables_all=tables):
                if stage_index == 0:
                    return [item for item in stage_tables_all if item.cohort == "old120"]
                if stage_index == 1:
                    return [item for item in stage_tables_all
                            if item.cohort in ("old120", "recent120")]
                return stage_tables_all

            reserve = float(stage.get("final_reserve_seconds", 300))
            evaluation_deadline = stage_start + stage_budget - stage_elapsed_prior - reserve
            initial_metrics = evaluate(model, stage_tables(), stage, seeds,
                                       plan["_config"], args.device,
                                       masks=int(plan.get("evaluation_masks", 8)),
                                       deadline=evaluation_deadline, loss_config=loss_config,
                                       numeric_query_guard=numeric_query_guard)
            if not initial_metrics["complete"]:
                raise WallLimit(f"stage {stage['name']} initial evaluation exceeded budget")
            initial_metrics["stage"] = stage["name"]
            _write(output / f"{stage['name']}-initial-metrics.json", initial_metrics)
            emit("phase", stage=f"{stage['name']}_initial_complete", metrics=initial_metrics)
            stage_stop = False
            for step_index, (table, cycle, position) in enumerate(_schedule(stage, tables, seeds)):
                if step_index < skip_updates:
                    continue
                stage_elapsed = stage_elapsed_prior + time.monotonic() - stage_start
                if stage_elapsed >= stage_budget - float(stage.get("final_reserve_seconds", 300)):
                    # The stage's training budget is exhausted.  Leave the
                    # reserved interval for the fixed final evaluation and
                    # checkpoint, then continue to the next curriculum stage.
                    stage_stop = True
                    break
                episode, mask_info = _episode_for(
                    table,
                    _episode_index(stage["name"], cycle, position),
                    stage,
                    seeds,
                    plan["_config"],
                    args.device,
                    numeric_query_guard=numeric_query_guard,
                )
                update_started = time.monotonic()
                model.train()
                prepared = prepare_episode(model, *episode)
                optimizer.zero_grad(set_to_none=True)
                score = score_prepared_episode(model, prepared, loss_config=loss_config)
                if not bool(torch.isfinite(score.loss)):
                    raise FloatingPointError("nonfinite training loss")
                score.loss.backward()
                parameters = [p for p in model.parameters() if p.grad is not None]
                if not parameters or not all(
                    bool(torch.isfinite(p.grad).all()) for p in parameters
                ):
                    raise FloatingPointError("nonfinite training gradient")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, cfg.grad_clip, error_if_nonfinite=True
                )
                post_clip_norm = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm(
                        [parameter.grad.detach() for parameter in parameters], 2
                    ))
                )
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError("nonfinite model parameter")
                update += 1
                cursor = step_index + 1
                cycle_losses.append(float(score.loss.detach().cpu()))
                cycle_source_losses.setdefault(table.cohort, []).append(cycle_losses[-1])
                row = {
                    "stage": stage["name"],
                    "cycle": cycle,
                    "update": update,
                    "table": table.name,
                    "loss": cycle_losses[-1],
                    **_loss_terms(score, prepared, loss_config),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "post_clip_gradient_norm": float(post_clip_norm.detach().cpu()),
                    "mask": mask_info,
                    "update_seconds": time.monotonic() - update_started,
                    "elapsed_seconds": elapsed(),
                }
                if args.device == "cuda:0":
                    row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                with (output / "updates.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                emit("update", **row)
                if (position + 1) % cycle_size == 0:
                    summary = {
                        "stage": stage["name"],
                        "cycle": cycle + 1,
                        "completed_round": cycle + 1,
                        "loss": _distribution(cycle_losses),
                        "loss_by_source": {
                            source: _distribution(values)
                            for source, values in sorted(cycle_source_losses.items())
                        },
                        "complete": True,
                        "total_tables": cycle_size,
                        "completed_tables": cycle_size,
                    }
                    with (output / "cycle-metrics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(summary, allow_nan=False) + "\n")
                    emit(
                        "round_summary",
                        training_round=cycle + 1,
                        train_round={
                            "loss": summary["loss"],
                            "loss_by_source": summary["loss_by_source"],
                            "complete": True,
                            "total_tables": cycle_size,
                            "completed_tables": cycle_size,
                        },
                    )
                    every = int(stage.get("evaluate_every_cycles", 1))
                    if (cycle + 1) % every == 0:
                        periodic = evaluate(model, stage_tables(), stage, seeds,
                                             plan["_config"], args.device,
                                             masks=int(plan.get("evaluation_masks", 8)),
                                             deadline=evaluation_deadline, loss_config=loss_config,
                                             numeric_query_guard=numeric_query_guard)
                        if not periodic["complete"]:
                            # A periodic evaluation that reaches the training
                            # cutoff ends updates for this stage; the final
                            # evaluation below gets the remaining reserved
                            # interval and decides whether the chain can move
                            # on safely.
                            stage_stop = True
                        else:
                            periodic["stage"] = stage["name"]
                            periodic["cycle"] = cycle + 1
                            _write(output / f"{stage['name']}-cycle-{cycle + 1:04d}-metrics.json",
                                   periodic)
                            emit("phase", stage=f"{stage['name']}_cycle_{cycle + 1}_complete",
                                 metrics=periodic)
                    cycle_losses = []
                    cycle_source_losses = {}
                _checkpoint(
                    output / "checkpoint-progress.pt",
                    model,
                    optimizer,
                    identity,
                    stage_index,
                    update,
                    step_index + 1,
                    elapsed(),
                    stage_name=stage["name"],
                    stage_elapsed=stage_elapsed,
                    cycle_losses=cycle_losses,
                    cycle_source_losses=cycle_source_losses,
                )
                if stage_stop:
                    break
            stage_eval = evaluate(
                model,
                stage_tables(),
                stage,
                seeds,
                plan["_config"],
                args.device,
                masks=int(plan.get("evaluation_masks", 8)),
                deadline=stage_start + stage_budget - stage_elapsed_prior,
                loss_config=loss_config,
                numeric_query_guard=numeric_query_guard,
            )
            if not stage_eval["complete"]:
                raise WallLimit(f"stage {stage['name']} final evaluation exceeded budget")
            stage_eval["stage"] = stage["name"]
            _write(output / f"{stage['name']}-final-metrics.json", stage_eval)
            _checkpoint(
                output / "checkpoint-progress.pt",
                model,
                optimizer,
                identity,
                stage_index + 1,
                update,
                0,
                elapsed(),
                stage_name=stage["name"],
                stage_elapsed=0.0,
                cycle_losses=(),
                cycle_source_losses={},
            )
            stage_elapsed_prior = 0.0
            resume_stage_index, resume_cursor = index + 1, 0
            resume = None
        stage_index = len(plan["_stages"])
        cursor = 0
        receipt["outcome"] = "completed"
    except WallLimit as error:
        receipt.update(outcome="wall_limit", error_type=type(error).__name__, error=str(error))
    except KeyboardInterrupt as error:
        receipt.update(outcome="interrupted", error_type=type(error).__name__)
    except Exception as error:
        receipt.update(outcome="failed", error_type=type(error).__name__, error=str(error))
    finally:
        signal.signal(signal.SIGTERM, signal_previous)
        receipt.update(stage=stage_index, update=update, elapsed_seconds=elapsed())
        _checkpoint(
            output / "checkpoint.pt",
            model,
            optimizer,
            identity,
            stage_index,
            update,
            cursor,
            elapsed(),
            stage_name=plan["_stages"][min(stage_index, 2)]["name"],
            stage_elapsed=stage_elapsed_prior,
        )
        (output / "terminal.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        emit("summary", metrics=receipt)
    return receipt


__all__ = ["CHECKPOINT_SCHEMA", "SCHEMA", "CurriculumTable", "prepare_plan", "run_curriculum_fit"]
