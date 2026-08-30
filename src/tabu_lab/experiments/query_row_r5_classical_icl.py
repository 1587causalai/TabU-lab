"""Matched frozen ICL comparison for the R5 diverse supervised synthetic prior.

This runner loads profile-bound R5 checkpoints without constructing an optimizer,
then evaluates them and context-only classical baselines on one frozen v2
validation panel.  The panel is generated once and reused for every checkpoint
and every baseline, so the comparison is descriptive rather than a collection
of independently selected test worlds.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabu_lab.contracts import canonical_hash
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig

from .query_row_pretraining import load_query_row_pretrain_checkpoint
from .query_row_supervised_synthetic_v2 import (
    GENERATOR_ID,
    build_query_row_supervised_synthetic_v2_plan,
    make_query_row_supervised_synthetic_v2_episode,
    substitute_query_truth,
)
from .tabubase_scale import resolve_device

CLASSICAL_R5_BASELINE_IDS = (
    "ordinary_least_squares.context_only.v1",
    "mlp_regressor.context_only.v1",
    "xgboost_regressor.context_only.v1",
)

CLASSICAL_R5_BASELINE_CONFIG: dict[str, Any] = {
    "metric": "raw_response_mse",
    "diagnostic_metric": "context_standardized_response_mse",
    "fit_scope": "context_rows_only",
    "features": "all_visible_predictor_columns",
    "missing_predictor_policy": "context_mean",
    "linear": {"ridge": 1.0e-4},
    "mlp": {
        "hidden_layer_sizes": (64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1.0e-4,
        "learning_rate_init": 1.0e-3,
        "max_iter": 500,
        "early_stopping": False,
        "shuffle": True,
        "tol": 1.0e-4,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 8,
        "tree_method": "hist",
        "objective": "reg:squarederror",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _context_mean_scale(episode: Any) -> tuple[float, float]:
    values = episode.evidence.forward_values.detach().cpu().to(torch.float32)
    response = values[: episode.context_rows, episode.width]
    mean = float(response.mean().item())
    scale = float(response.std(unbiased=False).clamp_min(1.0e-6).item())
    return mean, scale


def _design(episode: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    values = episode.evidence.forward_values.detach().cpu().to(torch.float32)
    target = episode.sidecar.target_mask.detach().cpu().to(torch.bool)
    context = episode.context_rows
    width = episode.width
    response_column = width
    if values.shape[1] != width + 1:
        raise RuntimeError("v2 episode width does not match evidence feature count")
    if bool(target[:context].any()):
        raise RuntimeError("v2 context rows unexpectedly contain target cells")
    train_x = values[:context, :width].numpy().astype(np.float32, copy=True)
    train_y = values[:context, response_column].numpy().astype(np.float32, copy=True)
    query_x = values[context:, :width].numpy().astype(np.float32, copy=True)
    query_target = target[context:, :width].numpy().astype(bool, copy=False)
    if query_target.any():
        mean = values[:context, :width].mean(dim=0).numpy().astype(np.float32)
        query_x[query_target] = np.broadcast_to(mean, query_x.shape)[query_target]
    query_target_response = target[context:, response_column].numpy().astype(bool, copy=True)
    response_mean, response_scale = _context_mean_scale(episode)
    return train_x, train_y, query_x, query_target_response, response_mean, response_scale


def _linear_prediction(episode: Any) -> tuple[np.ndarray, float, float]:
    train_x, train_y, query_x, _, response_mean, response_scale = _design(episode)
    design = np.concatenate((np.ones((train_x.shape[0], 1), dtype=np.float32), train_x), axis=1)
    ridge = float(CLASSICAL_R5_BASELINE_CONFIG["linear"]["ridge"])
    regularizer = np.eye(design.shape[1], dtype=np.float32) * ridge
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ train_y)
    query_design = np.concatenate(
        (np.ones((query_x.shape[0], 1), dtype=np.float32), query_x), axis=1
    )
    return query_design @ coefficients, response_mean, response_scale


def _mlp_prediction(episode: Any, *, seed: int) -> tuple[np.ndarray, float, float]:
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("MLP baseline requires scikit-learn") from exc
    train_x, train_y, query_x, _, response_mean, response_scale = _design(episode)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    query_scaled = scaler.transform(query_x)
    config = CLASSICAL_R5_BASELINE_CONFIG["mlp"]
    estimator = MLPRegressor(
        hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
        activation=str(config["activation"]),
        solver=str(config["solver"]),
        alpha=float(config["alpha"]),
        learning_rate_init=float(config["learning_rate_init"]),
        max_iter=int(config["max_iter"]),
        early_stopping=bool(config["early_stopping"]),
        shuffle=bool(config["shuffle"]),
        tol=float(config["tol"]),
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        estimator.fit(train_scaled, train_y)
    return estimator.predict(query_scaled).astype(np.float32), response_mean, response_scale


def _xgboost_prediction(episode: Any, *, seed: int) -> tuple[np.ndarray, float, float]:
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("XGBoost baseline requires xgboost") from exc
    train_x, train_y, query_x, _, response_mean, response_scale = _design(episode)
    estimator = xgb.XGBRegressor(
        **CLASSICAL_R5_BASELINE_CONFIG["xgboost"],
        random_state=seed,
    )
    estimator.fit(train_x, train_y, verbose=False)
    return estimator.predict(query_x).astype(np.float32), response_mean, response_scale


def _mse(prediction: np.ndarray, truth: np.ndarray, scale: float) -> tuple[float, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    raw = float(np.mean(np.square(error)))
    standardized = float(np.mean(np.square(error / max(scale, 1.0e-8))))
    if not math.isfinite(raw) or not math.isfinite(standardized):
        raise RuntimeError("non-finite classical baseline metric")
    return raw, standardized


def _baseline_world(episode: Any, *, seed: int) -> dict[str, Any]:
    _, _, _, target, _, response_scale = _design(episode)
    truth = (
        episode.sidecar.target_values[episode.context_rows :, episode.width]
        .detach()
        .cpu()
        .to(torch.float32)
        .numpy()
    )
    if target.shape != truth.shape or not bool(target.all()):
        raise RuntimeError("v2 response target mask is not contiguous query-only supervision")
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, function in (
        ("linear", _linear_prediction),
        ("mlp", lambda item: _mlp_prediction(item, seed=seed)),
        ("xgboost", lambda item: _xgboost_prediction(item, seed=seed)),
    ):
        prediction, _, scale = function(episode)
        predictions[name] = prediction
        raw, standardized = _mse(prediction, truth, scale)
        metrics[name] = {"raw_response_mse": raw, "context_standardized_response_mse": standardized}
    return {
        "target_count": int(target.sum()),
        "truth": truth,
        "response_scale": response_scale,
        "predictions": predictions,
        "metrics": metrics,
    }


def _tabur_world(model: torch.nn.Module, episode: Any) -> tuple[dict[str, float], bool]:
    with torch.no_grad():
        output = model(episode.evidence)
        raw = output["numeric_raw_prediction"]
        support = output["numeric_support_available"].to(torch.bool)
        substituted = substitute_query_truth(episode, value=123.456)
        substituted_output = model(substituted.evidence)
    if raw.ndim == 3:
        raw = raw.squeeze(0)
        support = support.squeeze(0)
    if substituted_output["numeric_raw_prediction"].ndim == 3:
        substituted_raw = substituted_output["numeric_raw_prediction"].squeeze(0)
    else:
        substituted_raw = substituted_output["numeric_raw_prediction"]
    target = episode.sidecar.target_mask.to(device=raw.device, dtype=torch.bool)
    target = target[episode.context_rows :, episode.width]
    prediction = raw[episode.context_rows :, episode.width]
    supported = support[episode.context_rows :, episode.width]
    if not bool(target.all()) or not bool(supported[target].all()):
        raise RuntimeError("TabUR checkpoint has unsupported v2 response targets")
    truth = episode.sidecar.target_values[episode.context_rows :, episode.width].to(
        device=raw.device, dtype=torch.float32
    )
    _, scale = _context_mean_scale(episode)
    raw_mse, standardized_mse = _mse(
        prediction.detach().cpu().numpy(), truth.detach().cpu().numpy(), scale
    )
    same_prediction = torch.equal(
        prediction.detach().cpu(),
        substituted_raw[episode.context_rows :, episode.width].detach().cpu(),
    )
    return {
        "raw_response_mse": raw_mse,
        "context_standardized_response_mse": standardized_mse,
    }, same_prediction


@dataclass(frozen=True, slots=True)
class QueryRowR5ClassicalRecord:
    world_id: str
    checkpoint_rung: str
    checkpoint_seed: int
    family: str
    width: int
    predictor_regime: str
    noise_level: str
    context_rows: int
    target_count: int
    tabur_raw_response_mse: float
    linear_raw_response_mse: float
    mlp_raw_response_mse: float
    xgboost_raw_response_mse: float
    tabur_context_standardized_response_mse: float
    linear_context_standardized_response_mse: float
    mlp_context_standardized_response_mse: float
    xgboost_context_standardized_response_mse: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryRowR5ClassicalCheckpointResult:
    checkpoint: str
    checkpoint_sha256: str
    identity: str
    identity_sha256: str
    rung: str
    root_seed: int
    worlds: int
    updates: int
    learning_rate: float
    parameter_hash_before: str
    parameter_hash_after: str
    parameter_hash_unchanged: bool
    truth_substitution_prediction_unchanged: bool
    optimizer_created: bool
    parameter_update_attempted: bool
    aggregate_metrics: dict[str, dict[str, float]]
    records: tuple[QueryRowR5ClassicalRecord, ...]
    status: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [record.as_dict() for record in self.records]
        return payload


@dataclass(frozen=True, slots=True)
class QueryRowR5ClassicalResult:
    schema_version: str
    generator_id: str
    evidence_status: str
    claim_boundary: str
    device: str
    panel_root_seed: int
    panel_worlds: int
    panel_plan_hash: str
    baseline_ids: tuple[str, ...]
    baseline_config_hash: str
    checkpoints: tuple[QueryRowR5ClassicalCheckpointResult, ...]
    status: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["baseline_ids"] = list(self.baseline_ids)
        payload["checkpoints"] = [checkpoint.as_dict() for checkpoint in self.checkpoints]
        return payload


def _checkpoint_model(
    path: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    identity_path = path.with_suffix(".identity.json")
    if not identity_path.is_file():
        raise FileNotFoundError(f"checkpoint identity sidecar is required: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    model_identity = identity.get("model_identity", {})
    reference = model_identity.get("reference_config", {})
    config = ReferenceConfig(
        d_model=int(reference["d_model"]),
        n_heads=int(reference["n_heads"]),
        d_ff=int(reference["d_ff"]),
        n_blocks=int(reference["n_blocks"]),
        inducing_slots=int(reference["inducing_slots"]),
        matched_slots=int(reference["matched_slots"]),
        max_features=int(reference["max_features"]),
    )
    model = build_model(
        "tabu.query.row",
        config=config,
        profile=str(model_identity["profile_id"]),
        row_token_count=int(model_identity["row_token_count"]),
    ).to(device)
    load_query_row_pretrain_checkpoint(model, path)
    return model, identity


def _aggregate(
    records: tuple[QueryRowR5ClassicalRecord, ...],
) -> dict[str, dict[str, float]]:
    names = ("tabur", "linear", "mlp", "xgboost")
    metrics = ("raw_response_mse", "context_standardized_response_mse")
    weighted = {name: {metric: 0.0 for metric in metrics} for name in names}
    total = 0
    for record in records:
        count = record.target_count
        total += count
        for name in names:
            for metric in metrics:
                weighted[name][metric] += (
                    getattr(record, f"{name}_{metric}") * count
                )
    if total <= 0:
        raise RuntimeError("R5 classical panel has no target cells")
    return {
        name: {metric: value / total for metric, value in values.items()}
        for name, values in weighted.items()
    }


def run_query_row_r5_classical_icl(
    *,
    checkpoints: tuple[Path, ...],
    panel_root_seed: int = 502729,
    panel_worlds: int = 512,
    device: str | torch.device = "cuda",
) -> QueryRowR5ClassicalResult:
    """Compare frozen R5 checkpoints with context-only linear/MLP/XGBoost fits."""

    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if panel_worlds <= 0:
        raise ValueError("panel_worlds must be positive")
    resolved_device = resolve_device(str(device))
    plan = build_query_row_supervised_synthetic_v2_plan(
        root_seed=panel_root_seed,
        worlds=panel_worlds,
        partition="validation",
    )
    panel_hash = canonical_hash(plan)
    baseline_cache: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(plan):
        episode = make_query_row_supervised_synthetic_v2_episode(
            root_seed=panel_root_seed,
            world_id=spec["world_id"],
            partition="validation",
            width=spec["width"],
            family=spec["family"],
            predictor_regime=spec["predictor_regime"],
            noise_level=spec["noise_level"],
            context_rows=spec["context_rows"],
        )
        baseline_cache[episode.world_id] = _baseline_world(episode, seed=panel_root_seed + index)

    checkpoint_results: list[QueryRowR5ClassicalCheckpointResult] = []
    for checkpoint_path in checkpoints:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
        model, identity = _checkpoint_model(checkpoint_path, device=resolved_device)
        metadata = identity["metadata"]
        parameter_hash_before = _state_hash(model)
        records: list[QueryRowR5ClassicalRecord] = []
        substitution_ok = True
        for spec in plan:
            episode = make_query_row_supervised_synthetic_v2_episode(
                root_seed=panel_root_seed,
                world_id=spec["world_id"],
                partition="validation",
                width=spec["width"],
                family=spec["family"],
                predictor_regime=spec["predictor_regime"],
                noise_level=spec["noise_level"],
                context_rows=spec["context_rows"],
            )
            baseline = baseline_cache[episode.world_id]
            tabur, unchanged = _tabur_world(model, episode)
            substitution_ok = substitution_ok and unchanged
            metrics = baseline["metrics"]
            records.append(
                QueryRowR5ClassicalRecord(
                    world_id=episode.world_id,
                    checkpoint_rung=str(metadata["rung"]),
                    checkpoint_seed=int(metadata["root_seed"]),
                    family=episode.family,
                    width=episode.width,
                    predictor_regime=episode.predictor_regime,
                    noise_level=episode.noise_level,
                    context_rows=episode.context_rows,
                    target_count=int(baseline["target_count"]),
                    tabur_raw_response_mse=tabur["raw_response_mse"],
                    linear_raw_response_mse=metrics["linear"]["raw_response_mse"],
                    mlp_raw_response_mse=metrics["mlp"]["raw_response_mse"],
                    xgboost_raw_response_mse=metrics["xgboost"]["raw_response_mse"],
                    tabur_context_standardized_response_mse=tabur[
                        "context_standardized_response_mse"
                    ],
                    linear_context_standardized_response_mse=metrics["linear"][
                        "context_standardized_response_mse"
                    ],
                    mlp_context_standardized_response_mse=metrics["mlp"][
                        "context_standardized_response_mse"
                    ],
                    xgboost_context_standardized_response_mse=metrics["xgboost"][
                        "context_standardized_response_mse"
                    ],
                )
            )
        frozen_records = tuple(records)
        parameter_hash_after = _state_hash(model)
        identity_path = checkpoint_path.with_suffix(".identity.json")
        checkpoint_result = QueryRowR5ClassicalCheckpointResult(
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=_sha256(checkpoint_path),
            identity=str(identity_path),
            identity_sha256=_sha256(identity_path),
            rung=str(metadata["rung"]),
            root_seed=int(metadata["root_seed"]),
            worlds=int(metadata["worlds"]),
            updates=int(metadata["updates"]),
            learning_rate=float(metadata["learning_rate"]),
            parameter_hash_before=parameter_hash_before,
            parameter_hash_after=parameter_hash_after,
            parameter_hash_unchanged=parameter_hash_before == parameter_hash_after,
            truth_substitution_prediction_unchanged=substitution_ok,
            optimizer_created=False,
            parameter_update_attempted=False,
            aggregate_metrics=_aggregate(frozen_records),
            records=frozen_records,
            status=(
                "passed"
                if parameter_hash_before == parameter_hash_after and substitution_ok
                else "failed"
            ),
        )
        checkpoint_results.append(checkpoint_result)

    return QueryRowR5ClassicalResult(
        schema_version="tabu.query-row.r5-v2-frozen-classical-panel.v1",
        generator_id=GENERATOR_ID,
        evidence_status="local_unissued",
        claim_boundary=(
            "Matched v2 synthetic held-out frozen-ICL diagnostic versus context-only "
            "linear, MLP, and XGBoost fits; no formal receipt, real-data transfer, "
            "or accepted capability claim"
        ),
        device=str(resolved_device),
        panel_root_seed=panel_root_seed,
        panel_worlds=panel_worlds,
        panel_plan_hash=panel_hash,
        baseline_ids=CLASSICAL_R5_BASELINE_IDS,
        baseline_config_hash=canonical_hash(CLASSICAL_R5_BASELINE_CONFIG),
        checkpoints=tuple(checkpoint_results),
        status=(
            "passed"
            if all(item.status == "passed" for item in checkpoint_results)
            else "failed"
        ),
    )


__all__ = [
    "CLASSICAL_R5_BASELINE_CONFIG",
    "CLASSICAL_R5_BASELINE_IDS",
    "QueryRowR5ClassicalCheckpointResult",
    "QueryRowR5ClassicalRecord",
    "QueryRowR5ClassicalResult",
    "run_query_row_r5_classical_icl",
]
