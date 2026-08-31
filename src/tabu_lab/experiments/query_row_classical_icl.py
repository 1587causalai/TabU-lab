"""Compare TabUR frozen ICL with per-world MLP and XGBoost baselines."""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from tabu_lab.contracts import canonical_hash

from .query_row_frozen_icl import _state_hash
from .query_row_icl_threshold import (
    _context_standardization,
    _linear_regression_mse,
    _model_mse,
)
from .query_row_identity import query_row_result_identity
from .query_row_pretraining import (
    QueryRowPretrainingResult,
    train_query_row_synthetic_pretraining_model,
)
from .query_row_synthetic_fit import (
    QueryRowSyntheticEpisode,
    make_query_row_synthetic_episode,
)
from .tabubase_scale import resolve_device

CLASSICAL_ICL_BASELINE_IDS = (
    "ordinary_least_squares.context_only.v1",
    "mlp_regressor.context_only.v1",
    "xgboost_regressor.context_only.v1",
)
CLASSICAL_ICL_CONFIG = {
    "metric": "context_standardized_target_mse",
    "fit_scope": "context_rows_only",
    "features": "all_other_visible_columns",
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


def _query_design(
    episode: QueryRowSyntheticEpisode,
    *,
    context_rows: int,
    feature: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = episode.evidence.forward_values.detach().cpu().numpy().astype(np.float32)
    target = episode.sidecar.target_mask.detach().cpu().numpy().astype(bool)
    rows, features = values.shape
    if context_rows < 1 or context_rows >= rows:
        raise ValueError("context_rows must be in [1, rows-1]")
    if bool(target[:context_rows].any()):
        raise ValueError("classical ICL context rows must contain no target cells")
    predictors = [index for index in range(features) if index != feature]
    mean, _ = _context_standardization(episode)
    mean_np = mean.detach().cpu().numpy().astype(np.float32)
    train_x = values[:context_rows, predictors]
    train_y = values[:context_rows, feature]
    query_x = values[context_rows:, predictors].copy()
    query_target = target[context_rows:, predictors]
    query_x[query_target] = np.broadcast_to(mean_np[predictors], query_x.shape)[query_target]
    query_target_feature = target[context_rows:, feature]
    return train_x, train_y, query_x, query_target_feature


def _fit_predictions(
    episode: QueryRowSyntheticEpisode,
    *,
    context_rows: int,
    estimator: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one classical estimator per target feature on context rows only."""

    if estimator not in {"mlp", "xgboost"}:
        raise ValueError(f"unsupported classical estimator: {estimator!r}")
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("MLP/XGBoost baselines require scikit-learn") from exc
    xgb: Any = None
    if estimator == "xgboost":
        try:
            import xgboost as xgb_module
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("XGBoost baseline requires xgboost") from exc
        xgb = xgb_module

    values = episode.evidence.forward_values.detach().cpu().to(torch.float32)
    features = values.shape[1]
    prediction = torch.zeros_like(values)
    for feature in range(features):
        train_x, train_y, query_x, _ = _query_design(
            episode,
            context_rows=context_rows,
            feature=feature,
        )
        if estimator == "mlp":
            # Scale predictors only; the scoring scale remains the shared
            # context standardization used by TabUR and the OLS baseline.
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_x)
            query_scaled = scaler.transform(query_x)
            model = MLPRegressor(
                hidden_layer_sizes=tuple(CLASSICAL_ICL_CONFIG["mlp"]["hidden_layer_sizes"]),
                activation=str(CLASSICAL_ICL_CONFIG["mlp"]["activation"]),
                solver=str(CLASSICAL_ICL_CONFIG["mlp"]["solver"]),
                alpha=float(CLASSICAL_ICL_CONFIG["mlp"]["alpha"]),
                learning_rate_init=float(CLASSICAL_ICL_CONFIG["mlp"]["learning_rate_init"]),
                max_iter=int(CLASSICAL_ICL_CONFIG["mlp"]["max_iter"]),
                early_stopping=bool(CLASSICAL_ICL_CONFIG["mlp"]["early_stopping"]),
                shuffle=bool(CLASSICAL_ICL_CONFIG["mlp"]["shuffle"]),
                tol=float(CLASSICAL_ICL_CONFIG["mlp"]["tol"]),
                random_state=seed + feature,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(train_scaled, train_y)
            predicted = model.predict(query_scaled)
        else:
            model = xgb.XGBRegressor(
                **CLASSICAL_ICL_CONFIG["xgboost"],
                random_state=seed + feature,
            )
            model.fit(train_x, train_y, verbose=False)
            predicted = model.predict(query_x)
        prediction[context_rows:, feature] = torch.as_tensor(predicted, dtype=torch.float32)
    mean, scale = _context_standardization(episode)
    return prediction, scale


def _classical_mse(
    episode: QueryRowSyntheticEpisode,
    *,
    context_rows: int,
    estimator: str,
    seed: int,
) -> tuple[float, int]:
    prediction, scale = _fit_predictions(
        episode,
        context_rows=context_rows,
        estimator=estimator,
        seed=seed,
    )
    target = episode.sidecar.target_mask.detach().cpu().to(torch.bool)
    truth = episode.sidecar.target_values.detach().cpu().to(torch.float32)
    scored = target & torch.arange(target.shape[0]).view(-1, 1).ge(context_rows)
    error = (prediction - truth) / scale.view(1, -1).clamp_min(1.0e-8)
    values = error.square()[scored]
    if not values.numel():
        raise RuntimeError("classical ICL episode has no query target cells")
    return float(values.mean().item()), int(values.numel())


@dataclass(frozen=True, slots=True)
class QueryRowClassicalICLRecord:
    world_id: str
    world_family: str
    context_rows: int
    target_count: int
    tabur_mse: float
    linear_regression_mse: float
    mlp_mse: float
    xgboost_mse: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryRowClassicalICLResult:
    status: str
    threshold_met: bool
    evidence_status: str
    claim_boundary: str
    baseline_ids: tuple[str, ...]
    baseline_config_hash: str
    model_id: str
    contract_version: str
    model_spec_hash: str
    profile_id: str
    variant_hash: str
    row_readout_mode: str
    row_readout_identity: dict[str, Any]
    device: str
    seed: int
    pretrain_rows: int
    pretrain_worlds: int
    pretrain_steps: int
    eval_rows: int
    eval_worlds: int
    context_rows: tuple[int, ...]
    tabur_mse: float
    linear_regression_mse: float
    mlp_mse: float
    xgboost_mse: float
    tabur_vs_mlp_ratio: float
    tabur_vs_xgboost_ratio: float
    parameter_hash_before: str
    parameter_hash_after: str
    parameter_hash_unchanged: bool
    pretraining: QueryRowPretrainingResult
    records: tuple[QueryRowClassicalICLRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pretraining"] = self.pretraining.as_dict()
        payload["records"] = [record.as_dict() for record in self.records]
        payload["baseline_ids"] = list(self.baseline_ids)
        payload["context_rows"] = list(self.context_rows)
        return payload


def run_query_row_classical_icl_benchmark(
    *,
    seed: int = 1729,
    pretrain_rows: int = 64,
    pretrain_worlds: int = 1024,
    pretrain_steps: int = 3000,
    eval_rows: int = 64,
    eval_worlds: int = 12,
    context_rows: tuple[int, ...] = (8, 16, 32),
    row_token_count: int = 4,
    learning_rate: float = 1.0e-2,
    device: str | torch.device = "cpu",
    checkpoint: Any = None,
) -> QueryRowClassicalICLResult:
    """Run a matched synthetic ICL comparison against MLP and XGBoost.

    Each classical model is fit independently for every held-out world and
    target feature using only visible context rows.  No target-row values or
    TruthSidecar entries are used as training inputs.
    """

    if pretrain_rows < 3 or pretrain_worlds <= 0 or pretrain_steps <= 0:
        raise ValueError("pretrain rows, worlds and steps must be positive")
    if eval_rows < 3 or eval_worlds <= 0:
        raise ValueError("eval rows and worlds must be positive")
    if not context_rows or any(size < 1 or size >= eval_rows for size in context_rows):
        raise ValueError("context_rows must be non-empty and smaller than eval_rows")
    resolved_device = resolve_device(str(device))
    model, pretraining = train_query_row_synthetic_pretraining_model(
        profile="completion.artificial_mask.v1",
        seed=seed,
        rows=pretrain_rows,
        worlds=pretrain_worlds,
        steps=pretrain_steps,
        learning_rate=learning_rate,
        row_token_count=row_token_count,
        device=resolved_device,
        output=checkpoint,
    )
    model.eval()
    families = ("row_latent_linear", "row_latent_periodic", "row_latent_polynomial")
    parameter_hash_before = _state_hash(model)
    records: list[QueryRowClassicalICLRecord] = []
    weighted = {name: 0.0 for name in ("tabur", "linear", "mlp", "xgboost")}
    total_targets = 0
    with torch.no_grad():
        for world_index in range(eval_worlds):
            family = families[world_index % len(families)]
            for context_size in context_rows:
                episode = make_query_row_synthetic_episode(
                    seed=seed + 100_000 + world_index * 101,
                    rows=eval_rows,
                    row_token_count=row_token_count,
                    context_rows=context_size,
                    world_id=f"heldout-classical-icl-{world_index}-{context_size}",
                    world_family=family,
                )
                tabur_mse, tabur_count = _model_mse(model, episode)
                linear_mse, linear_count = _linear_regression_mse(
                    episode,
                    context_rows=context_size,
                )
                mlp_mse, mlp_count = _classical_mse(
                    episode,
                    context_rows=context_size,
                    estimator="mlp",
                    seed=seed + world_index * 101 + context_size,
                )
                xgboost_mse, xgboost_count = _classical_mse(
                    episode,
                    context_rows=context_size,
                    estimator="xgboost",
                    seed=seed + world_index * 101 + context_size,
                )
                counts = (tabur_count, linear_count, mlp_count, xgboost_count)
                if len(set(counts)) != 1 or mlp_count != xgboost_count:
                    raise RuntimeError(
                        "classical and TabUR baselines scored different target counts"
                    )
                records.append(
                    QueryRowClassicalICLRecord(
                        world_id=episode.world_id,
                        world_family=family,
                        context_rows=context_size,
                        target_count=tabur_count,
                        tabur_mse=tabur_mse,
                        linear_regression_mse=linear_mse,
                        mlp_mse=mlp_mse,
                        xgboost_mse=xgboost_mse,
                    )
                )
                for name, value in (
                    ("tabur", tabur_mse),
                    ("linear", linear_mse),
                    ("mlp", mlp_mse),
                    ("xgboost", xgboost_mse),
                ):
                    weighted[name] += value * tabur_count
                total_targets += tabur_count
    parameter_hash_after = _state_hash(model)
    metrics = {name: weighted[name] / total_targets for name in weighted}
    threshold_met = (
        parameter_hash_before == parameter_hash_after
        and all(math.isfinite(value) for value in metrics.values())
        and metrics["tabur"] <= metrics["mlp"]
        and metrics["tabur"] <= metrics["xgboost"]
    )
    result_identity = query_row_result_identity(model.checkpoint_identity())
    return QueryRowClassicalICLResult(
        status="pass" if threshold_met else "continue",
        threshold_met=threshold_met,
        evidence_status="local_unissued",
        claim_boundary=(
            "matched synthetic frozen-ICL diagnostic versus per-world MLP/XGBoost; "
            "no real-data transfer, formal receipt, benchmark, or accepted capability claim"
        ),
        baseline_ids=CLASSICAL_ICL_BASELINE_IDS,
        baseline_config_hash=canonical_hash(CLASSICAL_ICL_CONFIG),
        **result_identity,
        device=str(resolved_device),
        seed=seed,
        pretrain_rows=pretrain_rows,
        pretrain_worlds=pretrain_worlds,
        pretrain_steps=pretrain_steps,
        eval_rows=eval_rows,
        eval_worlds=eval_worlds,
        context_rows=context_rows,
        tabur_mse=metrics["tabur"],
        linear_regression_mse=metrics["linear"],
        mlp_mse=metrics["mlp"],
        xgboost_mse=metrics["xgboost"],
        tabur_vs_mlp_ratio=metrics["tabur"] / max(metrics["mlp"], 1.0e-8),
        tabur_vs_xgboost_ratio=metrics["tabur"] / max(metrics["xgboost"], 1.0e-8),
        parameter_hash_before=parameter_hash_before,
        parameter_hash_after=parameter_hash_after,
        parameter_hash_unchanged=parameter_hash_before == parameter_hash_after,
        pretraining=pretraining,
        records=tuple(records),
    )


__all__ = [
    "CLASSICAL_ICL_BASELINE_IDS",
    "CLASSICAL_ICL_CONFIG",
    "QueryRowClassicalICLRecord",
    "QueryRowClassicalICLResult",
    "run_query_row_classical_icl_benchmark",
]
