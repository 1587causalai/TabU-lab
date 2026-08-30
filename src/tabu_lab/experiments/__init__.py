"""Small, reproducible evaluation runners."""

from .query_row_classical_icl import (
    CLASSICAL_ICL_BASELINE_IDS,
    CLASSICAL_ICL_CONFIG,
    QueryRowClassicalICLRecord,
    QueryRowClassicalICLResult,
    run_query_row_classical_icl_benchmark,
)
from .query_row_finetune_lift import (
    QueryRowFinetuneLiftRecord,
    QueryRowFinetuneLiftResult,
    run_query_row_finetune_lift,
)
from .query_row_frozen_icl import (
    QueryRowFrozenICLRecord,
    QueryRowFrozenICLResult,
    run_query_row_frozen_icl,
)
from .query_row_icl_threshold import (
    LINEAR_REGRESSION_BASELINE_ID,
    LINEAR_REGRESSION_BASELINE_SPEC,
    QueryRowLinearICLContextSummary,
    QueryRowLinearICLRecord,
    QueryRowLinearICLThresholdResult,
    run_query_row_linear_icl_threshold,
)
from .query_row_pretraining import (
    QueryRowPretrainingResult,
    load_query_row_pretrain_checkpoint,
    run_query_row_synthetic_pretraining,
    save_query_row_pretrain_checkpoint,
    train_query_row_synthetic_pretraining_model,
)
from .query_row_real_benchmark import (
    QueryRowRealBenchmarkResult,
    QueryRowRealDatasetResult,
    run_query_row_real_scratch_benchmark,
)
from .query_row_real_coordinates import (
    numeric_raw_prediction,
    numeric_raw_prediction_from_public,
    query_row_real_regression_loss,
    task_scale_to_raw,
)
from .query_row_r3_diagnosis import (
    DEFAULT_R3_DATASETS,
    DEFAULT_R3_SEEDS,
    DEFAULT_R3_UPDATES,
    run_query_row_r3_diagnosis,
)
from .query_row_supervised_synthetic import (
    QueryRowSupervisedSyntheticEpisode,
    make_query_row_supervised_synthetic_episode,
    supervised_synthetic_episode_loss,
)
from .query_row_supervised_synthetic_v2 import (
    CONTEXT_ANCHORS,
    GENERATOR_ID,
    NOISE_LEVELS,
    PREDICTOR_REGIMES,
    QueryRowSupervisedSyntheticV2Episode,
    WORLD_FAMILIES,
    WIDTH_BUCKETS,
    build_query_row_supervised_synthetic_v2_plan,
    make_query_row_supervised_synthetic_v2_episode,
    substitute_query_truth,
    supervised_synthetic_v2_episode_loss,
    validate_query_row_supervised_synthetic_v2,
)
from .query_row_r5_pretraining import (
    R5_LEARNING_RATES,
    R5_RUNG_SPECS,
    run_query_row_r5_bounded_pretraining,
)
from .query_row_synthetic_fit import (
    QueryRowSyntheticEpisode,
    QueryRowSyntheticFitResult,
    QueryRowSyntheticMultiWorldFitResult,
    make_query_row_synthetic_episode,
    run_query_row_fixed_world_fit,
    run_query_row_multi_world_fit,
)
from .synthetic_fit import (
    SyntheticFitResult,
    SyntheticWorldBatch,
    make_linear_world_batch,
    run_synthetic_fit,
)

__all__ = [
    "SyntheticFitResult",
    "SyntheticWorldBatch",
    "make_linear_world_batch",
    "run_synthetic_fit",
    "QueryRowSyntheticEpisode",
    "QueryRowSyntheticFitResult",
    "QueryRowSyntheticMultiWorldFitResult",
    "make_query_row_synthetic_episode",
    "run_query_row_fixed_world_fit",
    "run_query_row_multi_world_fit",
    "QueryRowRealBenchmarkResult",
    "QueryRowRealDatasetResult",
    "run_query_row_real_scratch_benchmark",
    "numeric_raw_prediction",
    "numeric_raw_prediction_from_public",
    "query_row_real_regression_loss",
    "task_scale_to_raw",
    "DEFAULT_R3_DATASETS",
    "DEFAULT_R3_SEEDS",
    "DEFAULT_R3_UPDATES",
    "run_query_row_r3_diagnosis",
    "QueryRowFrozenICLRecord",
    "QueryRowFrozenICLResult",
    "run_query_row_frozen_icl",
    "LINEAR_REGRESSION_BASELINE_ID",
    "LINEAR_REGRESSION_BASELINE_SPEC",
    "QueryRowLinearICLContextSummary",
    "QueryRowLinearICLRecord",
    "QueryRowLinearICLThresholdResult",
    "run_query_row_linear_icl_threshold",
    "QueryRowSupervisedSyntheticEpisode",
    "make_query_row_supervised_synthetic_episode",
    "supervised_synthetic_episode_loss",
    "CONTEXT_ANCHORS",
    "GENERATOR_ID",
    "NOISE_LEVELS",
    "PREDICTOR_REGIMES",
    "QueryRowSupervisedSyntheticV2Episode",
    "WORLD_FAMILIES",
    "WIDTH_BUCKETS",
    "build_query_row_supervised_synthetic_v2_plan",
    "make_query_row_supervised_synthetic_v2_episode",
    "substitute_query_truth",
    "supervised_synthetic_v2_episode_loss",
    "validate_query_row_supervised_synthetic_v2",
    "R5_LEARNING_RATES",
    "R5_RUNG_SPECS",
    "run_query_row_r5_bounded_pretraining",
    "QueryRowFinetuneLiftRecord",
    "QueryRowFinetuneLiftResult",
    "run_query_row_finetune_lift",
    "QueryRowPretrainingResult",
    "load_query_row_pretrain_checkpoint",
    "run_query_row_synthetic_pretraining",
    "save_query_row_pretrain_checkpoint",
    "train_query_row_synthetic_pretraining_model",
    "CLASSICAL_ICL_BASELINE_IDS",
    "CLASSICAL_ICL_CONFIG",
    "QueryRowClassicalICLRecord",
    "QueryRowClassicalICLResult",
    "run_query_row_classical_icl_benchmark",
]
