"""Small, reproducible evaluation runners."""

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
from .query_row_real_benchmark import (
    QueryRowRealBenchmarkResult,
    QueryRowRealDatasetResult,
    run_query_row_real_scratch_benchmark,
)
from .query_row_supervised_synthetic import (
    QueryRowSupervisedSyntheticEpisode,
    make_query_row_supervised_synthetic_episode,
    supervised_synthetic_episode_loss,
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
    "QueryRowFrozenICLRecord",
    "QueryRowFrozenICLResult",
    "run_query_row_frozen_icl",
    "QueryRowSupervisedSyntheticEpisode",
    "make_query_row_supervised_synthetic_episode",
    "supervised_synthetic_episode_loss",
    "QueryRowFinetuneLiftRecord",
    "QueryRowFinetuneLiftResult",
    "run_query_row_finetune_lift",
]
