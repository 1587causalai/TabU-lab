"""Public evaluator surface for dense reference models."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .artifacts import UnissuedArtifactSet, write_unissued_evaluation_artifacts
from .evaluator import Evaluator, evaluate, evaluate_model
from .foundry import (
    BaselineAdapter,
    ComparisonReport,
    DatasetUnavailableError,
    EvalResult,
    EvalSuiteSpec,
    ModelAdapter,
    ScenarioSpec,
    compare_results,
    dry_run_suite,
    load_suite,
    run_evaluation,
    validate_suite,
)

if TYPE_CHECKING:
    from .fit_artifacts import FitAttemptArtifacts

_FIT_ARTIFACT_EXPORTS = frozenset(
    {
        "FitAttemptArtifacts",
        "assert_public_artifact_tree_safe",
        "assert_public_payload_safe",
        "capture_environment",
        "verify_fit_attempt_artifacts",
        "write_fit_attempt_artifacts",
    }
)
_FORMAL_RECEIPT_EXPORTS = frozenset(
    {
        "FormalEvaluationReceiptError",
        "issue_formal_evaluation_receipt",
    }
)


def __getattr__(name: str) -> Any:
    """Load fit-attempt helpers lazily to keep package imports acyclic."""

    if name in _FIT_ARTIFACT_EXPORTS:
        module = "fit_artifacts"
    elif name in _FORMAL_RECEIPT_EXPORTS:
        module = "formal_receipt"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_FIT_ARTIFACT_EXPORTS, *_FORMAL_RECEIPT_EXPORTS})

__all__ = [
    "BaselineAdapter",
    "ComparisonReport",
    "DatasetUnavailableError",
    "EvalResult",
    "EvalSuiteSpec",
    "Evaluator",
    "FitAttemptArtifacts",
    "FormalEvaluationReceiptError",
    "ModelAdapter",
    "ScenarioSpec",
    "UnissuedArtifactSet",
    "assert_public_artifact_tree_safe",
    "assert_public_payload_safe",
    "capture_environment",
    "compare_results",
    "dry_run_suite",
    "evaluate",
    "evaluate_model",
    "issue_formal_evaluation_receipt",
    "load_suite",
    "run_evaluation",
    "validate_suite",
    "verify_fit_attempt_artifacts",
    "write_fit_attempt_artifacts",
    "write_unissued_evaluation_artifacts",
]
