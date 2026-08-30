"""Model Verification & Evaluation (MVE) public API."""

# Import built-in probes for their explicit registry side effects.
from . import probes as _probes  # noqa: F401
from .composition import describe_model
from .contracts import (
    AssessmentOutcome,
    EvidenceLevel,
    ModelCompositionDescriptor,
    VerificationAxis,
    VerificationCheck,
    VerificationCheckResult,
    VerificationResult,
    VerificationSuite,
)
from .registry import (
    VerificationRegistryError,
    get_check,
    list_checks,
    register_check,
    run_check,
)
from .runner import (
    VerificationRunnerError,
    list_suites,
    load_suite,
    read_result,
    run_suite,
    validate_suites,
    write_result,
)

__all__ = [
    "AssessmentOutcome",
    "EvidenceLevel",
    "ModelCompositionDescriptor",
    "VerificationAxis",
    "VerificationCheck",
    "VerificationCheckResult",
    "VerificationRegistryError",
    "VerificationResult",
    "VerificationRunnerError",
    "VerificationSuite",
    "describe_model",
    "get_check",
    "list_checks",
    "list_suites",
    "load_suite",
    "read_result",
    "register_check",
    "run_check",
    "run_suite",
    "validate_suites",
    "write_result",
]
