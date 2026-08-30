"""Multi-episode S1 execution over immutable synthetic fit corpora.

This module deliberately owns only the S1 orchestration layer.  The existing
``runner`` module remains the F0 implementation and supplies the shared model,
trainer, evidence, and receipt primitives.  S1 differs in four material ways:

* the next train episode is selected from ``trainer.step`` so exact resume also
  restores the corpus cursor;
* every unique train episode participates in initial/final evaluation and the
  gate uses target-weighted typed-family metrics;
* validation and test episodes are forward-evaluated as diagnostics only; and
* every realized recipe hash is passed explicitly into the immutable RunBundle.

TruthSidecar values remain host-side throughout.  Only ``episode.evidence`` is
ever passed to a model.
"""

from __future__ import annotations

import math
import os
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from tabu_lab.contracts import (
    FeatureKind,
    OriginState,
    PredictionBundle,
    canonical_hash,
    origin_mask,
    to_canonical_data,
)
from tabu_lab.evaluation import Evaluator, write_fit_attempt_artifacts
from tabu_lab.evaluation.fit_artifacts import assert_public_payload_safe
from tabu_lab.evidence import EnvironmentDisclosure, ReceiptStatus, RunIdentity
from tabu_lab.evidence.formal_authorization import (
    FormalAuthorizationContext,
    FormalAuthorizationError,
    VerifiedFormalAuthorization,
    verify_formal_authorization,
)
from tabu_lab.evidence.source_identity import SourceIdentity
from tabu_lab.observers import get_observer
from tabu_lab.training import Objective, Trainer

from .contracts import (
    FitEvaluationBundle,
    FitEvidenceMode,
    FitExperimentSpec,
    FitFamilyMetrics,
    FitMetricKind,
    FitStage,
    FitTargetFamily,
    derive_attempt_id,
)
from .corpus import CompiledFitEpisode, FitEpisodeCorpus
from .corpus_manifest import build_corpus_compiler_binding_manifest
from .feasibility import FeasibilityReportStatus, FitFeasibilityReport, assess_nw_targets

# Importing these implementation helpers is intentional: runner.py lazily
# imports this module only after it has parsed an S1 spec, avoiding a cycle.
from .runner import (
    ExperimentRunResult,
    FitExperimentError,
    SeedRunResult,
    _assert_formal_output_root_safe,
    _authorization_safe_command,
    _build_model,
    _device,
    _exception_boundary,
    _forward_in_eval,
    _gate_reasons,
    _identity,
    _mechanism_gradient_probe,
    _NonfiniteFitError,
    _parameter_delta_norm,
    _parameter_snapshot,
    _resolve_formal_authorization,
    _runtime_failure_artifacts,
    _seed_verdict,
    _selected_categorical_nll,
    _trainer,
    _tabubase_identity_metadata,
    _training_and_execution_configs,
    _write_experiment_aggregate,
    load_fit_experiment,
    source_tree_manifest,
    trivial_baseline,
)
from .s1_registry import build_registered_s1_corpus, get_s1_registration


@dataclass(frozen=True, slots=True)
class CheckpointCorpusRoundtrip:
    """Exact checkpoint/readout/cursor replay result for one S1 corpus."""

    reloaded: bool
    continuation_verified: bool
    saved_step: int
    restored_step: int
    continuation_step: int
    continuation_loss: float
    continuation_state_hash: str
    next_recipe_hash: str
    prediction_hashes: tuple[str, ...]
    continuation_prediction_hashes: tuple[str, ...]


class S1StopReason(StrEnum):
    """Typed terminal reason for the bounded S1 update loop."""

    MAX_UPDATES = "max_updates"
    WALL_CLOCK_BUDGET = "wall_clock_budget"


@dataclass(frozen=True, slots=True)
class _FailureFixtureAdapter:
    """Narrow adapter for the shared immutable runtime-failure writer."""

    dataset: Any
    recipe: Any


def _all_episodes(corpus: FitEpisodeCorpus) -> tuple[CompiledFitEpisode, ...]:
    return corpus.train_episodes + corpus.validation_episodes + corpus.test_episodes


def _all_recipe_hashes(corpus: FitEpisodeCorpus) -> tuple[str, ...]:
    return tuple(episode.recipe_hash for episode in _all_episodes(corpus))


def _flatten_train_targets(corpus: FitEpisodeCorpus) -> tuple[Any, ...]:
    return tuple(
        target for episode in corpus.train_episodes for target in episode.feasibility_targets
    )


def validate_s1_corpus_binding(
    spec: FitExperimentSpec,
    corpus: FitEpisodeCorpus,
) -> None:
    """Bind one loaded preregistration to its registered deterministic corpus."""

    if spec.stage is not FitStage.S1:
        raise FitExperimentError("S1 corpus binding requires an S1 experiment")
    try:
        registration = get_s1_registration(spec.experiment_id)
    except KeyError as exc:
        raise FitExperimentError(str(exc)) from exc
    if registration.contract_id != spec.contract_id:
        raise FitExperimentError("S1 registry contract does not match preregistration")
    if spec.dataset.source_uri != registration.source_uri:
        raise FitExperimentError("S1 dataset source URI is not the registered generator")
    if spec.dataset.source_sha256 != registration.source_hash:
        raise FitExperimentError("S1 generator source hash does not match this build")
    if spec.dataset.adapter.adapter_id != registration.adapter_id:
        raise FitExperimentError("S1 dataset adapter id does not match the registry")
    if spec.dataset.adapter.adapter_version != registration.adapter_version:
        raise FitExperimentError("S1 dataset adapter version does not match the registry")
    if (
        spec.dataset.dataset_id != corpus.dataset.dataset_id
        or spec.dataset.dataset_hash != corpus.dataset.dataset_hash
    ):
        raise FitExperimentError("S1 preregistration does not bind the generated dataset")
    if spec.split != corpus.typed_split:
        raise FitExperimentError("S1 preregistration does not bind the typed split")
    if spec.episode_schedule != corpus.schedule:
        raise FitExperimentError("S1 preregistration does not bind the episode schedule")
    if corpus.schedule_realization.schedule_hash != spec.episode_schedule.content_hash:
        raise FitExperimentError("S1 corpus realization does not bind the schedule hash")
    if corpus.schedule_realization.typed_split_hash != spec.split.content_hash:
        raise FitExperimentError("S1 corpus realization does not bind the typed split hash")
    if corpus.schedule_realization.fit_value_mask_hash != corpus.fit_value_mask_hash:
        raise FitExperimentError("S1 corpus realization does not bind fit-only statistics")
    if set(spec.target_families) != set(corpus.schedule.target_families):
        raise FitExperimentError("S1 corpus target families do not match preregistration")
    recipe_hashes = _all_recipe_hashes(corpus)
    if len(recipe_hashes) != spec.episode_schedule.episode_count:
        raise FitExperimentError("S1 realized episode count does not match preregistration")
    if len(recipe_hashes) != len(set(recipe_hashes)):
        raise FitExperimentError("S1 realized recipe hashes must be globally unique")

    options = corpus.builder_options
    if spec.contract_id in {"tabul", "tabufl"}:
        if tuple(options.get("label_columns", ())) != spec.semantic.label_columns:
            raise FitExperimentError("S1 label columns do not match corpus builder options")
        plan = spec.semantic.label_address_plan
        if plan is None or options.get("label_address_plan") != plan.value:
            raise FitExperimentError("S1 label address plan does not match corpus")
    elif spec.contract_id == "tabu.cell.base":
        if options.get("profile") != spec.semantic.profile_id:
            raise FitExperimentError("S1 Base profile does not match corpus builder options")
        if spec.semantic.profile_id == "supervised.label_broadcast.v1" and tuple(
            options.get("label_columns", (6,))
        ) != spec.semantic.label_columns:
            raise FitExperimentError("S1 Base response column does not match corpus")
    elif spec.contract_id == "tabu4graph":
        if options.get("target_feature") != spec.semantic.target_feature:
            raise FitExperimentError("S1 graph target feature does not match corpus")
        plan = spec.semantic.graph_unit_receiver_plan
        if plan is None or options.get("unit_receiver_plan") != plan.value:
            raise FitExperimentError("S1 graph Unit receiver plan does not match corpus")
    elif spec.contract_id == "tabu4rec":
        if options.get("response_family") != spec.semantic.response_family:
            raise FitExperimentError("S1 Rec response family does not match corpus")
        plan = spec.semantic.recommendation_address_plan
        if plan is None or options.get("recommendation_address_plan") != plan.value:
            raise FitExperimentError("S1 Rec address plan does not match corpus")
        if options.get("rec_axis_summary_dim") != spec.semantic.rec_axis_summary_dim:
            raise FitExperimentError("S1 Rec axis summary dimension does not match corpus")
        if not math.isclose(
            float(options.get("rec_matched_residual_scale", -1.0)),
            float(spec.semantic.rec_matched_residual_scale or -1.0),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise FitExperimentError("S1 Rec residual scale does not match corpus")


def build_s1_corpus(spec: FitExperimentSpec) -> FitEpisodeCorpus:
    """Resolve and validate the closed S1 registry entry named by ``spec``."""

    try:
        corpus = build_registered_s1_corpus(spec.experiment_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise FitExperimentError(f"cannot build registered S1 corpus: {exc}") from exc
    validate_s1_corpus_binding(spec, corpus)
    return corpus


def s1_compiler_binding_manifest(
    spec: FitExperimentSpec,
    corpus: FitEpisodeCorpus,
) -> dict[str, Any]:
    """Emit the complete truth-opaque v2 compiler/corpus hash preimage."""

    validate_s1_corpus_binding(spec, corpus)
    return build_corpus_compiler_binding_manifest(
        corpus,
        contract_id=spec.contract_id,
    )


def assess_s1_feasibility(
    spec: FitExperimentSpec,
    corpus: FitEpisodeCorpus,
) -> tuple[tuple[Any, ...], FitFeasibilityReport]:
    """Flatten every unique train target into one pre-training oracle report."""

    targets = _flatten_train_targets(corpus)
    report = assess_nw_targets(targets, report_id=f"{spec.experiment_id}-train")
    return targets, report


def _forward_episodes(
    model: torch.nn.Module,
    episodes: Sequence[CompiledFitEpisode],
    *,
    device: torch.device,
) -> tuple[PredictionBundle, ...]:
    return tuple(_forward_in_eval(model, episode.evidence, device=device) for episode in episodes)


def _target_weighted_objective(
    predictions: Sequence[PredictionBundle],
    episodes: Sequence[CompiledFitEpisode],
    *,
    device: torch.device,
) -> float:
    if len(predictions) != len(episodes) or not predictions:
        raise FitExperimentError("S1 objective requires aligned non-empty episodes")
    total = 0.0
    weight = 0
    objective = Objective()
    for prediction, episode in zip(predictions, episodes, strict=True):
        target_count = episode.truth.target_count
        value = float(objective(prediction, episode.truth.to(device)).total.detach().cpu())
        if not math.isfinite(value):
            raise _NonfiniteFitError("S1 corpus objective is NaN or Inf")
        total += value * target_count
        weight += target_count
    if weight <= 0:
        raise FitExperimentError("S1 train corpus contains zero targets")
    return total / weight


def _host_target_ledgers(
    episode: CompiledFitEpisode,
) -> dict[str, torch.Tensor]:
    """Resolve the four typed target ledgers without consulting model output.

    The compiler-owned schema determines numeric/categorical membership and the
    compiler-owned origin state determines completion/label membership.  The
    named corpus ledger must independently agree with that partition.  This is
    particularly important for TabUFL, where an aggregate target mask cannot
    reveal an F/L lane swap.
    """

    truth_target = episode.truth.target_mask.detach().cpu().to(torch.bool)
    evidence = episode.evidence
    if len(evidence.feature_specs) != truth_target.shape[1]:
        raise FitExperimentError("S1 host feature schema does not match target width")
    numeric_features = torch.tensor(
        tuple(spec.kind is FeatureKind.NUMERIC for spec in evidence.feature_specs),
        dtype=torch.bool,
    ).view(1, -1)
    numeric = truth_target & numeric_features
    categorical = truth_target & ~numeric_features
    origins = evidence.origin_states.detach().cpu()
    completion = truth_target & origin_mask(origins, OriginState.ARTIFICIAL_MASK)
    label = truth_target & origin_mask(origins, OriginState.QUERY)
    if bool((completion & label).any()) or not torch.equal(completion | label, truth_target):
        raise FitExperimentError(
            "S1 host family ledger must partition targets into artificial-mask F or query L"
        )

    declared_completion = torch.zeros_like(truth_target)
    declared_label = torch.zeros_like(truth_target)
    for name, raw_mask in episode.target_family_masks.items():
        mask = raw_mask.detach().cpu().to(torch.bool)
        if tuple(mask.shape) != tuple(truth_target.shape):
            raise FitExperimentError("S1 named target-family ledger has the wrong shape")
        selects_completion = bool((mask & completion).any())
        selects_label = bool((mask & label).any())
        if selects_completion == selects_label:
            raise FitExperimentError(
                f"S1 named target-family ledger {name!r} does not select exactly one F/L lane"
            )
        if selects_completion:
            declared_completion |= mask
        else:
            declared_label |= mask
    if not torch.equal(declared_completion, completion) or not torch.equal(
        declared_label, label
    ):
        raise FitExperimentError("S1 named target-family ledger disagrees with compiler origins")
    if "F" in episode.target_family_masks and not torch.equal(
        episode.target_family_masks["F"].detach().cpu().to(torch.bool), completion
    ):
        raise FitExperimentError("S1 TabUFL F ledger disagrees with completion targets")
    if "L" in episode.target_family_masks and not torch.equal(
        episode.target_family_masks["L"].detach().cpu().to(torch.bool), label
    ):
        raise FitExperimentError("S1 supervised L ledger disagrees with label targets")
    return {
        "target_mask": truth_target,
        "numeric_target_mask": numeric,
        "categorical_target_mask": categorical,
        "completion_target_mask": completion,
        "label_target_mask": label,
    }


def _validated_prediction_ledgers(
    prediction: PredictionBundle,
    episode: CompiledFitEpisode,
    *,
    expected: Mapping[str, torch.Tensor],
    phase: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Require one model forward to reproduce every host-side target ledger."""

    if prediction.episode_id != episode.truth.episode_id:
        raise FitExperimentError(f"S1 {phase} prediction episode id does not match truth sidecar")
    observed: dict[str, torch.Tensor] = {}
    for name, host_mask in expected.items():
        model_mask = prediction.auxiliaries.get(name)
        if model_mask is None:
            raise FitExperimentError(f"S1 {phase} prediction is missing {name}")
        if model_mask.dtype is not torch.bool:
            raise FitExperimentError(f"S1 {phase} prediction {name} must have bool dtype")
        resolved = model_mask.detach().to(device="cpu")
        if tuple(resolved.shape) != tuple(host_mask.shape) or not torch.equal(
            resolved, host_mask
        ):
            raise FitExperimentError(
                f"S1 {phase} prediction {name} disagrees with the host-side ledger"
            )
        observed[name] = model_mask.to(torch.bool)
    support = prediction.auxiliaries.get("support_available")
    if support is None or support.dtype is not torch.bool:
        raise FitExperimentError(f"S1 {phase} support_available must be a bool tensor")
    if tuple(support.shape) != tuple(expected["target_mask"].shape):
        raise FitExperimentError(f"S1 {phase} support_available has the wrong shape")
    return observed, support.to(torch.bool)


def _aggregate_family_metrics(
    *,
    initial: Sequence[PredictionBundle],
    final: Sequence[PredictionBundle],
    episodes: Sequence[CompiledFitEpisode],
    baseline: Mapping[str, Any],
) -> tuple[FitFamilyMetrics, ...]:
    """Compute exact train-corpus metrics directly over all typed targets.

    In particular, numeric NRMSE uses one truth standard deviation for the
    complete ``(family, numeric)`` ledger.  It is not an average of per-episode
    NRMSE values.  Categorical NLL and accuracy likewise aggregate target-level
    values before division.
    """

    if len(initial) != len(final) or len(final) != len(episodes) or not episodes:
        raise FitExperimentError("S1 family metrics require aligned non-empty episodes")
    results: list[FitFamilyMetrics] = []
    for family, family_key in (
        (FitTargetFamily.COMPLETION, "completion_target_mask"),
        (FitTargetFamily.LABEL, "label_target_mask"),
    ):
        numeric_targets = 0
        numeric_scored = 0
        numeric_truth: list[torch.Tensor] = []
        numeric_initial_errors: list[torch.Tensor] = []
        numeric_final_errors: list[torch.Tensor] = []
        categorical_targets = 0
        categorical_scored = 0
        categorical_initial_nll: list[torch.Tensor] = []
        categorical_final_nll: list[torch.Tensor] = []
        categorical_correct = 0

        for initial_prediction, final_prediction, episode in zip(
            initial, final, episodes, strict=True
        ):
            expected = _host_target_ledgers(episode)
            initial_ledgers, initial_support = _validated_prediction_ledgers(
                initial_prediction,
                episode,
                expected=expected,
                phase="initial",
            )
            final_ledgers, final_support = _validated_prediction_ledgers(
                final_prediction,
                episode,
                expected=expected,
                phase="final",
            )
            device = final_support.device
            truth_targets = expected["target_mask"].to(device=device)
            truth_values = episode.truth.target_values.to(device=device)
            final_family = final_ledgers[family_key].to(device=device)
            initial_family = initial_ledgers[family_key].to(device=device)
            final_numeric = final_ledgers["numeric_target_mask"].to(device=device)
            initial_numeric = initial_ledgers["numeric_target_mask"].to(device=device)
            final_categorical = final_ledgers["categorical_target_mask"].to(device=device)
            initial_categorical = initial_ledgers["categorical_target_mask"].to(device=device)
            initial_support = initial_support.to(device=device)
            if not (
                torch.equal(final_family, initial_family)
                and torch.equal(final_numeric, initial_numeric)
                and torch.equal(final_categorical, initial_categorical)
                and torch.equal(final_support, initial_support)
            ):
                raise FitExperimentError(
                    "S1 typed target/support geometry changed between initial and final forward"
                )

            numeric_mask = truth_targets & final_family & final_numeric
            numeric_support = numeric_mask & final_support
            numeric_targets += int(numeric_mask.sum().item())
            numeric_scored += int(numeric_support.sum().item())
            if bool(numeric_support.any()):
                initial_values = initial_prediction.entries["numeric"].values
                final_values = final_prediction.entries["numeric"].values
                if initial_values is None or final_values is None:
                    raise FitExperimentError("supported numeric targets require values")
                selected_truth = truth_values[numeric_support].float()
                numeric_truth.append(selected_truth.detach().cpu())
                numeric_initial_errors.append(
                    (initial_values[numeric_support].float() - selected_truth).detach().cpu()
                )
                numeric_final_errors.append(
                    (final_values[numeric_support].float() - selected_truth).detach().cpu()
                )

            categorical_mask = truth_targets & final_family & final_categorical
            categorical_support = categorical_mask & final_support
            categorical_targets += int(categorical_mask.sum().item())
            categorical_scored += int(categorical_support.sum().item())
            if bool(categorical_support.any()):
                raw_codes = truth_values[categorical_support]
                codes = raw_codes.round().long()
                if not bool(torch.isclose(raw_codes, codes.to(raw_codes.dtype)).all()):
                    raise FitExperimentError(
                        "categorical train truth must contain integer domain codes"
                    )
                initial_probabilities = initial_prediction.entries["distribution"].values
                final_probabilities = final_prediction.entries["distribution"].values
                if initial_probabilities is None or final_probabilities is None:
                    raise FitExperimentError("supported categorical targets require distributions")
                categorical_initial_nll.append(
                    _selected_categorical_nll(
                        initial_prediction,
                        initial_probabilities,
                        categorical_support,
                        codes,
                    )
                    .detach()
                    .float()
                    .cpu()
                )
                categorical_final_nll.append(
                    _selected_categorical_nll(
                        final_prediction,
                        final_probabilities,
                        categorical_support,
                        codes,
                    )
                    .detach()
                    .float()
                    .cpu()
                )
                categorical_correct += int(
                    (final_probabilities[categorical_support].argmax(-1) == codes).sum().item()
                )

        family_baseline = baseline.get("families", {}).get(family.value, {})
        if numeric_targets:
            if numeric_scored:
                initial_errors = torch.cat(numeric_initial_errors)
                final_errors = torch.cat(numeric_final_errors)
                all_truth = torch.cat(numeric_truth)
                initial_mse = float(initial_errors.square().mean().item())
                final_mse = float(final_errors.square().mean().item())
                truth_scale = float(all_truth.std(unbiased=False).item())
                nrmse = final_mse**0.5 / max(truth_scale, 1.0e-8)
            else:
                initial_mse = final_mse = nrmse = 0.0
            results.append(
                FitFamilyMetrics(
                    family=family,
                    kind=FitMetricKind.NUMERIC,
                    targets=numeric_targets,
                    scored_targets=numeric_scored,
                    initial_loss=initial_mse,
                    final_loss=final_mse,
                    trivial_baseline_loss=family_baseline.get("numeric_mse"),
                    mse=final_mse,
                    nrmse=nrmse,
                )
            )
        if categorical_targets:
            if categorical_scored:
                initial_nll = float(torch.cat(categorical_initial_nll).mean().item())
                final_nll = float(torch.cat(categorical_final_nll).mean().item())
                accuracy = categorical_correct / categorical_scored
            else:
                initial_nll = final_nll = accuracy = 0.0
            results.append(
                FitFamilyMetrics(
                    family=family,
                    kind=FitMetricKind.CATEGORICAL,
                    targets=categorical_targets,
                    scored_targets=categorical_scored,
                    initial_loss=initial_nll,
                    final_loss=final_nll,
                    trivial_baseline_loss=family_baseline.get("categorical_nll"),
                    accuracy=accuracy,
                    nll=final_nll,
                )
            )
    if not results:
        raise FitExperimentError("S1 train corpus produced no typed family metrics")
    return tuple(results)


def _aggregate_mechanism_probe(
    model: torch.nn.Module,
    episodes: Sequence[CompiledFitEpisode],
    *,
    contract_id: str,
    device: torch.device,
) -> tuple[dict[str, int], dict[str, int], dict[str, float], int]:
    source_counts: dict[str, int] = defaultdict(int)
    active_counts: dict[str, int] = defaultdict(int)
    gradient_squares: dict[str, float] = defaultdict(float)
    scored = 0
    for episode in episodes:
        if contract_id == "tabu4graph":
            sources, active, gradients, episode_scored = _graph_mechanism_gradient_probe(
                model,
                episode,
                device=device,
            )
        else:
            if contract_id == "tabu4rec":
                prediction = _forward_in_eval(model, episode.evidence, device=device)
                if not (
                    prediction.trace is not None
                    and prediction.trace.metadata.get("recommendation_address_plan")
                    == "matched_uf"
                    and prediction.trace.metadata.get("numeric_terminal")
                    == "parameterized_matching"
                ):
                    _validate_rec_dual_arm_trace(prediction)
            sources, active, gradients, episode_scored = _mechanism_gradient_probe(
                model,
                episode.evidence,
                episode.truth,
                contract_id=contract_id,
                device=device,
            )
        for name, count in sources.items():
            source_counts[name] += count
        for name, count in active.items():
            active_counts[name] += count
        for name, norm in gradients.items():
            gradient_squares[name] += float(norm) ** 2
        scored += episode_scored
    return (
        dict(sorted(source_counts.items())),
        dict(sorted(active_counts.items())),
        {name: value**0.5 for name, value in sorted(gradient_squares.items())},
        scored,
    )


def _validate_rec_dual_arm_trace(prediction: PredictionBundle) -> None:
    """Require the S1 Rec forward trace to declare both frozen support arms."""

    if prediction.trace is None:
        raise FitExperimentError("S1 TabU4Rec mechanism probe requires a ForwardTrace")
    events = {event.name: event for event in prediction.trace.events}
    ledger = events.get("recommendation_support_ledger")
    if ledger is None:
        raise FitExperimentError("S1 TabU4Rec trace omits recommendation_support_ledger")
    operations = tuple(ledger.metadata.get("operation_trace", ()))
    required = {
        "same_item_other_users",
        "same_user_other_response_columns",
        "equal_active_arm_mix",
        "single_active_arm_renormalizes_to_one",
    }
    if not required.issubset(operations):
        missing = ", ".join(sorted(required.difference(operations)))
        raise FitExperimentError(f"S1 TabU4Rec trace omits dual-arm operations: {missing}")
    if tuple(ledger.metadata.get("arm_order", ())) != ("user", "item"):
        raise FitExperimentError("S1 TabU4Rec trace does not declare user/item arm order")
    source_count = ledger.metadata.get("source_count")
    if type(source_count) is not int or source_count <= 0:
        raise FitExperimentError("S1 TabU4Rec trace requires nonzero response sources")


def _graph_mechanism_gradient_probe(
    model: torch.nn.Module,
    episode: CompiledFitEpisode,
    *,
    device: torch.device,
) -> tuple[dict[str, int], dict[str, int], dict[str, float], int]:
    """Audit graph-local computation and the parameter-free global readout.

    ``MatchedUFReadout`` has no learned parameter of its own, so the readout
    path is probed through the gradient of the objective with respect to its
    live coordinate tensor.  The graph-local path is independently probed on
    the parameters of every ``graph_mab`` block.  This is more specific than
    the coarse trainer ``dynamics``/``readout`` gradient groups.
    """

    readout = getattr(model, "readout", None)
    if not isinstance(readout, torch.nn.Module):
        raise FitExperimentError("TabU4Graph mechanism probe requires a readout module")
    graph_parameters = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if ".graph_mab." in name and parameter.requires_grad
    )
    if not graph_parameters:
        raise FitExperimentError("TabU4Graph has no auditable graph_mab parameters")

    captured: dict[str, torch.Tensor] = {}

    def capture_readout(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
            raise FitExperimentError("TabU4Graph readout hook did not observe coordinates")
        captured["coordinates"] = output[0]

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    handle = readout.register_forward_hook(capture_readout)
    try:
        prediction = model(episode.evidence.to(device))
    finally:
        handle.remove()
    try:
        if not isinstance(prediction, PredictionBundle):
            raise FitExperimentError("TabU4Graph mechanism probe requires PredictionBundle")
        if prediction.trace.metadata.get("dynamics_plan") != "graph_four_stage":
            raise FitExperimentError("TabU4Graph trace does not declare graph_four_stage")
        if prediction.trace.metadata.get("readout_path") != "global_same_column_visible_support":
            raise FitExperimentError("TabU4Graph trace does not declare global readout path")
        events = {event.name: event for event in prediction.trace.events}
        dynamics_event = events.get("dynamics_plan")
        readout_event = events.get("readout")
        if dynamics_event is None or readout_event is None:
            raise FitExperimentError("TabU4Graph trace is missing dynamics/readout events")
        dynamics_operations = tuple(dynamics_event.metadata.get("operation_trace", ()))
        readout_operations = tuple(readout_event.metadata.get("operation_trace", ()))
        if "graph_local_unit_feature_evidence" not in dynamics_operations:
            raise FitExperimentError("TabU4Graph trace omits graph-local operation")
        if "global_feature_prototype_for_readout" not in readout_operations:
            raise FitExperimentError("TabU4Graph trace omits global readout operation")
        graph_source_count = dynamics_event.metadata.get("source_count")
        readout_source_count = readout_event.metadata.get("source_count")
        if (
            type(graph_source_count) is not int
            or graph_source_count <= 0
            or type(readout_source_count) is not int
            or readout_source_count <= 0
        ):
            raise FitExperimentError("TabU4Graph trace requires nonzero path sources")

        coordinates = captured.get("coordinates")
        if coordinates is None or not coordinates.requires_grad:
            raise FitExperimentError("TabU4Graph global readout coordinates are not differentiable")
        loss = Objective()(prediction, episode.truth.to(device)).total
        gradients = torch.autograd.grad(
            loss,
            (*graph_parameters, coordinates),
            allow_unused=True,
        )
        local_squared = sum(
            float(gradient.detach().float().square().sum().cpu())
            for gradient in gradients[:-1]
            if gradient is not None
        )
        coordinate_gradient = gradients[-1]
        readout_squared = (
            0.0
            if coordinate_gradient is None
            else float(coordinate_gradient.detach().float().square().sum().cpu())
        )
        target_mask = episode.truth.target_mask.detach().cpu().to(torch.bool)
        support = prediction.auxiliaries["support_available"].detach().cpu().to(torch.bool)
        if tuple(support.shape) != tuple(target_mask.shape):
            raise FitExperimentError("TabU4Graph support mask does not match host targets")
        scored_mask = target_mask & support
        scored = int(scored_mask.sum().item())

        routing_source_count = prediction.auxiliaries.get("routing_source_count")
        if routing_source_count is None:
            raise FitExperimentError("TabU4Graph probe requires per-target routing source counts")
        routing_source_count = routing_source_count.detach().cpu()
        if tuple(routing_source_count.shape) != tuple(target_mask.shape) or bool(
            (routing_source_count < 0).any()
        ):
            raise FitExperimentError("TabU4Graph routing source ledger is invalid")
        global_active_mask = scored_mask & (routing_source_count > 0)

        topology = episode.evidence.graph_topology
        if topology is None:
            raise FitExperimentError("TabU4Graph probe requires typed graph topology")
        adjacency = topology.adjacency.to(torch.bool)
        closed = adjacency | adjacency.transpose(0, 1) | torch.eye(
            adjacency.shape[0], dtype=torch.bool
        )
        evidence_sources = episode.evidence.source_mask.detach().cpu().to(torch.bool)
        local_source_count = torch.zeros_like(target_mask, dtype=torch.int64)
        for row, feature in torch.nonzero(scored_mask, as_tuple=False).tolist():
            local_source_count[row, feature] = int(
                (closed[row] & evidence_sources[:, feature]).sum().item()
            )
        local_active_mask = scored_mask & (local_source_count > 0)
        return (
            {
                "graph_global_readout": int(routing_source_count[scored_mask].sum().item()),
                "graph_local_path": int(local_source_count[scored_mask].sum().item()),
            },
            {
                "graph_global_readout": int(global_active_mask.sum().item()),
                "graph_local_path": int(local_active_mask.sum().item()),
            },
            {
                "graph_global_readout": readout_squared**0.5,
                "graph_local_path": local_squared**0.5,
            },
            scored,
        )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def _s1_mechanism_gate_reasons(
    spec: FitExperimentSpec,
    evaluation: FitEvaluationBundle,
) -> tuple[str, ...]:
    if spec.contract_id != "tabu4graph":
        return ()
    reasons: list[str] = []
    if evaluation.mechanism_scored_target_count != evaluation.scored_targets:
        reasons.append("graph_mechanism_scored_target_count_mismatch")
    for mechanism in ("graph_local_path", "graph_global_readout"):
        if evaluation.mechanism_source_counts.get(mechanism, 0) <= 0:
            reasons.append(f"{mechanism}_has_no_source")
        if evaluation.mechanism_active_target_counts.get(mechanism, 0) != evaluation.scored_targets:
            reasons.append(f"{mechanism}_not_active_for_every_scored_target")
        if evaluation.mechanism_gradient_norms.get(mechanism, 0.0) <= 0.0:
            reasons.append(f"{mechanism}_has_no_target_gradient")
    return tuple(reasons)


def _diagnostic_partition(
    model: torch.nn.Module,
    episodes: Sequence[CompiledFitEpisode],
    *,
    evaluation_id: str,
    device: torch.device,
) -> tuple[tuple[PredictionBundle, ...], Mapping[str, Any] | None]:
    if not episodes:
        return (), None
    predictions = _forward_episodes(model, episodes, device=device)
    evaluation = Evaluator().evaluate(
        predictions,
        tuple(episode.truth for episode in episodes),
        evaluation_id=evaluation_id,
    )
    payload = to_canonical_data(evaluation)
    if not isinstance(payload, Mapping):  # pragma: no cover - dataclass canonicalizes to dict
        raise FitExperimentError("diagnostic EvaluationBundle did not canonicalize to a mapping")
    return predictions, payload


def _model_parameter_state_hash(model: torch.nn.Module) -> str:
    """Hash exact model state while preserving scalar tensor shape metadata."""

    return canonical_hash(
        {
            name: {
                "shape": tuple(value.shape),
                "tensor": value.detach().reshape(-1),
            }
            for name, value in model.state_dict().items()
        }
    )


def checkpoint_corpus_roundtrip(
    trainer: Trainer,
    spec: FitExperimentSpec,
    identity: RunIdentity,
    training: Mapping[str, Any],
    execution: Mapping[str, Any],
    corpus: FitEpisodeCorpus,
    expected: Sequence[PredictionBundle],
    *,
    seed: int,
    device: torch.device,
) -> CheckpointCorpusRoundtrip:
    """Reload exact state and verify one optimizer continuation without mutation.

    The live trainer is temporarily advanced on the next scheduled episode and
    then restored from the saved checkpoint before this function returns.  A
    separately constructed trainer executes the identical continuation.  Exact
    agreement therefore audits optimizer/RNG resume in addition to readout and
    cursor replay, while the later immutable artifact still captures the true
    final training state.
    """

    import tempfile

    expected_items = tuple(expected)
    if len(expected_items) != len(corpus.train_episodes):
        raise FitExperimentError("checkpoint comparison must cover every train episode")
    with tempfile.TemporaryDirectory(prefix="tabu-s1-checkpoint-") as directory:
        path = Path(directory) / "checkpoint.safetensors"
        trainer.save_checkpoint(path)
        saved_step = trainer.step
        restored_model = _build_model(spec, seed=seed, device=device)
        restored = _trainer(restored_model, spec, identity, training, execution)
        restored.load_checkpoint(path)
        restored_step = restored.step
        actual = _forward_episodes(restored_model, corpus.train_episodes, device=device)
        next_episode = corpus.episode_at_update(saved_step)
        next_recipe_hash = next_episode.recipe_hash

        try:
            # Reset global/backend RNG from the same checkpoint before each
            # branch, because Trainer exact-resume owns those RNG states.
            trainer.load_checkpoint(path)
            original_step = trainer.train_step(next_episode.evidence, next_episode.truth)
            original_state_hash = _model_parameter_state_hash(trainer.model)
            original_continuation = _forward_episodes(
                trainer.model, corpus.train_episodes, device=device
            )

            restored.load_checkpoint(path)
            resumed_step = restored.train_step(next_episode.evidence, next_episode.truth)
            restored_state_hash = _model_parameter_state_hash(restored.model)
            restored_continuation = _forward_episodes(
                restored.model, corpus.train_episodes, device=device
            )
        finally:
            # The artifact writer must see precisely the state that entered the
            # roundtrip audit, never either temporary continuation branch.
            trainer.load_checkpoint(path)

    expected_hashes = tuple(item.prediction_hash for item in expected_items)
    actual_hashes = tuple(item.prediction_hash for item in actual)
    original_continuation_hashes = tuple(
        item.prediction_hash for item in original_continuation
    )
    restored_continuation_hashes = tuple(
        item.prediction_hash for item in restored_continuation
    )
    original_loss = float(original_step.loss.total.detach().cpu())
    restored_loss = float(resumed_step.loss.total.detach().cpu())
    continuation_verified = (
        original_step.step == resumed_step.step == saved_step + 1
        and original_loss == restored_loss
        and original_step.prediction.prediction_hash
        == resumed_step.prediction.prediction_hash
        and original_state_hash == restored_state_hash
        and original_continuation_hashes == restored_continuation_hashes
    )
    reloaded = (
        restored_step == saved_step
        and actual_hashes == expected_hashes
        and corpus.episode_at_update(restored_step).recipe_hash == next_recipe_hash
        and continuation_verified
        and trainer.step == saved_step
        and tuple(
            item.prediction_hash
            for item in _forward_episodes(trainer.model, corpus.train_episodes, device=device)
        )
        == expected_hashes
    )
    return CheckpointCorpusRoundtrip(
        reloaded=reloaded,
        continuation_verified=continuation_verified,
        saved_step=saved_step,
        restored_step=restored_step,
        continuation_step=resumed_step.step,
        continuation_loss=restored_loss,
        continuation_state_hash=restored_state_hash,
        next_recipe_hash=next_recipe_hash,
        prediction_hashes=actual_hashes,
        continuation_prediction_hashes=restored_continuation_hashes,
    )


def _run_s1_seed(
    *,
    spec: FitExperimentSpec,
    corpus: FitEpisodeCorpus,
    identity: RunIdentity,
    seed: int,
    device: torch.device,
    output_root: Path,
    preregistration_text: str,
    code_manifest: Mapping[str, Any],
    code_hash: str,
    compiler_manifest: Mapping[str, Any],
    feasibility: FitFeasibilityReport,
    baseline: Mapping[str, Any],
    training: Mapping[str, Any],
    execution: Mapping[str, Any],
    environment: EnvironmentDisclosure,
    environment_payload: Mapping[str, Any],
    command: Sequence[str],
    formal_authorization: FormalAuthorizationContext | None,
    max_updates_override: int | None = None,
    monotonic_clock: Callable[[], float] | None = None,
) -> SeedRunResult:
    """Execute one S1 seed; override hooks are internal test-only."""

    if max_updates_override is not None and (
        type(max_updates_override) is not int
        or max_updates_override <= 0
        or max_updates_override > spec.training.max_updates
    ):
        raise ValueError("max_updates_override must be in [1, training.max_updates]")
    update_budget = (
        spec.training.max_updates if max_updates_override is None else max_updates_override
    )
    clock = time.monotonic if monotonic_clock is None else monotonic_clock
    try:
        source_identity = SourceIdentity.model_validate(code_manifest["source_identity"])
    except (KeyError, ValueError) as exc:
        raise FitExperimentError("fit source identity is invalid") from exc
    source_identity_hash = canonical_hash(source_identity)
    started_at = datetime.now(UTC)
    attempt_nonce = f"{started_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
    attempt_id = derive_attempt_id(run_id=identity.run_id, attempt_nonce=attempt_nonce)
    attempt_directory = output_root / identity.run_id / attempt_id
    all_recipe_hashes = _all_recipe_hashes(corpus)
    observer = get_observer(
        run_id=identity.run_id,
        attempt_id=attempt_id,
        experiment_id=spec.experiment_id,
        contract_id=spec.contract_id,
        seed=seed,
        stage=spec.stage.value,
        environment_payload=environment_payload,
    )
    phase = "feasibility"
    try:
        if feasibility.status is not FeasibilityReportStatus.READY:
            raise FitExperimentError("positive S1 train corpus is not terminal-feasible")

        phase = "build"
        model = _build_model(spec, seed=seed, device=device)
        trainer = _trainer(model, spec, identity, training, execution)
        initial_parameters = _parameter_snapshot(model)

        phase = "initial_evaluation"
        initial_predictions = _forward_episodes(model, corpus.train_episodes, device=device)
        initial_objective = _target_weighted_objective(
            initial_predictions, corpus.train_episodes, device=device
        )

        phase = "mechanism_probe"
        (
            mechanism_source_counts,
            mechanism_active_target_counts,
            mechanism_gradient_norms,
            mechanism_scored_target_count,
        ) = _aggregate_mechanism_probe(
            model,
            corpus.train_episodes,
            contract_id=spec.contract_id,
            device=device,
        )
        gradient_nonzero_by_step: int | None = None
        gradient_group_nonzero_by_step: dict[str, int] = {}
        gradient_group_max_norms: dict[str, float] = {}
        first_episode = corpus.episode_at_update(trainer.step)
        history: list[dict[str, Any]] = [
            {
                "record_type": "step",
                "step": 0,
                "loss": initial_objective,
                "gradient_norm": None,
                "gradient_norms": {},
                "episode_recipe_hash": first_episode.recipe_hash,
                "episode_partition": "train",
                "mechanism_source_counts": mechanism_source_counts,
                "mechanism_active_target_counts": mechanism_active_target_counts,
                "mechanism_scored_target_count": mechanism_scored_target_count,
                "mechanism_gradient_norms": mechanism_gradient_norms,
                "elapsed_seconds": 0.0,
            }
        ]
        observer.log_step(history[-1])

        phase = "train"
        started = clock()
        wall_time_seconds = 0.0
        stop_reason = S1StopReason.MAX_UPDATES
        for _ in range(update_budget):
            # The completed update count is the entire resumable corpus cursor.
            episode = corpus.episode_at_update(trainer.step)
            step = trainer.train_step(episode.evidence, episode.truth)
            elapsed = max(0.0, clock() - started)
            wall_time_seconds = elapsed
            loss_value = float(step.loss.total.detach().cpu())
            if not math.isfinite(loss_value) or not math.isfinite(step.gradient_norm):
                raise _NonfiniteFitError(
                    f"nonfinite S1 training state observed at step {step.step}"
                )
            if step.gradient_norm > 0.0 and gradient_nonzero_by_step is None:
                gradient_nonzero_by_step = step.step
            for group, norm in step.gradient_norms.items():
                gradient_group_max_norms[group] = max(
                    gradient_group_max_norms.get(group, 0.0), float(norm)
                )
                if norm > 0.0 and group not in gradient_group_nonzero_by_step:
                    gradient_group_nonzero_by_step[group] = step.step
            if step.step == 1 or step.step % 10 == 0:
                history.append(
                    {
                        "record_type": "step",
                        "step": step.step,
                        "loss": loss_value,
                        "gradient_norm": step.gradient_norm,
                        "gradient_norms": dict(step.gradient_norms),
                        "episode_recipe_hash": episode.recipe_hash,
                        "episode_partition": episode.partition,
                        "episode_ordinal": episode.ordinal,
                        "elapsed_seconds": elapsed,
                    }
                )
                observer.log_step(history[-1])
            if elapsed >= spec.training.wall_clock_budget_minutes * 60:
                stop_reason = S1StopReason.WALL_CLOCK_BUDGET
                break

        phase = "final_evaluation"
        final_train_predictions = _forward_episodes(model, corpus.train_episodes, device=device)
        final_objective = _target_weighted_objective(
            final_train_predictions, corpus.train_episodes, device=device
        )
        evaluation = Evaluator().evaluate(
            final_train_predictions,
            tuple(episode.truth for episode in corpus.train_episodes),
            evaluation_id=f"{spec.experiment_id}-{seed}-train",
        )
        validation_predictions, validation_diagnostic = _diagnostic_partition(
            model,
            corpus.validation_episodes,
            evaluation_id=f"{spec.experiment_id}-{seed}-validation-diagnostic",
            device=device,
        )
        test_predictions, test_diagnostic = _diagnostic_partition(
            model,
            corpus.test_episodes,
            evaluation_id=f"{spec.experiment_id}-{seed}-test-diagnostic",
            device=device,
        )
        parameter_delta = _parameter_delta_norm(initial_parameters, model)

        phase = "checkpoint"
        checkpoint = checkpoint_corpus_roundtrip(
            trainer,
            spec,
            identity,
            training,
            execution,
            corpus,
            final_train_predictions,
            seed=seed,
            device=device,
        )

        phase = "evaluation"
        families = _aggregate_family_metrics(
            initial=initial_predictions,
            final=final_train_predictions,
            episodes=corpus.train_episodes,
            baseline=baseline,
        )
        fit_evaluation = FitEvaluationBundle(
            evaluation_id=f"{spec.experiment_id}-{seed}-fit",
            experiment_id=spec.experiment_id,
            stage=spec.stage,
            model_seed=seed,
            targets=evaluation.counts["targets"],
            scored_targets=evaluation.counts["scored_targets"],
            coverage=float(evaluation.metrics["coverage"] or 0.0),
            families=families,
            gradient_nonzero_by_step=gradient_nonzero_by_step,
            gradient_group_nonzero_by_step=gradient_group_nonzero_by_step,
            gradient_group_max_norms=gradient_group_max_norms,
            mechanism_source_counts=mechanism_source_counts,
            mechanism_active_target_counts=mechanism_active_target_counts,
            mechanism_scored_target_count=mechanism_scored_target_count,
            mechanism_gradient_norms=mechanism_gradient_norms,
            parameter_delta_norm=parameter_delta,
            nonfinite_seen=False,
            checkpoint_reloaded=checkpoint.reloaded,
        )
        reasons = tuple(
            dict.fromkeys(
                (
                    *_gate_reasons(
                        spec,
                        fit_evaluation,
                        initial_objective=initial_objective,
                        final_objective=final_objective,
                    ),
                    *_s1_mechanism_gate_reasons(spec, fit_evaluation),
                )
            )
        )
        diagnostic = spec.execution.evidence_mode is FitEvidenceMode.DIAGNOSTIC_NONDETERMINISTIC
        verdict = _seed_verdict(fit_evaluation, reasons, diagnostic=diagnostic)
        status = (
            ReceiptStatus.SUCCEEDED
            if verdict in {"pass", "diagnostic_pass"}
            else ReceiptStatus.FAILED
        )
        verdict_text = "\n".join(
            (
                f"# Fit verdict: {verdict}",
                "",
                f"- experiment: `{spec.experiment_id}`",
                f"- contract: `{spec.contract_id}`",
                f"- stage: `{spec.stage.value}`",
                f"- model seed: `{seed}`",
                f"- stop reason: `{stop_reason.value}`",
                f"- training wall time: `{wall_time_seconds:.6f}` seconds",
                f"- reasons: `{', '.join(reasons) if reasons else 'none'}`",
                "- gate scope: all unique train episodes, target-weighted family metrics",
                "- held-out scope: validation/test diagnostics only",
                "- boundary: support-realizable multi-episode synthetic fit only; "
                "no generalization claim",
            )
        )
        summary = {
            "record_type": "summary",
            "fit_evaluation": fit_evaluation,
            "initial_objective": initial_objective,
            "final_objective": final_objective,
            "loss_ratio": final_objective / max(initial_objective, 1.0e-12),
            "configured_max_updates": spec.training.max_updates,
            "effective_max_updates": update_budget,
            "executed_steps": trainer.step,
            "wall_time_seconds": wall_time_seconds,
            "stop_reason": stop_reason.value,
            "next_episode_recipe_hash": checkpoint.next_recipe_hash,
            "checkpoint_saved_step": checkpoint.saved_step,
            "checkpoint_restored_step": checkpoint.restored_step,
            "checkpoint_train_prediction_hashes": checkpoint.prediction_hashes,
            "checkpoint_continuation_verified": checkpoint.continuation_verified,
            "checkpoint_continuation_step": checkpoint.continuation_step,
            "checkpoint_continuation_loss": checkpoint.continuation_loss,
            "checkpoint_continuation_state_hash": checkpoint.continuation_state_hash,
            "checkpoint_continuation_prediction_hashes": (
                checkpoint.continuation_prediction_hashes
            ),
            "train_recipe_hashes": corpus.schedule_realization.train_recipe_hashes,
            "validation_diagnostic": validation_diagnostic,
            "test_diagnostic": test_diagnostic,
            "verdict": verdict,
            "reasons": reasons,
        }
        observer.log_summary(summary)

        phase = "artifact"
        fit_metadata: dict[str, Any] = {
            "attempt_nonce": attempt_nonce,
            "attempt_verdict": verdict,
            "experiment_id": spec.experiment_id,
            "contract_version": spec.contract_version,
            "checkpoint_license_id": "Apache-2.0",
            "stage": spec.stage.value,
            "model_seed": seed,
            "code_hash": code_hash,
            "issuance_status": source_identity.issuance_status,
            "source_identity_hash": source_identity_hash,
            "evidence_mode": spec.execution.evidence_mode,
            "corpus_hash": corpus.corpus_hash,
            "schedule_realization_hash": corpus.schedule_realization.content_hash,
            "train_recipe_hashes": list(corpus.schedule_realization.train_recipe_hashes),
            "validation_recipe_hashes": list(corpus.schedule_realization.validation_recipe_hashes),
            "test_recipe_hashes": list(corpus.schedule_realization.test_recipe_hashes),
            "final_episode_cursor": trainer.step,
            "next_episode_recipe_hash": checkpoint.next_recipe_hash,
            "wall_time_seconds": wall_time_seconds,
            "stop_reason": stop_reason.value,
            "checkpoint_continuation_verified": checkpoint.continuation_verified,
            "checkpoint_continuation_step": checkpoint.continuation_step,
            "checkpoint_continuation_state_hash": checkpoint.continuation_state_hash,
        }
        fit_metadata.update(_tabubase_identity_metadata(spec, model=model))
        all_predictions = final_train_predictions + validation_predictions + test_predictions
        artifacts = write_fit_attempt_artifacts(
            attempt_directory,
            attempt_id=attempt_id,
            run_identity=identity,
            model_id=spec.contract_id,
            dataset_id=corpus.dataset.dataset_id,
            fit_partition=spec.split.fit_partition,
            preregistration_text=preregistration_text,
            resolved_configs={
                "code": code_manifest,
                "experiment": spec,
                "semantic": spec.semantic,
                "training": training,
                "execution": execution,
                "seeds": identity.seeds,
            },
            dataset_manifest={
                "schema": "tabu.fit-dataset-manifest.v1",
                "dataset": spec.dataset,
                "dataset_id": corpus.dataset.dataset_id,
                "dataset_hash": corpus.dataset.dataset_hash,
                "feature_specs": corpus.dataset.feature_specs,
                "row_ids": corpus.dataset.row_ids,
                "metadata": corpus.dataset.metadata,
            },
            split_manifest=spec.split,
            compiler_manifest=compiler_manifest,
            feasibility=feasibility,
            metrics={"summary": summary, "history": tuple(history)},
            evaluation=evaluation,
            predictions=all_predictions,
            episode_recipe_hashes=all_recipe_hashes,
            baselines={
                **baseline,
                "heldout_diagnostics": {
                    "validation": validation_diagnostic,
                    "test": test_diagnostic,
                },
            },
            verdict=verdict_text,
            status=status,
            error=(None if status is ReceiptStatus.SUCCEEDED else "; ".join(reasons)),
            command=tuple(command),
            checkpoint_writer=lambda path, bound=trainer: bound.save_checkpoint(path),
            metadata=fit_metadata,
            formal_authorization=formal_authorization,
        )
        observer.close()
        return SeedRunResult(
            model_seed=seed,
            verdict=verdict,
            fit_evaluation=fit_evaluation,
            artifacts=artifacts,
            error=(None if status is ReceiptStatus.SUCCEEDED else "; ".join(reasons)),
        )
    except Exception as error:
        boundary = _exception_boundary(
            phase,
            error,
            formal=source_identity.issuance_status == "formal",
        )
        failure_fixture = _FailureFixtureAdapter(
            dataset=corpus.dataset,
            recipe=corpus.train_episodes[0].recipe,
        )
        artifacts = _runtime_failure_artifacts(
            destination=attempt_directory,
            attempt_id=attempt_id,
            attempt_nonce=attempt_nonce,
            identity=identity,
            spec=spec,
            fixture=failure_fixture,  # type: ignore[arg-type]
            episode_recipe_hashes=all_recipe_hashes,
            preregistration_text=preregistration_text,
            code_manifest=code_manifest,
            compiler_manifest=compiler_manifest,
            feasibility=feasibility,
            baseline=baseline,
            training=training,
            execution=execution,
            environment=environment,
            environment_payload=environment_payload,
            command=command,
            formal_authorization=formal_authorization,
            phase=phase,
            error=error,
            started_at=started_at,
        )
        if boundary["code"] == "out_of_memory" and device.type == "cuda":
            torch.cuda.empty_cache()
        observer.close()
        return SeedRunResult(
            model_seed=seed,
            verdict="failed",
            fit_evaluation=None,
            artifacts=artifacts,
            failure_phase=phase,
            error=f"{boundary['exception_type']}: {boundary['message']}",
        )


def run_s1_experiment(
    preregistration: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str],
    repository: str | os.PathLike[str] | None = None,
    command: Sequence[str] = (),
    formal: bool = False,
    source_reviewed: bool = False,
    authorization_catalog: str | os.PathLike[str] | None = None,
    source_identity: SourceIdentity | None = None,
    distribution_artifact: bytes | str | os.PathLike[str] | None = None,
    distribution_lock: bytes | str | os.PathLike[str] | None = None,
) -> ExperimentRunResult:
    """Run all frozen seeds for one registered S1 synthetic experiment."""

    source_path = Path(preregistration)
    preregistration_text = source_path.read_text(encoding="utf-8")
    spec = load_fit_experiment(source_path)
    if spec.stage is not FitStage.S1:
        raise FitExperimentError("run_s1_experiment requires stage S1")
    if source_reviewed and not formal:
        raise FitExperimentError("source_reviewed is only meaningful for a formal request")
    if formal and authorization_catalog is None:
        raise FitExperimentError(
            "formal runs require authorization_catalog; source_reviewed cannot self-authorize"
        )
    if not formal and authorization_catalog is not None:
        raise FitExperimentError("authorization_catalog is only valid for a formal request")
    authorization_context: FormalAuthorizationContext | None = None
    verified_authorization: VerifiedFormalAuthorization | None = None
    if formal:
        assert authorization_catalog is not None
        _assert_formal_output_root_safe(output_root, repository=repository)
        authorization_context, verified_authorization = _resolve_formal_authorization(
            authorization_catalog,
            spec=spec,
            preregistration_path=source_path,
            preregistration_text=preregistration_text,
            repository=repository,
        )
    if formal and spec.execution.evidence_mode is FitEvidenceMode.DIAGNOSTIC_NONDETERMINISTIC:
        raise FitExperimentError(
            "nondeterministic diagnostic execution cannot issue formal or Gate 1 evidence"
        )

    corpus = build_s1_corpus(spec)
    compiler_manifest = s1_compiler_binding_manifest(spec, corpus)
    targets, feasibility = assess_s1_feasibility(spec, corpus)
    baseline = trivial_baseline(targets)
    device = _device(spec)
    code_manifest = source_tree_manifest(
        repository,
        preregistration=source_path,
        request_formal=formal,
        reviewed=authorization_context is not None,
        source_identity=source_identity,
        distribution_artifact=distribution_artifact,
        distribution_lock=distribution_lock,
    )
    resolved_source_identity = SourceIdentity.model_validate(code_manifest["source_identity"])
    if formal and resolved_source_identity.issuance_status != "formal":
        reasons = ", ".join(resolved_source_identity.reasons)
        raise FitExperimentError(
            f"formal receipt refused by SourceIdentity: {reasons or 'source is not formal'}"
        )
    if authorization_context is not None:
        assert verified_authorization is not None
        try:
            verified_authorization = verify_formal_authorization(
                authorization_context,
                preregistration_text=preregistration_text,
                live_source_identity=resolved_source_identity,
                expected_summary=verified_authorization.summary,
            )
        except (FormalAuthorizationError, TypeError, ValueError) as exc:
            raise FitExperimentError(
                "live formal SourceIdentity canonical hash does not match authorization catalog"
            ) from exc
    recorded_command = _authorization_safe_command(
        command,
        None if verified_authorization is None else verified_authorization.summary,
        output_root=output_root if formal else None,
        preregistration_path=source_path if formal else None,
    )
    if formal:
        assert_public_payload_safe(
            {"command": recorded_command}, location="formal recorded command"
        )
    code_hash = canonical_hash(code_manifest)
    compiler_hash = canonical_hash(compiler_manifest)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(spec.execution.deterministic_algorithms)
    results: list[SeedRunResult] = []
    try:
        try:
            training, execution, environment, environment_payload = _training_and_execution_configs(
                spec, device=device
            )
            identities = tuple(
                _identity(
                    spec,
                    corpus,  # type: ignore[arg-type]
                    seed=seed,
                    code_hash=code_hash,
                    compiler_hash=compiler_hash,
                    training=training,
                    execution=execution,
                )
                for seed in spec.seeds.model_seeds
            )
        except Exception as error:
            raise FitExperimentError(
                "S1 preflight failed before RunIdentity formation; the current public "
                "Receipt schema cannot safely represent an identity-free attempt"
            ) from error
        for seed, identity in zip(spec.seeds.model_seeds, identities, strict=True):
            results.append(
                _run_s1_seed(
                    spec=spec,
                    corpus=corpus,
                    identity=identity,
                    seed=seed,
                    device=device,
                    output_root=Path(output_root),
                    preregistration_text=preregistration_text,
                    code_manifest=code_manifest,
                    code_hash=code_hash,
                    compiler_manifest=compiler_manifest,
                    feasibility=feasibility,
                    baseline=baseline,
                    training=training,
                    execution=execution,
                    environment=environment,
                    environment_payload=environment_payload,
                    command=recorded_command,
                    formal_authorization=authorization_context,
                )
            )
        aggregate = _write_experiment_aggregate(
            output_root=Path(output_root), spec=spec, results=results
        )
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    return ExperimentRunResult(
        experiment_id=spec.experiment_id,
        stage=spec.stage,
        seed_results=tuple(results),
        aggregate=aggregate,
    )


__all__ = [
    "CheckpointCorpusRoundtrip",
    "assess_s1_feasibility",
    "build_s1_corpus",
    "checkpoint_corpus_roundtrip",
    "run_s1_experiment",
    "s1_compiler_binding_manifest",
    "validate_s1_corpus_binding",
]
