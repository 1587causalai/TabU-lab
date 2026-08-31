"""Deterministic snapshot diff and minimum-rerun impact classification."""

from __future__ import annotations

from tabu_lab.contracts import canonical_hash

from .models import (
    CompatibilityDisposition,
    EvaluationProtocolNode,
    ImpactAction,
    ImpactDisposition,
    ImpactReport,
    NodeRef,
    ResolvedProgramSnapshot,
    SnapshotChange,
    StateProjectionNode,
)
from .repository import EvolutionRepository

_TRAINING_INPUT_SLOTS = frozenset(
    {
        "model_contract",
        "component_graph",
        "world_mixture",
        "sampling_policy",
        "objective_bundle",
        "training_recipe",
        "state_projection",
    }
)
_MODEL_SLOTS = frozenset({"model_contract", "component_graph"})


def diff_snapshots(
    source: ResolvedProgramSnapshot,
    target: ResolvedProgramSnapshot,
) -> tuple[SnapshotChange, ...]:
    changes: list[SnapshotChange] = []
    for slot in sorted(set(source.slots).union(target.slots)):
        left = source.slots.get(slot)
        right = target.slots.get(slot)
        if left != right:
            changes.append(SnapshotChange(slot=slot, source=left, target=right))
    return tuple(changes)


def _slot_action(change: SnapshotChange) -> ImpactAction:
    if change.slot == "evaluation_protocol":
        disposition = ImpactDisposition.RESCORE
        reason = "evaluation protocol changed; model training is not an upstream dependency"
    elif change.slot == "state_projection":
        disposition = ImpactDisposition.WARM_START_AVAILABLE
        reason = "target snapshot explicitly selected a validated state projection"
    else:
        disposition = ImpactDisposition.RETRAIN
        reason = f"{change.slot} participates in training identity"
    return ImpactAction(
        object_kind=f"slot:{change.slot}",
        disposition=disposition,
        reason=reason,
    )


def _warm_start_action(
    repository: EvolutionRepository,
    source: ResolvedProgramSnapshot,
    target: ResolvedProgramSnapshot,
) -> ImpactAction:
    projection_ref = target.slots.get("state_projection")
    if projection_ref is None:
        return ImpactAction(
            object_kind="initialization",
            disposition=ImpactDisposition.BLOCKED,
            reason="warm start is unavailable because no StateProjection is selected",
        )
    projection = repository.node(
        NodeRef(node_id=projection_ref.node_id, version=projection_ref.version)
    )
    if not isinstance(projection, StateProjectionNode) or not projection.verified:
        return ImpactAction(
            object_kind="initialization",
            disposition=ImpactDisposition.BLOCKED,
            reason="selected StateProjection is not verified",
        )
    source_graph = source.slots["component_graph"]
    target_graph = target.slots["component_graph"]
    edges = repository.compatibility_edges(
        source_graph,
        target_graph,
        CompatibilityDisposition.WARM_START_AVAILABLE,
    )
    if not edges:
        return ImpactAction(
            object_kind="initialization",
            disposition=ImpactDisposition.BLOCKED,
            reason="default incompatibility applies: no verified warm-start compatibility edge",
        )
    if (
        projection.source_graph.ref != source_graph.ref
        or projection.target_graph.ref != target_graph.ref
    ):
        return ImpactAction(
            object_kind="initialization",
            disposition=ImpactDisposition.BLOCKED,
            reason="StateProjection endpoints do not match source and target snapshots",
        )
    return ImpactAction(
        object_kind="initialization",
        disposition=ImpactDisposition.WARM_START_AVAILABLE,
        reason=(
            f"validated by {projection.ref} and {edges[0].ref}; "
            "new run identity is still required"
        ),
    )


def impact_report(
    repository: EvolutionRepository,
    source: ResolvedProgramSnapshot,
    target: ResolvedProgramSnapshot,
) -> ImpactReport:
    changes = diff_snapshots(source, target)
    changed_slots = {change.slot for change in changes}
    actions: list[ImpactAction] = [_slot_action(change) for change in changes]

    unchanged_slots = sorted(set(source.slots).intersection(target.slots) - changed_slots)
    actions.extend(
        ImpactAction(
            object_kind=f"slot:{slot}",
            disposition=ImpactDisposition.UNCHANGED,
            reason="resolved node identity is unchanged",
        )
        for slot in unchanged_slots
    )

    training_changed = bool(changed_slots.intersection(_TRAINING_INPUT_SLOTS))
    model_changed = bool(changed_slots.intersection(_MODEL_SLOTS))
    evaluation_changed = "evaluation_protocol" in changed_slots

    if not changes:
        actions.extend(
            (
                ImpactAction(
                    object_kind="checkpoint",
                    disposition=ImpactDisposition.REUSE_EXACT,
                    reason="the complete resolved snapshot is identical",
                ),
                ImpactAction(
                    object_kind="predictions",
                    disposition=ImpactDisposition.REUSE_EXACT,
                    reason="model, data, and evaluation inputs are identical",
                ),
                ImpactAction(
                    object_kind="evaluation",
                    disposition=ImpactDisposition.REUSE_EXACT,
                    reason="the evaluation protocol is identical",
                ),
            )
        )
    elif training_changed:
        actions.append(
            ImpactAction(
                object_kind="training_run",
                disposition=ImpactDisposition.RETRAIN,
                reason="at least one training-identity node changed",
            )
        )
        actions.append(_warm_start_action(repository, source, target))
        actions.append(
            ImpactAction(
                object_kind="predictions",
                disposition=ImpactDisposition.RERUN_INFERENCE,
                reason=(
                    "model computation changed" if model_changed else "target checkpoint changes"
                ),
            )
        )
        actions.append(
            ImpactAction(
                object_kind="evaluation",
                disposition=ImpactDisposition.RESCORE,
                reason="new predictions require a new evaluation receipt",
            )
        )
    elif evaluation_changed:
        source_eval_ref = source.slots["evaluation_protocol"]
        target_eval_ref = target.slots["evaluation_protocol"]
        source_eval = repository.node(source_eval_ref.ref)
        target_eval = repository.node(target_eval_ref.ref)
        assert isinstance(source_eval, EvaluationProtocolNode)
        assert isinstance(target_eval, EvaluationProtocolNode)
        compatible_predictions = (
            target_eval.can_rescore_from_predictions
            and source_eval.prediction_compatibility_key
            == target_eval.prediction_compatibility_key
            and source_eval.prediction_interface == target_eval.prediction_interface
        )
        actions.append(
            ImpactAction(
                object_kind="checkpoint",
                disposition=ImpactDisposition.REUSE_EXACT,
                reason="evaluation-only change does not alter training identity",
            )
        )
        actions.append(
            ImpactAction(
                object_kind="predictions",
                disposition=(
                    ImpactDisposition.REUSE_EXACT
                    if compatible_predictions
                    else ImpactDisposition.RERUN_INFERENCE
                ),
                reason=(
                    "stored PredictionBundle is compatible with the target scorer"
                    if compatible_predictions
                    else "target evaluator requires a different prediction artifact"
                ),
            )
        )
        actions.append(
            ImpactAction(
                object_kind="evaluation",
                disposition=(
                    ImpactDisposition.RESCORE
                    if compatible_predictions
                    else ImpactDisposition.RERUN_INFERENCE
                ),
                reason=(
                    "only scoring changes"
                    if compatible_predictions
                    else "inference must be regenerated before scoring"
                ),
            )
        )

    ordered = tuple(sorted(actions, key=lambda action: action.object_kind))
    payload = {
        "schema_version": "tabu.impact-report.v1",
        "source_snapshot_hash": source.snapshot_hash,
        "target_snapshot_hash": target.snapshot_hash,
        "changes": changes,
        "actions": ordered,
    }
    return ImpactReport(**payload, report_hash=canonical_hash(payload))


__all__ = ["diff_snapshots", "impact_report"]
