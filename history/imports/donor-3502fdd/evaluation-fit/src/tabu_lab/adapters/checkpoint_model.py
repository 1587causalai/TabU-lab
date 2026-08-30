"""Executable Evaluation Foundry adapter for cataloged TabU checkpoints.

The foundry's generic ``BlindExample`` contract cannot infer a TabU table,
codebook, graph, or target coordinate from arbitrary feature dictionaries.
This adapter therefore accepts one deliberately narrow input: every blind
example must carry a complete, precompiled, truth-free ``EvidenceEpisode`` and
an explicit readout selector.  Dataset adapters remain responsible for that
projection and the evaluator still owns every held-out target value.

Construction fails closed unless the supplied ``ModelSpec``, semantic config,
and training compiler manifest reproduce the hashes embedded in the producer
checkpoint.  Checkpoint bytes are independently bound by ``ModelArtifact``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from tabu_lab.catalog import CatalogIndex, ModelArtifact, ModelArtifactStatus
from tabu_lab.contracts import (
    EvidenceEpisode,
    FeatureKind,
    FeatureSpec,
    GraphTopology,
    PredictionStatus,
    canonical_hash,
)
from tabu_lab.evaluation.foundry import (
    AdapterKind,
    AdapterLaunchSpec,
    AdapterSpec,
    BlindExample,
    FailureCategory,
    PreparedExample,
    RawPrediction,
    ScenarioSpec,
    TargetKind,
)
from tabu_lab.experiments import ModelSemanticConfig
from tabu_lab.models import DenseReferenceModel, ReferenceConfig, build_model
from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE
from tabu_lab.registry import ModelSpec, get_model_spec

from .cataloged_checkpoint import (
    CatalogedCheckpointError,
    VerifiedCatalogedCheckpoint,
    resolve_model_artifact,
    verify_artifact_checkpoint,
)

ADAPTER_VERSION = "1.0.0"
EPISODE_PAYLOAD_KEY = "tabu_episode"
EPISODE_PAYLOAD_SCHEMA = "tabu.eval-evidence-episode-payload.v1"
READOUT_SELECTOR_KEY = "tabu_readout"
READOUT_SELECTOR_SCHEMA = "tabu.eval-readout-selector.v1"

_EPISODE_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "dataset_id",
        "source_partition",
        "fit_partition",
        "row_ids",
        "feature_specs",
        "forward_values",
        "origin_states",
        "forward_roles",
        "graph_topology",
        "metadata",
    }
)
_READOUT_FIELDS = frozenset({"schema_version", "row_id", "feature_name"})


class ExplicitEpisodeInputError(ValueError):
    """A blind example is not an unambiguous truth-free TabU episode."""


def _adapter_spec(artifact: ModelArtifact, *, profile_id: str | None) -> AdapterSpec:
    return AdapterSpec(
        adapter_id=f"tabu-checkpoint-{artifact.artifact_id}",
        adapter_version=ADAPTER_VERSION,
        kind=AdapterKind.MODEL,
        fit_iterations=0,
        device_class="single_device",
        deterministic=True,
        contract_id=artifact.contract_id,
        artifact_id=artifact.artifact_id,
        profile_id=profile_id,
    )


def _resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise CatalogedCheckpointError("checkpoint adapter device is invalid") from error
    if device.type == "cpu" and device.index is None:
        return device
    if device.type == "mps" and device.index is None:
        if not torch.backends.mps.is_available():
            raise CatalogedCheckpointError("checkpoint adapter requested unavailable MPS")
        return device
    if device.type == "cuda" and device.index is not None:
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise CatalogedCheckpointError("checkpoint adapter requested unavailable CUDA device")
        return device
    raise CatalogedCheckpointError("checkpoint adapter supports cpu, mps, or indexed cuda only")


def _reference_model(
    *,
    artifact: ModelArtifact,
    model_spec: ModelSpec,
    semantic: ModelSemanticConfig,
) -> DenseReferenceModel:
    if semantic.reference.backend != "dense_reference_v0":
        raise CatalogedCheckpointError("checkpoint adapter supports dense_reference_v0 only")
    contract_id = artifact.contract_id
    if semantic.categorical_terminal.value != "nadaraya_watson":
        raise CatalogedCheckpointError(
            "checkpoint adapter cannot reconstruct another categorical terminal"
        )

    packaged_spec = get_model_spec(artifact.contract_id)
    if canonical_hash(packaged_spec.model_dump(mode="json")) != canonical_hash(
        model_spec.model_dump(mode="json")
    ):
        raise CatalogedCheckpointError(
            "installed model contract differs from the checkpoint-bound ModelSpec"
        )

    reference_values = semantic.reference.model_dump(mode="python")
    reference_values.pop("backend")
    reference_values["block_kind"] = semantic.dynamics.block_kind
    kwargs: dict[str, Any] = {
        "config": ReferenceConfig(**reference_values),
        "numeric_terminal": semantic.numeric_terminal.value,
    }
    if contract_id in {"tabuf", "tabul", "tabufl", "tabu4rec"}:
        if semantic.augmented_readout_geometry is None:
            raise CatalogedCheckpointError("augmented checkpoint lacks readout geometry")
        kwargs["readout_geometry"] = semantic.augmented_readout_geometry.value
    if contract_id in {"tabul", "tabufl"}:
        if not semantic.label_columns or semantic.label_address_plan is None:
            raise CatalogedCheckpointError("supervised checkpoint lacks label address semantics")
        kwargs["label_columns"] = semantic.label_columns
        kwargs["label_address_plan"] = semantic.label_address_plan.value
    if contract_id == "tabu4graph":
        if semantic.target_feature is None or semantic.graph_unit_receiver_plan is None:
            raise CatalogedCheckpointError("graph checkpoint lacks target/receiver semantics")
        kwargs["target_feature"] = semantic.target_feature
        kwargs["unit_receiver_plan"] = semantic.graph_unit_receiver_plan.value
    if contract_id == "tabu4rec":
        if semantic.recommendation_address_plan is None:
            raise CatalogedCheckpointError("recommendation checkpoint lacks address semantics")
        kwargs["recommendation_address_plan"] = semantic.recommendation_address_plan.value
        if semantic.rec_axis_summary_dim is not None:
            kwargs["rec_axis_summary_dim"] = semantic.rec_axis_summary_dim
        if semantic.rec_matched_residual_scale is not None:
            kwargs["rec_matched_residual_scale"] = semantic.rec_matched_residual_scale
    if contract_id == "tabu.cell.base":
        if semantic.profile_id is None:
            raise CatalogedCheckpointError(
                "tabu.cell.base checkpoint lacks a declared profile; "
                "the profile is never inferred from the contract defaults"
            )
        kwargs["profile"] = semantic.profile_id

    model = build_model(contract_id, **kwargs)
    if not isinstance(model, DenseReferenceModel):
        raise CatalogedCheckpointError("checkpoint contract has no executable reference model")
    model.semantic_config_hash = semantic.content_hash
    return model


def _model_profile_id(model: DenseReferenceModel) -> str | None:
    """Read the profile a rebuilt model resolved to, when it declares one.

    Profile-bearing contracts such as ``tabu.cell.base`` keep two disjoint
    evidence profiles under one contract id.  The rebuild path therefore has to
    report which profile the loaded weights actually belong to, so a scenario
    can refuse a checkpoint that was trained under a different one.
    """

    profile = getattr(model, "profile", None)
    if profile is None:
        return None
    value = getattr(profile, "value", profile)
    return value if isinstance(value, str) else None


def _load_checkpoint_model_state(
    *,
    model: DenseReferenceModel,
    checkpoint_path: Path,
) -> None:
    from safetensors import safe_open

    try:
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
            model_state = {
                name.removeprefix("model."): checkpoint.get_tensor(name)
                for name in checkpoint.keys()  # noqa: SIM118 - safe_open is not iterable
                if name.startswith("model.")
            }
    except Exception as error:
        raise CatalogedCheckpointError("checkpoint model state cannot be read") from error

    expected = model.state_dict()
    if set(model_state) != set(expected):
        raise CatalogedCheckpointError(
            "checkpoint model tensor set does not match reconstructed model"
        )
    for name, expected_tensor in expected.items():
        observed = model_state[name]
        if observed.shape != expected_tensor.shape or observed.dtype != expected_tensor.dtype:
            raise CatalogedCheckpointError(
                f"checkpoint model tensor schema differs at {name!r}"
            )
        if observed.is_floating_point() and observed.dtype is not DEFAULT_FLOAT_DTYPE:
            raise CatalogedCheckpointError("checkpoint adapter supports float32 model tensors only")
    try:
        model.load_state_dict(model_state, strict=True)
    except (RuntimeError, ValueError) as error:
        raise CatalogedCheckpointError("checkpoint state cannot be loaded strictly") from error


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExplicitEpisodeInputError(f"{field_name} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise ExplicitEpisodeInputError(f"{field_name} keys must be strings")
    return dict(value)


def _episode_from_blind_example(
    example: BlindExample,
) -> tuple[EvidenceEpisode, int, int, str, str]:
    if set(example.features) != {EPISODE_PAYLOAD_KEY}:
        raise ExplicitEpisodeInputError(
            f"blind features must contain exactly {EPISODE_PAYLOAD_KEY!r}"
        )
    if set(example.context) != {READOUT_SELECTOR_KEY}:
        raise ExplicitEpisodeInputError(
            f"blind context must contain exactly {READOUT_SELECTOR_KEY!r}"
        )
    payload = _mapping(example.features[EPISODE_PAYLOAD_KEY], field_name=EPISODE_PAYLOAD_KEY)
    if set(payload) != _EPISODE_FIELDS:
        raise ExplicitEpisodeInputError("evidence episode payload has missing or unknown fields")
    if payload["schema_version"] != EPISODE_PAYLOAD_SCHEMA:
        raise ExplicitEpisodeInputError("unsupported evidence episode payload schema")
    selector = _mapping(
        example.context[READOUT_SELECTOR_KEY],
        field_name=READOUT_SELECTOR_KEY,
    )
    if set(selector) != _READOUT_FIELDS:
        raise ExplicitEpisodeInputError("readout selector has missing or unknown fields")
    if selector["schema_version"] != READOUT_SELECTOR_SCHEMA:
        raise ExplicitEpisodeInputError("unsupported readout selector schema")

    raw_feature_specs = payload["feature_specs"]
    if not isinstance(raw_feature_specs, list):
        raise ExplicitEpisodeInputError("feature_specs must be a JSON list")
    string_fields = ("episode_id", "dataset_id", "source_partition", "fit_partition")
    if any(
        not isinstance(payload[field_name], str) or not payload[field_name].strip()
        for field_name in string_fields
    ):
        raise ExplicitEpisodeInputError("episode identity and partition fields must be strings")
    row_ids = payload["row_ids"]
    if (
        not isinstance(row_ids, list)
        or not row_ids
        or any(not isinstance(row_id, str) or not row_id.strip() for row_id in row_ids)
    ):
        raise ExplicitEpisodeInputError("row_ids must be a non-empty JSON string list")
    for field_name in ("forward_values", "origin_states", "forward_roles"):
        if not isinstance(payload[field_name], list):
            raise ExplicitEpisodeInputError(f"{field_name} must be a JSON list")
    try:
        feature_specs = tuple(
            FeatureSpec(**_mapping(item, field_name="feature_spec"))
            for item in raw_feature_specs
        )
        topology_payload = payload["graph_topology"]
        graph_topology = None
        if topology_payload is not None:
            topology = _mapping(topology_payload, field_name="graph_topology")
            if set(topology) != {"node_ids", "adjacency", "direction"}:
                raise ExplicitEpisodeInputError(
                    "graph_topology has missing or unknown fields"
                )
            graph_topology = GraphTopology(**topology)
        metadata = _mapping(payload["metadata"], field_name="metadata")
        episode = EvidenceEpisode(
            episode_id=payload["episode_id"],
            dataset_id=payload["dataset_id"],
            source_partition=payload["source_partition"],
            fit_partition=payload["fit_partition"],
            row_ids=tuple(row_ids),
            feature_names=tuple(spec.name for spec in feature_specs),
            feature_specs=feature_specs,
            forward_values=torch.as_tensor(payload["forward_values"], dtype=DEFAULT_FLOAT_DTYPE),
            origin_states=payload["origin_states"],
            forward_roles=payload["forward_roles"],
            graph_topology=graph_topology,
            metadata=metadata,
        )
    except ExplicitEpisodeInputError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ExplicitEpisodeInputError("invalid truth-free EvidenceEpisode payload") from error

    if int(episode.target_mask.sum().item()) != 1:
        raise ExplicitEpisodeInputError("each explicit evaluation episode needs exactly one target")
    row_id = selector["row_id"]
    feature_name = selector["feature_name"]
    if not isinstance(row_id, str) or not isinstance(feature_name, str):
        raise ExplicitEpisodeInputError("readout row_id and feature_name must be strings")
    try:
        row = episode.row_ids.index(row_id)
        feature = episode.feature_names.index(feature_name)
    except ValueError as error:
        raise ExplicitEpisodeInputError(
            "readout selector is outside the evidence episode"
        ) from error
    if not bool(episode.target_mask[row, feature]):
        raise ExplicitEpisodeInputError("readout selector does not identify the episode target")

    feature_kind = episode.feature_specs[feature].kind
    expected_target_kind = (
        TargetKind.NUMERIC if feature_kind is FeatureKind.NUMERIC else TargetKind.CATEGORICAL
    )
    if example.target_kind is not expected_target_kind:
        raise ExplicitEpisodeInputError("blind target kind differs from selected feature schema")
    return episode, row, feature, row_id, feature_name


def _at_cell(
    value: torch.Tensor,
    row: int,
    feature: int,
    *,
    table_shape: tuple[int, int],
) -> torch.Tensor:
    if value.ndim < 2:
        raise ExplicitEpisodeInputError("model prediction tensor has no row/feature axes")
    if tuple(value.shape[:2]) == table_shape:
        return value[row, feature]
    if value.ndim >= 3 and value.shape[0] == 1 and tuple(value.shape[1:3]) == table_shape:
        return value[0, row, feature]
    raise ExplicitEpisodeInputError(
        "model prediction tensor axes differ from the explicit episode table"
    )


class CatalogedCheckpointModelAdapter:
    """Load one verified checkpoint and evaluate explicit truth-free episodes."""

    def __init__(
        self,
        *,
        artifact: Mapping[str, object],
        checkpoint_path: str,
        model_spec: Mapping[str, object],
        semantic_config: Mapping[str, object],
        compiler_manifest: Mapping[str, object],
        device: str = "cpu",
    ) -> None:
        try:
            resolved_artifact = ModelArtifact.model_validate(artifact)
            resolved_model_spec = ModelSpec.model_validate(model_spec)
            resolved_semantic = ModelSemanticConfig.model_validate(semantic_config)
        except ValueError as error:
            raise CatalogedCheckpointError("checkpoint adapter manifest is invalid") from error
        if resolved_artifact.status is ModelArtifactStatus.RETRACTED:
            raise CatalogedCheckpointError("retracted model artifacts cannot be evaluated")
        source = Path(checkpoint_path)
        verified = verify_artifact_checkpoint(resolved_artifact, source)
        if resolved_model_spec.contract_id != resolved_artifact.contract_id:
            raise CatalogedCheckpointError("ModelSpec contract differs from ModelArtifact")
        if resolved_model_spec.contract_version != resolved_artifact.contract_version:
            raise CatalogedCheckpointError("ModelSpec version differs from ModelArtifact")
        if canonical_hash(resolved_model_spec.model_dump(mode="json")) != (
            verified.model_spec_sha256
        ):
            raise CatalogedCheckpointError("ModelSpec hash differs from checkpoint metadata")
        if resolved_semantic.content_hash != verified.semantic_config_sha256:
            raise CatalogedCheckpointError("semantic config hash differs from checkpoint metadata")
        if canonical_hash(dict(compiler_manifest)) != verified.compiler_sha256:
            raise CatalogedCheckpointError("compiler manifest hash differs from RunIdentity")

        resolved_device = _resolve_device(device)
        model = _reference_model(
            artifact=resolved_artifact,
            model_spec=resolved_model_spec,
            semantic=resolved_semantic,
        )
        _load_checkpoint_model_state(model=model, checkpoint_path=source)
        model.to(device=resolved_device, dtype=DEFAULT_FLOAT_DTYPE)
        model.eval()

        self._artifact = resolved_artifact
        self._verified = verified
        self._device = resolved_device
        self._model = model
        self._profile_id = _model_profile_id(model)
        self._spec = _adapter_spec(resolved_artifact, profile_id=self._profile_id)

    @property
    def spec(self) -> AdapterSpec:
        return self._spec

    @property
    def profile_id(self) -> str | None:
        """The profile the rebuilt model resolved to, when it declares one."""

        return self._profile_id

    @property
    def verified_checkpoint(self) -> VerifiedCatalogedCheckpoint:
        return self._verified

    def predict(
        self,
        *,
        scenario: ScenarioSpec,
        fit_examples: Sequence[PreparedExample],
        examples: Sequence[BlindExample],
        seed: int,
    ) -> tuple[RawPrediction, ...]:
        del fit_examples  # support must already be inside each truth-free episode
        if self._artifact.contract_id not in scenario.applicable_contracts:
            raise ExplicitEpisodeInputError("checkpoint contract is not applicable to scenario")
        if (
            scenario.applicable_profiles
            and self._profile_id is not None
            and self._profile_id not in scenario.applicable_profiles
        ):
            raise ExplicitEpisodeInputError("checkpoint profile is not applicable to scenario")
        if type(seed) is not int or seed < 0:
            raise ExplicitEpisodeInputError("evaluation seed must be a non-negative integer")
        torch.manual_seed(seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        elif self._device.type == "mps":
            torch.mps.manual_seed(seed)

        outputs: list[RawPrediction] = []
        with torch.inference_mode():
            for example in examples:
                episode, row, feature, row_id, feature_name = _episode_from_blind_example(
                    example
                )
                if episode.dataset_id != scenario.dataset.dataset_id:
                    raise ExplicitEpisodeInputError(
                        "evidence episode dataset differs from scenario"
                    )
                prediction = self._model(episode)
                table_shape = tuple(episode.forward_values.shape)
                support_available = bool(
                    _at_cell(
                        prediction.auxiliaries["support_available"],
                        row,
                        feature,
                        table_shape=table_shape,
                    ).item()
                )
                unsupported = bool(
                    _at_cell(
                        prediction.auxiliaries["unsupported_target_mask"],
                        row,
                        feature,
                        table_shape=table_shape,
                    ).item()
                )
                diagnostics = {
                    "artifact_id": self._artifact.artifact_id,
                    "checkpoint_sha256": self._verified.checkpoint_sha256,
                    "compiler_sha256": self._verified.compiler_sha256,
                    "evidence_sha256": episode.evidence_hash,
                    "model_spec_sha256": self._verified.model_spec_sha256,
                    "prediction_sha256": prediction.prediction_hash,
                    "readout_feature_name": feature_name,
                    "readout_row_id": row_id,
                    "semantic_config_sha256": self._verified.semantic_config_sha256,
                }
                if unsupported or not support_available:
                    outputs.append(
                        RawPrediction(
                            example_id=example.example_id,
                            abstained=True,
                            diagnostics=diagnostics,
                            failure_category=FailureCategory.MODEL,
                            failure_code=(
                                "checkpoint_model_unsupported_target"
                                if unsupported
                                else "checkpoint_model_no_support"
                            ),
                        )
                    )
                    continue

                if example.target_kind is TargetKind.NUMERIC:
                    entry = prediction.entries.get("numeric")
                    if (
                        entry is None
                        or entry.status is not PredictionStatus.OK
                        or entry.values is None
                    ):
                        raise ExplicitEpisodeInputError("model omitted numeric prediction entry")
                    value = float(
                        _at_cell(
                            entry.values,
                            row,
                            feature,
                            table_shape=table_shape,
                        ).item()
                    )
                    outputs.append(
                        RawPrediction(
                            example_id=example.example_id,
                            value=value,
                            diagnostics=diagnostics,
                        )
                    )
                    continue

                entry = prediction.entries.get("distribution")
                if (
                    entry is None
                    or entry.status is not PredictionStatus.OK
                    or entry.values is None
                ):
                    raise ExplicitEpisodeInputError(
                        "model omitted categorical distribution entry"
                    )
                domain = episode.feature_specs[feature].domain
                probabilities_tensor = (
                    _at_cell(
                        entry.values,
                        row,
                        feature,
                        table_shape=table_shape,
                    )
                    .detach()
                    .cpu()
                )
                domain_mask = prediction.auxiliaries.get("categorical_domain_mask")
                if domain_mask is None:
                    raise ExplicitEpisodeInputError(
                        "categorical prediction omitted its domain mask"
                    )
                if domain_mask.ndim == 2 and domain_mask.shape[0] == table_shape[1]:
                    active_domain = domain_mask[feature]
                elif (
                    domain_mask.ndim == 3
                    and domain_mask.shape[0] == 1
                    and domain_mask.shape[1] == table_shape[1]
                ):
                    active_domain = domain_mask[0, feature]
                else:
                    raise ExplicitEpisodeInputError(
                        "categorical domain mask axes differ from the episode schema"
                    )
                active_domain = active_domain.detach().cpu().bool()
                if (
                    probabilities_tensor.ndim != 1
                    or active_domain.ndim != 1
                    or active_domain.numel() != probabilities_tensor.numel()
                    or int(active_domain.sum().item()) != len(domain)
                ):
                    raise ExplicitEpisodeInputError(
                        "categorical distribution differs from declared feature domain"
                    )
                probabilities_tensor = probabilities_tensor[active_domain]
                raw_probabilities = [float(value) for value in probabilities_tensor.tolist()]
                total = sum(raw_probabilities)
                if total <= 0.0:
                    raise ExplicitEpisodeInputError("categorical distribution has no mass")
                normalized = [value / total for value in raw_probabilities]
                selected = max(range(len(domain)), key=normalized.__getitem__)
                outputs.append(
                    RawPrediction(
                        example_id=example.example_id,
                        value=domain[selected],
                        probabilities=dict(zip(domain, normalized, strict=True)),
                        diagnostics=diagnostics,
                    )
                )
        return tuple(outputs)


def cataloged_checkpoint_launch_spec(
    *,
    catalog: CatalogIndex | str | Path,
    artifact_id: str,
    checkpoint_path: str | Path,
    model_spec: ModelSpec | Mapping[str, object],
    semantic_config: ModelSemanticConfig | Mapping[str, object],
    compiler_manifest: Mapping[str, object],
    device: str = "cpu",
) -> AdapterLaunchSpec:
    """Create the inert launch manifest required by isolated foundry execution."""

    artifact = resolve_model_artifact(catalog, artifact_id)
    resolved_model_spec = (
        model_spec if isinstance(model_spec, ModelSpec) else ModelSpec.model_validate(model_spec)
    )
    resolved_semantic = (
        semantic_config
        if isinstance(semantic_config, ModelSemanticConfig)
        else ModelSemanticConfig.model_validate(semantic_config)
    )
    return AdapterLaunchSpec(
        module=CatalogedCheckpointModelAdapter.__module__,
        qualname=CatalogedCheckpointModelAdapter.__qualname__,
        kwargs={
            "artifact": artifact.model_dump(mode="json"),
            "checkpoint_path": str(checkpoint_path),
            "compiler_manifest": dict(compiler_manifest),
            "device": device,
            "model_spec": resolved_model_spec.model_dump(mode="json"),
            "semantic_config": resolved_semantic.model_dump(mode="json"),
        },
        declared_spec=_adapter_spec(artifact, profile_id=resolved_semantic.profile_id),
    )


__all__ = [
    "ADAPTER_VERSION",
    "EPISODE_PAYLOAD_KEY",
    "EPISODE_PAYLOAD_SCHEMA",
    "READOUT_SELECTOR_KEY",
    "READOUT_SELECTOR_SCHEMA",
    "CatalogedCheckpointModelAdapter",
    "ExplicitEpisodeInputError",
    "cataloged_checkpoint_launch_spec",
]
