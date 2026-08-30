"""Public, truth-opaque compiler bindings for multi-episode fit corpora.

The compiler manifest is an evidence-boundary artifact, not model input.  It
therefore records the hashes needed to audit a realized corpus while keeping
``TruthSidecar`` tensors and target values out of the public payload.  Opaque
sidecar and feasibility hashes remain part of each compiled-episode preimage,
so changing loss-side truth still changes the corpus identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tabu_lab.contracts import canonical_hash, require_sha256, to_canonical_data

from .contracts import EpisodeSchedule, ScheduleSampling
from .corpus import EpisodeScheduleRealization, FitEpisodeCorpus

CORPUS_COMPILER_BINDING_SCHEMA = "tabu.fit-corpus-compiler-binding.v2"
CORPUS_EPISODE_BINDING_SCHEMA = "tabu.fit-corpus-episode-binding.v1"
_PARTITIONS = ("train", "validation", "test")
_PROJECTIONS = {
    "tabu4rec": "observed_interactions_to_full_matrix_row_carrier",
    "tabu4graph": "nodes_to_graph_row_carrier",
}
_MANIFEST_FIELDS = {
    "binding_kind",
    "builder_options",
    "carrier_definition_hash",
    "carrier_manifest_hash",
    "carrier_view_hashes",
    "contract_id",
    "corpus_hash",
    "dataset_hash",
    "episodes",
    "fit_partition",
    "fit_value_mask_hash",
    "numeric_normalizer",
    "projection",
    "schedule",
    "schedule_hash",
    "schedule_realization",
    "schedule_realization_hash",
    "schema",
    "typed_split_hash",
    "typed_split_kind",
}
_EPISODE_FIELDS = {
    "compiled_episode_hash",
    "compilation_provenance",
    "compilation_provenance_hash",
    "episode_binding_hash",
    "episode_id",
    "evidence_hash",
    "feasibility_target_hashes",
    "ordinal",
    "partition",
    "recipe_hash",
    "sidecar_hash",
    "source_ledger_hash",
    "target_family_mask_hash",
}
_PROVENANCE_FIELDS = {
    "dataset_hash",
    "fit_view_hash",
    "graph_topology_hash",
    "numeric_normalizer_hash",
    "recipe_hash",
    "source_view_hash",
    "split_manifest_hash",
}
_NORMALIZER_FIELDS = {
    "artifact_hash",
    "config_hash",
    "counts",
    "epsilon",
    "feature_kinds",
    "feature_names",
    "fit_value_mask_hash",
    "fit_view_hash",
    "means",
    "scales",
    "schema",
    "shared_numeric_groups",
    "split_definition_hash",
}


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    canonical = to_canonical_data(value)
    if not isinstance(canonical, dict):
        raise ValueError(f"{name} must be a mapping")
    return canonical


def _sequence(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{name} must be a sequence")
    return list(value)


def _sha256(value: Any, *, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 string")
    try:
        normalized = require_sha256(value, field_name=name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical SHA-256") from exc
    if value != normalized:
        raise ValueError(f"{name} must be a lowercase canonical SHA-256")
    return normalized


def _projection(contract_id: str) -> str:
    if not isinstance(contract_id, str) or not contract_id.strip():
        raise ValueError("contract_id must be non-empty")
    return _PROJECTIONS.get(contract_id, "rows_to_tabular_row_carrier")


def _normalizer_manifest(corpus: FitEpisodeCorpus) -> dict[str, Any]:
    normalizer = corpus.numeric_normalizer
    statistics = normalizer.statistics
    statistics_preimage = {
        "schema": "tabu.fitted-statistics.v2",
        "fit_view_hash": statistics.fit_view_hash,
        "split_definition_hash": statistics.split_definition_hash,
        "config_hash": statistics.config_hash,
        "fit_value_mask_hash": statistics.fit_value_mask_hash,
        "feature_names": statistics.feature_names,
        "feature_kinds": statistics.feature_kinds,
        "counts": statistics.counts,
        "means": statistics.means,
        "scales": statistics.scales,
    }
    artifact_hash = canonical_hash(statistics_preimage)
    if artifact_hash != statistics.artifact_hash:
        raise ValueError("numeric normalizer artifact hash does not match statistics")
    return _mapping(
        {
            "schema": "tabu.numeric-normalizer-binding.v1",
            **{name: value for name, value in statistics_preimage.items() if name != "schema"},
            "epsilon": float(normalizer.epsilon),
            "shared_numeric_groups": normalizer.shared_numeric_groups,
            "artifact_hash": artifact_hash,
        },
        name="numeric normalizer manifest",
    )


def _provenance_manifest(episode: Any) -> dict[str, Any]:
    provenance = episode.compilation.provenance
    payload = {
        "dataset_hash": provenance.dataset_hash,
        "split_manifest_hash": provenance.split_manifest_hash,
        "source_view_hash": provenance.source_view_hash,
        "fit_view_hash": provenance.fit_view_hash,
        "recipe_hash": provenance.recipe_hash,
        "graph_topology_hash": provenance.graph_topology_hash,
        "numeric_normalizer_hash": provenance.numeric_normalizer_hash,
    }
    if canonical_hash({"schema": "tabu.compilation-provenance.v2", **payload}) != (
        provenance.provenance_hash
    ):
        raise ValueError("episode compiler provenance hash does not match its preimage")
    return payload


def _episode_binding(episode: Any, *, source_ledger_hash: str) -> dict[str, Any]:
    provenance = _provenance_manifest(episode)
    payload = {
        "partition": episode.partition,
        "ordinal": episode.ordinal,
        "episode_id": episode.evidence.episode_id,
        "recipe_hash": episode.recipe_hash,
        "compilation_provenance": provenance,
        "compilation_provenance_hash": episode.compilation.provenance.provenance_hash,
        "evidence_hash": episode.evidence.evidence_hash,
        # These hashes bind loss-side state without serializing target values,
        # masks, support values, or any TruthSidecar tensor into this manifest.
        "sidecar_hash": episode.truth.truth_hash,
        "target_family_mask_hash": episode.target_family_mask_hash,
        "feasibility_target_hashes": tuple(
            target.content_hash for target in episode.feasibility_targets
        ),
        "compiled_episode_hash": episode.compiled_episode_hash,
        "source_ledger_hash": source_ledger_hash,
    }
    payload["episode_binding_hash"] = canonical_hash(
        {"schema": CORPUS_EPISODE_BINDING_SCHEMA, **payload}
    )
    return _mapping(payload, name="corpus episode binding")


def build_corpus_compiler_binding_manifest(
    corpus: FitEpisodeCorpus,
    *,
    contract_id: str,
) -> dict[str, Any]:
    """Build the canonical compiler-hash preimage for one realized corpus.

    The returned mapping is JSON-ready and can be passed directly to
    ``canonical_hash`` and ``write_fit_attempt_artifacts``.  Episode order is
    semantic: train, validation, test, with contiguous ordinal order inside
    each partition.
    """

    if not isinstance(corpus, FitEpisodeCorpus):
        raise TypeError("corpus must be a FitEpisodeCorpus")
    projection = _projection(contract_id)
    source_ledgers = corpus.source_ledger_hashes
    episodes = tuple(
        _episode_binding(
            episode,
            source_ledger_hash=source_ledgers[episode.recipe_hash],
        )
        for partition in _PARTITIONS
        for episode in corpus.episodes(partition)  # type: ignore[arg-type]
    )
    manifest = _mapping(
        {
            "schema": CORPUS_COMPILER_BINDING_SCHEMA,
            "binding_kind": "multi_episode_corpus",
            "contract_id": contract_id,
            "projection": projection,
            "dataset_hash": corpus.dataset.dataset_hash,
            "typed_split_hash": corpus.typed_split.content_hash,
            "typed_split_kind": corpus.typed_split.kind.value,
            "fit_partition": corpus.typed_split.fit_partition,
            "carrier_manifest_hash": corpus.carrier_manifest.manifest_hash,
            "carrier_definition_hash": corpus.carrier_manifest.definition_hash,
            "carrier_view_hashes": {
                name: view.view_hash for name, view in corpus.carrier_views.items()
            },
            "fit_value_mask_hash": corpus.fit_value_mask_hash,
            "numeric_normalizer": _normalizer_manifest(corpus),
            "schedule": corpus.schedule.model_dump(mode="python"),
            "schedule_hash": corpus.schedule.content_hash,
            "schedule_realization": corpus.schedule_realization.model_dump(mode="python"),
            "schedule_realization_hash": corpus.schedule_realization.content_hash,
            "episodes": episodes,
            "builder_options": corpus.builder_options,
            "corpus_hash": corpus.corpus_hash,
        },
        name="corpus compiler manifest",
    )
    validate_corpus_compiler_binding_manifest(
        manifest,
        expected_hash=canonical_hash(manifest),
        contract_id=contract_id,
        dataset_hash=corpus.dataset.dataset_hash,
        typed_split_hash=corpus.typed_split.content_hash,
        typed_split_kind=corpus.typed_split.kind.value,
        fit_partition=corpus.typed_split.fit_partition,
        episode_schedule=corpus.schedule,
        expected_corpus_hash=corpus.corpus_hash,
    )
    return manifest


def corpus_compiler_episode_recipe_hashes(value: Any) -> tuple[str, ...]:
    """Project the canonical train/validation/test recipe ledger from a v2 manifest."""

    manifest = _mapping(value, name="corpus compiler manifest")
    if manifest.get("schema") != CORPUS_COMPILER_BINDING_SCHEMA:
        raise ValueError("recipe projection requires a corpus compiler binding manifest")
    realization_payload = _mapping(
        manifest.get("schedule_realization"),
        name="episode schedule realization",
    )
    try:
        realization = EpisodeScheduleRealization.model_validate(realization_payload)
    except ValueError as exc:
        raise ValueError("corpus compiler manifest has an invalid schedule realization") from exc
    return tuple(
        recipe_hash
        for partition in _PARTITIONS
        for recipe_hash in realization.recipe_hashes(partition)  # type: ignore[arg-type]
    )


def _validate_normalizer(
    value: Any,
    *,
    fit_value_mask_hash: str,
    fit_view_hash: str,
    carrier_definition_hash: str,
) -> dict[str, Any]:
    normalizer = _mapping(value, name="numeric normalizer")
    if set(normalizer) != _NORMALIZER_FIELDS:
        raise ValueError("numeric normalizer manifest has an unexpected shape")
    if normalizer.get("schema") != "tabu.numeric-normalizer-binding.v1":
        raise ValueError("numeric normalizer manifest has an unsupported schema")
    for name in (
        "artifact_hash",
        "config_hash",
        "fit_value_mask_hash",
        "fit_view_hash",
        "split_definition_hash",
    ):
        _sha256(normalizer.get(name), name=f"numeric_normalizer.{name}")
    if normalizer["fit_value_mask_hash"] != fit_value_mask_hash:
        raise ValueError("numeric normalizer is not bound to the corpus fit mask")
    if normalizer["fit_view_hash"] != fit_view_hash:
        raise ValueError("numeric normalizer is not bound to the carrier fit view")
    if normalizer["split_definition_hash"] != carrier_definition_hash:
        raise ValueError("numeric normalizer is not bound to the carrier definition")

    feature_names = _sequence(normalizer.get("feature_names"), name="feature_names")
    feature_kinds = _sequence(normalizer.get("feature_kinds"), name="feature_kinds")
    if (
        not feature_names
        or len(feature_names) != len(set(feature_names))
        or any(not isinstance(name, str) or not name for name in feature_names)
        or len(feature_kinds) != len(feature_names)
        or any(kind not in {"numeric", "categorical", "ordinal"} for kind in feature_kinds)
    ):
        raise ValueError("numeric normalizer feature schema is invalid")
    epsilon = normalizer.get("epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError("numeric normalizer epsilon must be positive")
    groups = _sequence(
        normalizer.get("shared_numeric_groups"),
        name="shared_numeric_groups",
    )
    normalized_groups: list[list[str]] = []
    grouped_names: list[str] = []
    for raw_group in groups:
        group = _sequence(raw_group, name="shared_numeric_group")
        if (
            len(group) < 2
            or len(group) != len(set(group))
            or any(name not in feature_names for name in group)
            or any(feature_kinds[feature_names.index(name)] != "numeric" for name in group)
        ):
            raise ValueError("numeric normalizer shared group is invalid")
        normalized_groups.append(group)
        grouped_names.extend(group)
    if len(grouped_names) != len(set(grouped_names)):
        raise ValueError("numeric normalizer shared groups must be disjoint")
    expected_config_hash = canonical_hash(
        {
            "kind": "numeric_normalizer",
            "epsilon": float(epsilon),
            "shared_numeric_groups": normalized_groups,
        }
    )
    if normalizer["config_hash"] != expected_config_hash:
        raise ValueError("numeric normalizer config hash does not match its preimage")
    statistics_preimage = {
        "schema": "tabu.fitted-statistics.v2",
        "fit_view_hash": normalizer["fit_view_hash"],
        "split_definition_hash": normalizer["split_definition_hash"],
        "config_hash": normalizer["config_hash"],
        "fit_value_mask_hash": normalizer["fit_value_mask_hash"],
        "feature_names": feature_names,
        "feature_kinds": feature_kinds,
        "counts": normalizer["counts"],
        "means": normalizer["means"],
        "scales": normalizer["scales"],
    }
    if canonical_hash(statistics_preimage) != normalizer["artifact_hash"]:
        raise ValueError("numeric normalizer artifact hash does not match statistics")
    return normalizer


def validate_corpus_compiler_binding_manifest(
    value: Any,
    *,
    expected_hash: str,
    contract_id: str,
    dataset_hash: str,
    typed_split_hash: str,
    typed_split_kind: str,
    fit_partition: str,
    episode_schedule: EpisodeSchedule,
    expected_corpus_hash: str | None = None,
) -> None:
    """Validate the complete public preimage of a corpus compiler hash."""

    manifest = _mapping(value, name="corpus compiler manifest")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("corpus compiler manifest has an unexpected shape")
    if manifest.get("schema") != CORPUS_COMPILER_BINDING_SCHEMA:
        raise ValueError("corpus compiler manifest has an unsupported schema")
    if manifest.get("binding_kind") != "multi_episode_corpus":
        raise ValueError("corpus compiler manifest has an invalid binding kind")
    if manifest.get("contract_id") != contract_id:
        raise ValueError("corpus compiler manifest contract differs from the experiment")
    if manifest.get("projection") != _projection(contract_id):
        raise ValueError("corpus compiler manifest projection differs from the contract")

    expected_bindings = {
        "dataset_hash": dataset_hash,
        "typed_split_hash": typed_split_hash,
        "typed_split_kind": typed_split_kind,
        "fit_partition": fit_partition,
    }
    for name, expected in expected_bindings.items():
        if manifest.get(name) != expected:
            raise ValueError(f"corpus compiler manifest {name} differs from the experiment")
    for name in (
        "corpus_hash",
        "dataset_hash",
        "typed_split_hash",
        "carrier_manifest_hash",
        "carrier_definition_hash",
        "fit_value_mask_hash",
        "schedule_hash",
        "schedule_realization_hash",
    ):
        _sha256(manifest.get(name), name=name)
    if expected_corpus_hash is not None and manifest["corpus_hash"] != expected_corpus_hash:
        raise ValueError("corpus compiler manifest does not match the live corpus hash")

    carrier_views = _mapping(manifest.get("carrier_view_hashes"), name="carrier views")
    if not carrier_views or fit_partition not in carrier_views:
        raise ValueError("carrier views must contain the fit partition")
    for name, view_hash in carrier_views.items():
        if not name:
            raise ValueError("carrier view names must be non-empty")
        _sha256(view_hash, name=f"carrier_view_hashes.{name}")

    schedule_payload = _mapping(manifest.get("schedule"), name="episode schedule")
    try:
        schedule = EpisodeSchedule.model_validate(schedule_payload)
    except ValueError as exc:
        raise ValueError("corpus compiler manifest has an invalid episode schedule") from exc
    if schedule.content_hash != manifest["schedule_hash"]:
        raise ValueError("episode schedule hash does not match its preimage")
    if not isinstance(episode_schedule, EpisodeSchedule) or schedule != episode_schedule:
        raise ValueError("corpus episode schedule differs from the preregistration")
    if schedule.sampling is not ScheduleSampling.DETERMINISTIC_SHUFFLE:
        raise ValueError("corpus compiler manifests require deterministic-shuffle schedules")

    realization_payload = _mapping(
        manifest.get("schedule_realization"),
        name="episode schedule realization",
    )
    try:
        realization = EpisodeScheduleRealization.model_validate(realization_payload)
    except ValueError as exc:
        raise ValueError("corpus compiler manifest has an invalid schedule realization") from exc
    if realization.content_hash != manifest["schedule_realization_hash"]:
        raise ValueError("schedule realization hash does not match its preimage")
    if (
        realization.schedule_hash != schedule.content_hash
        or realization.typed_split_hash != typed_split_hash
        or realization.fit_value_mask_hash != manifest["fit_value_mask_hash"]
        or realization.order_seed != schedule.order_seed
    ):
        raise ValueError("schedule realization is not bound to schedule/split/fit mask")

    normalizer = _validate_normalizer(
        manifest.get("numeric_normalizer"),
        fit_value_mask_hash=manifest["fit_value_mask_hash"],
        fit_view_hash=carrier_views[fit_partition],
        carrier_definition_hash=manifest["carrier_definition_hash"],
    )

    raw_episodes = _sequence(manifest.get("episodes"), name="corpus episodes")
    if len(raw_episodes) != schedule.episode_count:
        raise ValueError("corpus episode count differs from the schedule")
    episodes: list[dict[str, Any]] = []
    for index, raw_episode in enumerate(raw_episodes):
        episode = _mapping(raw_episode, name=f"corpus episode {index}")
        if set(episode) != _EPISODE_FIELDS:
            raise ValueError("corpus episode binding has an unexpected shape")
        partition = episode.get("partition")
        ordinal = episode.get("ordinal")
        episode_id = episode.get("episode_id")
        if partition not in _PARTITIONS:
            raise ValueError("corpus episode has an invalid partition")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("corpus episode ordinal must be non-negative")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("corpus episode id must be non-empty")
        for name in (
            "compiled_episode_hash",
            "compilation_provenance_hash",
            "episode_binding_hash",
            "evidence_hash",
            "recipe_hash",
            "sidecar_hash",
            "source_ledger_hash",
            "target_family_mask_hash",
        ):
            _sha256(episode.get(name), name=f"episode.{name}")
        feasibility_hashes = _sequence(
            episode.get("feasibility_target_hashes"),
            name="feasibility_target_hashes",
        )
        if len(feasibility_hashes) != schedule.targets_per_episode:
            raise ValueError("episode feasibility hashes do not cover every scheduled target")
        for target_hash in feasibility_hashes:
            _sha256(target_hash, name="feasibility_target_hash")

        provenance = _mapping(
            episode.get("compilation_provenance"),
            name="compilation provenance",
        )
        if set(provenance) != _PROVENANCE_FIELDS:
            raise ValueError("compilation provenance has an unexpected shape")
        for name in _PROVENANCE_FIELDS - {"graph_topology_hash"}:
            _sha256(provenance.get(name), name=f"provenance.{name}")
        _sha256(
            provenance.get("graph_topology_hash"),
            name="provenance.graph_topology_hash",
            optional=True,
        )
        if (
            provenance["dataset_hash"] != dataset_hash
            or provenance["split_manifest_hash"] != manifest["carrier_manifest_hash"]
            or provenance["source_view_hash"] not in set(carrier_views.values())
            or provenance["fit_view_hash"] != carrier_views[fit_partition]
            or provenance["recipe_hash"] != episode["recipe_hash"]
            or provenance["numeric_normalizer_hash"] != normalizer["artifact_hash"]
        ):
            raise ValueError("episode provenance escaped the corpus compiler bindings")
        expected_provenance_hash = canonical_hash(
            {"schema": "tabu.compilation-provenance.v2", **provenance}
        )
        if episode["compilation_provenance_hash"] != expected_provenance_hash:
            raise ValueError("episode provenance hash does not match its preimage")
        expected_compiled_hash = canonical_hash(
            {
                "schema": "tabu.compiled-fit-episode.v1",
                "partition": partition,
                "ordinal": ordinal,
                "recipe_hash": episode["recipe_hash"],
                "compilation_provenance_hash": expected_provenance_hash,
                "evidence_hash": episode["evidence_hash"],
                "truth_hash": episode["sidecar_hash"],
                "target_family_mask_hash": episode["target_family_mask_hash"],
                "feasibility_target_hashes": feasibility_hashes,
            }
        )
        if episode["compiled_episode_hash"] != expected_compiled_hash:
            raise ValueError("compiled episode hash does not match its opaque preimage")
        binding_preimage = {
            name: child
            for name, child in episode.items()
            if name != "episode_binding_hash"
        }
        if episode["episode_binding_hash"] != canonical_hash(
            {"schema": CORPUS_EPISODE_BINDING_SCHEMA, **binding_preimage}
        ):
            raise ValueError("episode binding hash does not match its preimage")
        episodes.append(episode)

    if len({episode["episode_id"] for episode in episodes}) != len(episodes):
        raise ValueError("corpus episode ids must be globally unique")
    if len({episode["recipe_hash"] for episode in episodes}) != len(episodes):
        raise ValueError("corpus recipe hashes must be globally unique")
    if len({episode["compiled_episode_hash"] for episode in episodes}) != len(episodes):
        raise ValueError("compiled episode hashes must be globally unique")

    flattened_recipes: list[str] = []
    partition_episode_hashes: dict[str, tuple[str, ...]] = {}
    for partition in _PARTITIONS:
        partition_items = [episode for episode in episodes if episode["partition"] == partition]
        if [episode["ordinal"] for episode in partition_items] != list(
            range(len(partition_items))
        ):
            raise ValueError("corpus episode ordinals/order are not canonical")
        recipes = tuple(episode["recipe_hash"] for episode in partition_items)
        if recipes != realization.recipe_hashes(partition):  # type: ignore[arg-type]
            raise ValueError("corpus episode order differs from the schedule realization")
        flattened_recipes.extend(recipes)
        partition_episode_hashes[partition] = tuple(
            episode["compiled_episode_hash"] for episode in partition_items
        )
    expected_partition_order = tuple(
        episode
        for partition in _PARTITIONS
        for episode in episodes
        if episode["partition"] == partition
    )
    if tuple(episodes) != expected_partition_order:
        raise ValueError("corpus episodes must be ordered train, validation, test")
    if len(flattened_recipes) != schedule.episode_count:
        raise ValueError("schedule realization does not cover every corpus episode")

    builder_options = _mapping(manifest.get("builder_options"), name="builder options")
    source_ledger_hashes = {
        episode["recipe_hash"]: episode["source_ledger_hash"] for episode in episodes
    }
    expected_corpus_preimage = {
        "schema": "tabu.fit-episode-corpus.v1",
        "dataset_hash": dataset_hash,
        "typed_split_hash": typed_split_hash,
        "carrier_manifest_hash": manifest["carrier_manifest_hash"],
        "carrier_view_hashes": carrier_views,
        "fit_value_mask_hash": manifest["fit_value_mask_hash"],
        "numeric_normalizer_hash": normalizer["artifact_hash"],
        "partition_episode_hashes": partition_episode_hashes,
        "schedule_realization_hash": realization.content_hash,
        "schedule_hash": schedule.content_hash,
        "source_ledger_hashes": source_ledger_hashes,
        "builder_options": builder_options,
    }
    if canonical_hash(expected_corpus_preimage) != manifest["corpus_hash"]:
        raise ValueError("corpus hash does not match its public opaque preimage")
    if canonical_hash(manifest) != _sha256(expected_hash, name="expected_hash"):
        raise ValueError("corpus compiler manifest does not match RunIdentity compiler_hash")


__all__ = [
    "CORPUS_COMPILER_BINDING_SCHEMA",
    "CORPUS_EPISODE_BINDING_SCHEMA",
    "build_corpus_compiler_binding_manifest",
    "corpus_compiler_episode_recipe_hashes",
    "validate_corpus_compiler_binding_manifest",
]
