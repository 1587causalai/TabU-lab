"""Deterministic S1 table corpora for completion and supervised fitting.

The generators in this module are intentionally independent of the F0 fixture
registry and the experiment runner.  They materialize immutable
:class:`~tabu_lab.experiments.corpus.FitEpisodeCorpus` values that can be
consumed by any runner without giving model code access to a
:class:`~tabu_lab.contracts.TruthSidecar`.

Two source datasets are frozen here:

* a 256-row, four-numeric/two-categorical latent mixed table shared by TabUF,
  Unit-as-row, and Unit-as-cell completion;
* a 512-row supervised table with six predictors plus numeric and categorical
  responses, used by TabUL and TabUFL.

The compiler carrier contains all rows so one episode can combine train
context with held-out query predictors.  The typed row split remains the
authority for targets, sources, and statistics.  In particular, the one
numeric normalizer is fitted with an explicit value mask containing only
typed-train cells that are never truth targets in the generated train corpus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from tabu_lab.compiler import CompilationResult, NumericNormalizer, compile_episode
from tabu_lab.contracts import (
    EpisodeRecipe,
    FeatureKind,
    FeatureRole,
    FeatureSpec,
    ForwardRole,
    OriginState,
    RawDataset,
    SplitManifest,
    SplitView,
    canonical_hash,
    origin_code,
    origin_mask,
    origin_value_mask,
)

from .contracts import (
    EpisodeSchedule,
    FitTargetFamily,
    FitTargetOrigin,
    ScheduleSampling,
)
from .corpus import CompiledFitEpisode, EpisodeScheduleRealization, FitEpisodeCorpus
from .feasibility import (
    CategoricalNWTarget,
    NumericNWTarget,
    NWFeasibilityTarget,
    NWSupportArm,
    assess_nw_targets,
)
from .splits import RowPartition, RowSplitManifest

DATA_SEED = 104729
SPLIT_SEED = 130363
EPISODE_SAMPLER_SEED = DATA_SEED
EPISODE_ORDER_SEED = SPLIT_SEED

COMPLETION_CONTRACTS = ("tabuf", "tabu.unit_row", "tabu.unit_pair", "tabu.cell.base")
SUPERVISED_CONTRACTS = ("tabul", "tabufl", "tabu.cell.base.supervised")

COMPLETION_DATASET_ID = "s1-latent-mixed-completion-v1"
SUPERVISED_DATASET_ID = "s1-compositional-xor-supervised-v1"

_SOURCE = int(ForwardRole.RECEIVER | ForwardRole.SOURCE)
_TARGET = int(ForwardRole.RECEIVER | ForwardRole.TARGET)
_RECEIVER = int(ForwardRole.RECEIVER)
_PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class _EpisodeMasks:
    partition: str
    ordinal: int
    artificial: torch.Tensor
    query: torch.Tensor

    @property
    def targets(self) -> torch.Tensor:
        return self.artificial | self.query


def _hash_order(
    values: Sequence[str],
    *,
    namespace: str,
    seed: int = EPISODE_SAMPLER_SEED,
) -> tuple[str, ...]:
    """Order stable identifiers without depending on process RNG state."""

    if len(values) != len(set(values)):
        raise ValueError("hash-ordered identifiers must be unique")
    return tuple(
        sorted(
            values,
            key=lambda value: (
                canonical_hash(
                    {
                        "schema": "tabu.s1-synthetic-order-key.v1",
                        "namespace": namespace,
                        "seed": seed,
                        "value": value,
                    }
                ),
                value,
            ),
        )
    )


def build_s1_completion_dataset(*, dataset_id: str = COMPLETION_DATASET_ID) -> RawDataset:
    """Build the frozen 256-row mixed completion source table.

    All 64 combinations of two eight-level latents occur four times.  The
    third numeric feature is low-rank in the two latents, while the fourth is a
    nonlinear row-local relation.  Repetition makes every positive target
    support-realizable without turning row identifiers into model inputs.
    """

    row = torch.arange(256, dtype=torch.int64)
    combination = row.remainder(64)
    latent_a_code = combination.remainder(8)
    latent_b_code = torch.div(combination, 8, rounding_mode="floor")
    latent_a = (latent_a_code.to(torch.float32) - 3.5) / 4.0
    latent_b = (latent_b_code.to(torch.float32) - 3.5) / 4.0
    low_rank = 0.75 * latent_a + 0.25 * latent_b
    nonlinear = latent_a * latent_b + 0.25 * latent_a.square()
    parity = latent_a_code.remainder(2).bitwise_xor(latent_b_code.remainder(2)).to(torch.float32)
    bucket = (latent_a_code + 2 * latent_b_code).remainder(3).to(torch.float32)
    return RawDataset.from_values(
        dataset_id=dataset_id,
        values=torch.stack((latent_a, latent_b, low_rank, nonlinear, parity, bucket), dim=1),
        row_ids=tuple(f"completion-row-{index:03d}" for index in range(256)),
        feature_specs=(
            FeatureSpec(name="latent_a"),
            FeatureSpec(name="latent_b"),
            FeatureSpec(name="low_rank_relation"),
            FeatureSpec(name="nonlinear_row_relation"),
            FeatureSpec(
                name="latent_parity",
                kind=FeatureKind.CATEGORICAL,
                domain=("same", "different"),
                codebook_id="s1-latent-parity-v1",
            ),
            FeatureSpec(
                name="latent_bucket",
                kind=FeatureKind.CATEGORICAL,
                domain=("bucket-0", "bucket-1", "bucket-2"),
                codebook_id="s1-latent-bucket-v1",
            ),
        ),
        metadata={
            "data_seed": DATA_SEED,
            "generator": "deterministic_low_rank_nonlinear_mixed_v1",
            "latent_combinations": 64,
            "replicates_per_combination": 4,
            "scope": "support_realizable_multi_episode_fit_not_generalization",
            "stage": "S1",
        },
    )


def build_s1_supervised_dataset(*, single_response: str | None = None) -> RawDataset:
    """Build the frozen 512-row compositional/XOR supervised table."""

    row = torch.arange(512, dtype=torch.int64)
    pattern = row.remainder(16)
    binary_a = pattern.remainder(2)
    binary_b = torch.div(pattern, 2, rounding_mode="floor").remainder(2)
    phase = torch.div(pattern, 4, rounding_mode="floor").remainder(4)
    numeric_response = (binary_a + 2 * binary_b + phase).to(torch.float32)
    categorical_response = (
        binary_a.bitwise_xor(binary_b).bitwise_xor(phase.remainder(2)).to(torch.float32)
    )
    a = binary_a.to(torch.float32)
    b = binary_b.to(torch.float32)
    p = phase.to(torch.float32)
    all_values = torch.stack((a, a, b, b, p, p, numeric_response, categorical_response), dim=1)
    all_specs = (
        FeatureSpec(name="binary_a_witness_a"),
        FeatureSpec(name="binary_a_witness_b"),
        FeatureSpec(name="binary_b_witness_a"),
        FeatureSpec(name="binary_b_witness_b"),
        FeatureSpec(name="phase_witness_a"),
        FeatureSpec(name="phase_witness_b"),
        FeatureSpec(name="bounded_numeric_response", role=FeatureRole.RESPONSE),
        FeatureSpec(name="compositional_xor_response", kind=FeatureKind.CATEGORICAL,
                    domain=("class-0", "class-1"), codebook_id="s1-compositional-xor-v1",
                    role=FeatureRole.RESPONSE),
    )
    if single_response is not None:
        response_index = 6 if single_response == "regression" else 7
        values = torch.cat((all_values[:, :6], all_values[:, response_index:response_index + 1]), dim=1)
        specs = all_specs[:6] + (all_specs[response_index],)
        dataset_id = f"s1-tabu-cell-base-supervised-{single_response}-v1"
    else:
        values, specs, dataset_id = all_values, all_specs, SUPERVISED_DATASET_ID
    return RawDataset.from_values(
        dataset_id=dataset_id,
        values=values,
        row_ids=tuple(f"supervised-row-{index:03d}" for index in range(512)),
        feature_specs=specs,
        metadata={
            "context_rows": 128,
            "data_seed": DATA_SEED,
            "generator": "deterministic_compositional_xor_duplicate_predictors_v1",
            "patterns": 16,
            "query_pool_rows": 384,
            "replicates_per_pattern": 32,
            "scope": "support_realizable_multi_episode_fit_not_generalization",
            "stage": "S1",
        },
    )


def _completion_partitions(dataset: RawDataset) -> Mapping[str, tuple[str, ...]]:
    train = dataset.row_ids[:192]
    final_replicate = dataset.row_ids[192:]
    validation = tuple(row_id for index, row_id in enumerate(final_replicate) if index % 2 == 0)
    test = tuple(row_id for index, row_id in enumerate(final_replicate) if index % 2 == 1)
    return {"train": train, "validation": validation, "test": test}


def _supervised_partitions(dataset: RawDataset) -> Mapping[str, tuple[str, ...]]:
    return {
        "train": dataset.row_ids[:384],
        "validation": dataset.row_ids[384:448],
        "test": dataset.row_ids[448:],
    }


def _typed_row_split(
    dataset: RawDataset,
    partitions: Mapping[str, tuple[str, ...]],
    *,
    split_id: str,
) -> RowSplitManifest:
    return RowSplitManifest(
        dataset_id=dataset.dataset_id,
        dataset_hash=dataset.dataset_hash,
        split_id=split_id,
        fit_partition="train",
        strategy="deterministic_synthetic_row_split_v1",
        seed=SPLIT_SEED,
        require_complete=True,
        partitions=tuple(RowPartition(name=name, row_ids=partitions[name]) for name in _PARTITIONS),
    )


def _carrier(
    dataset: RawDataset,
    *,
    split_id: str,
) -> tuple[SplitManifest, Mapping[str, SplitView]]:
    # The SOURCE ledger, not carrier membership, decides which cells are
    # model-visible.  A single all-row carrier is needed for context+query
    # supervised episodes; fit_value_mask below remains typed-train-only.
    manifest = SplitManifest.create(
        dataset,
        {"train": dataset.row_ids},
        split_id=split_id,
        fit_partition="train",
        strategy="all_rows_episode_carrier_v1",
        seed=SPLIT_SEED,
        metadata={
            "statistical_scope": "typed_split_train_fit_value_mask",
            "stage": "S1",
        },
    )
    return manifest, {"train": SplitView(dataset=dataset, manifest=manifest, partition="train")}


def _completion_episode_masks(
    dataset: RawDataset,
    partitions: Mapping[str, tuple[str, ...]],
) -> tuple[_EpisodeMasks, ...]:
    row_index = {row_id: index for index, row_id in enumerate(dataset.row_ids)}
    episode_counts = {"train": 24, "validation": 4, "test": 4}
    masks: list[_EpisodeMasks] = []
    for partition in _PARTITIONS:
        rows = partitions[partition]
        # Two targets from every feature per episode: 8 numeric + 4 categorical.
        feature_orders = {
            feature: _hash_order(
                rows,
                namespace=f"completion:{partition}:feature-{feature}",
            )
            for feature in range(dataset.shape[1])
        }
        cursors = {feature: 0 for feature in range(dataset.shape[1])}
        used_coordinates: set[tuple[int, int]] = set()
        for ordinal in range(episode_counts[partition]):
            artificial = torch.zeros(dataset.shape, dtype=torch.bool)
            used_rows: set[int] = set()
            for feature in range(dataset.shape[1]):
                selected = 0
                ordered = feature_orders[feature]
                while selected < 2:
                    if cursors[feature] >= len(ordered):
                        raise RuntimeError("completion target schedule exhausted its row pool")
                    row = row_index[ordered[cursors[feature]]]
                    cursors[feature] += 1
                    coordinate = (row, feature)
                    if row in used_rows or coordinate in used_coordinates:
                        continue
                    artificial[coordinate] = True
                    used_rows.add(row)
                    used_coordinates.add(coordinate)
                    selected += 1
            masks.append(
                _EpisodeMasks(
                    partition=partition,
                    ordinal=ordinal,
                    artificial=artificial,
                    query=torch.zeros_like(artificial),
                )
            )
    return tuple(masks)


def _supervised_episode_masks(
    dataset: RawDataset,
    partitions: Mapping[str, tuple[str, ...]],
    *,
    include_completion: bool,
    response_features: tuple[int, ...] = (6, 7),
) -> tuple[_EpisodeMasks, ...]:
    row_index = {row_id: index for index, row_id in enumerate(dataset.row_ids)}
    query_pools = {
        "train": partitions["train"][128:],
        "validation": partitions["validation"],
        "test": partitions["test"],
    }
    masks: list[_EpisodeMasks] = []
    for partition in _PARTITIONS:
        ordered = _hash_order(query_pools[partition], namespace=f"supervised:{partition}:query")
        if len(ordered) % 16:
            raise RuntimeError("supervised query pool must split into 16-row episodes")
        for ordinal, offset in enumerate(range(0, len(ordered), 16)):
            query_rows = ordered[offset : offset + 16]
            query = torch.zeros(dataset.shape, dtype=torch.bool)
            artificial = torch.zeros_like(query)
            for local_index, row_id in enumerate(query_rows):
                row = row_index[row_id]
                query[row, list(response_features)] = True
                if include_completion:
                    # One predictor completion target per query row.  Rotating
                    # through all six predictors keeps the F ledger balanced,
                    # while its duplicated witness remains visible for L.
                    artificial[row, (ordinal * 16 + local_index) % 6] = True
            masks.append(
                _EpisodeMasks(
                    partition=partition,
                    ordinal=ordinal,
                    artificial=artificial,
                    query=query,
                )
            )
    return tuple(masks)


def _fit_value_mask(
    fit_view: SplitView,
    typed_split: RowSplitManifest,
    train_masks: Sequence[_EpisodeMasks],
) -> torch.Tensor:
    train_rows = set(typed_split.partition("train").row_ids)
    typed_train = torch.tensor(
        tuple(row_id in train_rows for row_id in fit_view.row_ids),
        dtype=torch.bool,
    ).unsqueeze(1)
    numeric = torch.tensor(
        tuple(spec.kind is FeatureKind.NUMERIC for spec in fit_view.feature_specs),
        dtype=torch.bool,
    ).unsqueeze(0)
    target_union = torch.zeros(fit_view.shape, dtype=torch.bool)
    for masks in train_masks:
        target_union |= masks.targets
    return origin_value_mask(fit_view.origin_states) & typed_train & numeric & ~target_union


def _roles_and_origins(
    source_view: SplitView,
    *,
    source_row_ids: set[str],
    masks: _EpisodeMasks,
    local_predictors: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    roles = torch.full(source_view.shape, _RECEIVER, dtype=torch.uint8)
    source_rows = torch.tensor(
        tuple(row_id in source_row_ids for row_id in source_view.row_ids),
        dtype=torch.bool,
    )
    roles[source_rows] = _SOURCE
    if local_predictors:
        query_rows = masks.query.any(dim=1)
        predictors = torch.tensor(
            tuple(spec.role is FeatureRole.PREDICTOR for spec in source_view.feature_specs),
            dtype=torch.bool,
        )
        roles[query_rows.unsqueeze(1) & predictors.unsqueeze(0)] = _SOURCE
    roles[masks.targets] = _TARGET

    origins = source_view.origin_states
    origins[masks.artificial] = origin_code(OriginState.ARTIFICIAL_MASK)
    origins[masks.query] = origin_code(OriginState.QUERY)
    return roles, origins


def _same_column_feasibility_targets(
    compilation: CompilationResult,
    *,
    categorical_max_nll: float = 0.10,
) -> tuple[NWFeasibilityTarget, ...]:
    # The helper consumes the host-side pair here only; callers still pass
    # ``CompiledFitEpisode.evidence`` alone to model forward.
    evidence = compilation.evidence
    truth = compilation.truth
    source = evidence.source_mask.cpu()
    query_rows = origin_mask(evidence.origin_states, OriginState.QUERY).any(dim=1)
    source = source & ~query_rows.unsqueeze(1)
    width = evidence.forward_values.shape[1]
    targets: list[NWFeasibilityTarget] = []
    for row, feature in torch.nonzero(truth.target_mask, as_tuple=False).tolist():
        support_rows = torch.nonzero(source[:, feature], as_tuple=False).flatten()
        arm = NWSupportArm(
            arm_id="same_column",
            support_ids=tuple(int(index) * width + feature for index in support_rows.tolist()),
            support_values=tuple(
                float(evidence.forward_values[index, feature]) for index in support_rows.tolist()
            ),
        )
        target_id = f"{evidence.row_ids[row]}:{evidence.feature_names[feature]}"
        family = (
            FitTargetFamily.LABEL
            if bool(origin_mask(evidence.origin_states, OriginState.QUERY)[row, feature])
            else FitTargetFamily.COMPLETION
        )
        value = float(truth.target_values[row, feature])
        if evidence.feature_specs[feature].kind is FeatureKind.CATEGORICAL:
            targets.append(
                CategoricalNWTarget(
                    target_id=target_id,
                    family=family,
                    truth_code=int(value),
                    arms=(arm,),
                    max_nll=categorical_max_nll,
                )
            )
        else:
            targets.append(
                NumericNWTarget(
                    target_id=target_id,
                    family=family,
                    truth_value=value,
                    arms=(arm,),
                )
            )
    return tuple(targets)


def _compile_corpus_episodes(
    *,
    family: str,
    source_view: SplitView,
    normalizer: NumericNormalizer,
    masks: Sequence[_EpisodeMasks],
    source_row_ids: set[str],
) -> tuple[CompiledFitEpisode, ...]:
    episodes: list[CompiledFitEpisode] = []
    supervised = family in SUPERVISED_CONTRACTS
    for item in masks:
        roles, origins = _roles_and_origins(
            source_view,
            source_row_ids=source_row_ids,
            masks=item,
            local_predictors=supervised,
        )
        recipe = EpisodeRecipe.create(
            source_view,
            source_view,
            roles,
            origin_states=origins,
            recipe_id=f"S1-{family}-{item.partition}-{item.ordinal:02d}-v1",
            metadata={
                "data_seed": DATA_SEED,
                "episode_sampler_seed": EPISODE_SAMPLER_SEED,
                "ordinal": item.ordinal,
                "partition": item.partition,
                "stage": "S1",
            },
        )
        compilation = compile_episode(
            source_view,
            recipe,
            fit_view=source_view,
            numeric_normalizer=normalizer,
        )
        feasibility_targets = _same_column_feasibility_targets(compilation)
        assess_nw_targets(
            feasibility_targets,
            report_id=f"S1-{family}-{item.partition}-{item.ordinal:02d}-feasibility",
        ).require_ready()
        if family == "completion":
            numeric = item.targets & torch.tensor(
                tuple(spec.kind is FeatureKind.NUMERIC for spec in source_view.feature_specs),
                dtype=torch.bool,
            ).unsqueeze(0)
            family_masks = {"numeric": numeric, "categorical": item.targets & ~numeric}
        elif family in {"tabul", "tabu.cell.base.supervised"}:
            family_masks = {"L": item.query}
        else:
            family_masks = {"F": item.artificial, "L": item.query}
        episodes.append(
            CompiledFitEpisode(
                partition=item.partition,  # type: ignore[arg-type]
                ordinal=item.ordinal,
                recipe=recipe,
                compilation=compilation,
                target_family_masks=family_masks,
                feasibility_targets=feasibility_targets,
            )
        )
    return tuple(episodes)


def _partition_episodes(
    episodes: Sequence[CompiledFitEpisode],
) -> Mapping[str, tuple[CompiledFitEpisode, ...]]:
    return {
        partition: tuple(episode for episode in episodes if episode.partition == partition)
        for partition in _PARTITIONS
    }


def build_s1_completion_corpus(contract_id: str = "tabuf") -> FitEpisodeCorpus:
    """Build the legacy shared corpus or the independent TabUBase asset."""

    if contract_id not in COMPLETION_CONTRACTS:
        raise ValueError(f"unsupported S1 completion contract: {contract_id!r}")
    dataset = build_s1_completion_dataset(
        dataset_id=(
            "s1-tabu-cell-base-completion-v1"
            if contract_id == "tabu.cell.base" else COMPLETION_DATASET_ID
        )
    )
    partitions = _completion_partitions(dataset)
    typed_split = _typed_row_split(
        dataset,
        partitions,
        split_id="S1-completion-row-split-v1",
    )
    carrier_manifest, carrier_views = _carrier(
        dataset,
        split_id="S1-completion-carrier-v1",
    )
    source_view = carrier_views["train"]
    masks = _completion_episode_masks(dataset, partitions)
    train_masks = tuple(item for item in masks if item.partition == "train")
    fit_value_mask = _fit_value_mask(source_view, typed_split, train_masks)
    normalizer = NumericNormalizer.fit(
        source_view,
        excluded_mask=~fit_value_mask,
    )
    episodes = _compile_corpus_episodes(
        family="completion",
        source_view=source_view,
        normalizer=normalizer,
        masks=masks,
        source_row_ids=set(partitions["train"]),
    )
    grouped = _partition_episodes(episodes)
    schedule = EpisodeSchedule(
        schedule_id=(
            "S1-010-tabu-cell-base-completion-v1"
            if contract_id == "tabu.cell.base" else "S1-completion-multi-episode-v1"
        ),
        sampling=ScheduleSampling.DETERMINISTIC_SHUFFLE,
        episode_count=len(episodes),
        targets_per_episode=12,
        target_families=(FitTargetFamily.COMPLETION,),
        target_origins=(FitTargetOrigin.ARTIFICIAL_MASK,),
        sampler_seed=EPISODE_SAMPLER_SEED,
        order_seed=EPISODE_ORDER_SEED,
    )
    realization = EpisodeScheduleRealization.create(
        schedule,
        typed_split_hash=typed_split.content_hash,
        fit_value_mask_hash=normalizer.statistics.fit_value_mask_hash,
        train_recipe_hashes=tuple(episode.recipe_hash for episode in grouped["train"]),
        validation_recipe_hashes=tuple(episode.recipe_hash for episode in grouped["validation"]),
        test_recipe_hashes=tuple(episode.recipe_hash for episode in grouped["test"]),
    )
    # Legacy carriers deliberately share builder options; TabUBase is an
    # independent asset and carries its explicit profile binding.
    return FitEpisodeCorpus(
        dataset=dataset,
        typed_split=typed_split,
        carrier_manifest=carrier_manifest,
        carrier_views=carrier_views,
        fit_value_mask=fit_value_mask,
        numeric_normalizer=normalizer,
        train_episodes=grouped["train"],
        validation_episodes=grouped["validation"],
        test_episodes=grouped["test"],
        schedule=schedule,
        schedule_realization=realization,
        builder_options=(
            {"profile": "completion.artificial_mask.v1"}
            if contract_id == "tabu.cell.base" else {}
        ),
    )


def build_s1_supervised_corpus(
    contract_id: str, *, single_response: str | None = None
) -> FitEpisodeCorpus:
    """Build the S1 TabUL or joint TabUFL multi-episode corpus."""

    if contract_id not in {"tabul", "tabufl", "tabu.cell.base.supervised"}:
        raise ValueError(f"unsupported S1 supervised contract: {contract_id!r}")
    dataset = build_s1_supervised_dataset(single_response=single_response)
    partitions = _supervised_partitions(dataset)
    typed_split = _typed_row_split(
        dataset,
        partitions,
        split_id="S1-supervised-row-split-v1",
    )
    carrier_manifest, carrier_views = _carrier(
        dataset,
        split_id="S1-supervised-carrier-v1",
    )
    source_view = carrier_views["train"]
    masks = _supervised_episode_masks(
        dataset,
        partitions,
        include_completion=contract_id == "tabufl",
        response_features=(6,) if single_response is not None else (6, 7),
    )
    train_masks = tuple(item for item in masks if item.partition == "train")
    fit_value_mask = _fit_value_mask(source_view, typed_split, train_masks)
    normalizer = NumericNormalizer.fit(
        source_view,
        excluded_mask=~fit_value_mask,
        shared_numeric_groups=(
            ("binary_a_witness_a", "binary_a_witness_b"),
            ("binary_b_witness_a", "binary_b_witness_b"),
            ("phase_witness_a", "phase_witness_b"),
        ),
    )
    episodes = _compile_corpus_episodes(
        family=("tabu.cell.base.supervised" if single_response is not None else contract_id),
        source_view=source_view,
        normalizer=normalizer,
        masks=masks,
        source_row_ids=set(partitions["train"][:128]),
    )
    grouped = _partition_episodes(episodes)
    tabufl = contract_id == "tabufl"
    schedule = EpisodeSchedule(
        schedule_id=f"S1-{contract_id}-multi-episode-v1",
        sampling=ScheduleSampling.DETERMINISTIC_SHUFFLE,
        episode_count=len(episodes),
        targets_per_episode=(16 if single_response is not None else (48 if tabufl else 32)),
        target_families=(
            (FitTargetFamily.COMPLETION, FitTargetFamily.LABEL)
            if tabufl
            else (FitTargetFamily.LABEL,)
        ),
        target_origins=(
            (FitTargetOrigin.ARTIFICIAL_MASK, FitTargetOrigin.QUERY)
            if tabufl
            else (FitTargetOrigin.QUERY,)
        ),
        sampler_seed=EPISODE_SAMPLER_SEED,
        order_seed=EPISODE_ORDER_SEED,
    )
    realization = EpisodeScheduleRealization.create(
        schedule,
        typed_split_hash=typed_split.content_hash,
        fit_value_mask_hash=normalizer.statistics.fit_value_mask_hash,
        train_recipe_hashes=tuple(episode.recipe_hash for episode in grouped["train"]),
        validation_recipe_hashes=tuple(episode.recipe_hash for episode in grouped["validation"]),
        test_recipe_hashes=tuple(episode.recipe_hash for episode in grouped["test"]),
    )
    return FitEpisodeCorpus(
        dataset=dataset,
        typed_split=typed_split,
        carrier_manifest=carrier_manifest,
        carrier_views=carrier_views,
        fit_value_mask=fit_value_mask,
        numeric_normalizer=normalizer,
        train_episodes=grouped["train"],
        validation_episodes=grouped["validation"],
        test_episodes=grouped["test"],
        schedule=schedule,
        schedule_realization=realization,
        builder_options=(
            {"profile": "supervised.label_broadcast.v1"}
            if single_response is not None else {
                "label_columns": (6, 7),
                "label_address_plan": "predictor_unit_linked_per_label_v2",
            }
        ),
    )


def build_s1_base_supervised_corpus(kind: str) -> FitEpisodeCorpus:
    if kind not in {"regression", "classification"}:
        raise ValueError("Base supervised S1 kind must be regression or classification")
    return build_s1_supervised_corpus(
        "tabu.cell.base.supervised",
        single_response=kind,
    )


def build_s1_table_corpus(contract_id: str) -> FitEpisodeCorpus:
    """Dispatch the five table contracts to their frozen S1 corpus family."""

    if contract_id in COMPLETION_CONTRACTS:
        return build_s1_completion_corpus(contract_id)
    if contract_id in {"tabul", "tabufl"}:
        return build_s1_supervised_corpus(contract_id)
    if contract_id == "tabu.cell.base":
        return build_s1_completion_corpus(contract_id)
    raise ValueError(f"contract has no S1 table corpus: {contract_id!r}")


__all__ = [
    "COMPLETION_CONTRACTS",
    "COMPLETION_DATASET_ID",
    "DATA_SEED",
    "EPISODE_ORDER_SEED",
    "EPISODE_SAMPLER_SEED",
    "SPLIT_SEED",
    "SUPERVISED_CONTRACTS",
    "SUPERVISED_DATASET_ID",
    "build_s1_completion_corpus",
    "build_s1_completion_dataset",
    "build_s1_supervised_corpus",
    "build_s1_supervised_dataset",
    "build_s1_table_corpus",
]
