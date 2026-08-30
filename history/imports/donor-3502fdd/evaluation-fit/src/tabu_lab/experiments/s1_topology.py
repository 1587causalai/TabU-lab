"""Deterministic S1 topology-family synthetic corpora.

The generators in this module are deliberately isolated from the F0 fixture
registry and from the experiment runner.  They materialize the two topology
contracts as typed :class:`FitEpisodeCorpus` objects:

* TabU4Graph: six train, one validation, and one test 128-node SBM graph;
  categorical community and numeric graph-diffusion are separate recipes
  because the graph contract declares exactly one ``target_feature``.
* TabU4Rec: one 64-by-48, exactly 40%-observed latent-factor matrix;
  numeric rating and categorical preference are separate response families.

All statistics use typed train scope only.  Episode compilation continues to
put target truth solely in ``TruthSidecar``; model-facing target values are
physical zeros.  These corpora are executable inputs, not experiment receipts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum

import torch

from tabu_lab.compiler import NumericNormalizer, compile_episode
from tabu_lab.contracts import (
    EpisodeRecipe,
    FeatureKind,
    FeatureRole,
    FeatureSpec,
    ForwardRole,
    GraphDirection,
    GraphTopology,
    OriginState,
    RawDataset,
    SplitManifest,
    SplitView,
    canonical_hash,
    origin_code,
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
    NWSupportArm,
    assess_nw_targets,
)
from .splits import (
    GraphElementId,
    GraphPartition,
    GraphSplitManifest,
    GraphSplitScope,
    InteractionId,
    InteractionPartition,
    InteractionSplitManifest,
)

DATA_SEED = 104729
SPLIT_SEED = 130363
ORDER_SEED = 130363

GRAPH_COUNT = 8
GRAPH_TRAIN_COUNT = 6
GRAPH_NODES = 128
GRAPH_COMMUNITIES = 4
GRAPH_TARGETS_PER_EPISODE = 16

REC_USERS = 64
REC_ITEMS = 48
REC_OBSERVED_INTERACTIONS = 1229
REC_TARGETS_PER_EPISODE = 16
REC_TRAIN_EPISODES = 8
REC_VALIDATION_EPISODES = 1
REC_TEST_EPISODES = 1

_SOURCE = int(ForwardRole.RECEIVER | ForwardRole.SOURCE)
_TARGET = int(ForwardRole.RECEIVER | ForwardRole.TARGET)
_RECEIVER = int(ForwardRole.RECEIVER)


class GraphSyntheticRecipe(StrEnum):
    COMMUNITY = "categorical_community"
    DIFFUSION = "numeric_diffusion"


class RecSyntheticRecipe(StrEnum):
    RATING = "numeric_rating"
    PREFERENCE = "categorical_preference"


def _stable_coordinate_order(
    coordinates: Iterable[tuple[int, int]],
    *,
    seed: int,
    namespace: str,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            coordinates,
            key=lambda coordinate: (
                canonical_hash(
                    {
                        "schema": "tabu.s1-coordinate-order.v1",
                        "seed": seed,
                        "namespace": namespace,
                        "coordinate": coordinate,
                    }
                ),
                coordinate,
            ),
        )
    )


def _balanced_order(
    coordinates: Sequence[tuple[int, int]],
    values: torch.Tensor,
    *,
    seed: int,
    namespace: str,
) -> tuple[tuple[int, int], ...]:
    buckets: dict[float, list[tuple[int, int]]] = {}
    for coordinate in coordinates:
        value = float(values[coordinate])
        buckets.setdefault(value, []).append(coordinate)
    ordered_buckets = {
        value: list(
            _stable_coordinate_order(
                bucket,
                seed=seed,
                namespace=f"{namespace}-value-{value}",
            )
        )
        for value, bucket in buckets.items()
    }
    keys = sorted(
        ordered_buckets,
        key=lambda value: canonical_hash(
            {
                "schema": "tabu.s1-value-order.v1",
                "seed": seed,
                "namespace": namespace,
                "value": value,
            }
        ),
    )
    ordered: list[tuple[int, int]] = []
    while any(ordered_buckets.values()):
        for value in keys:
            bucket = ordered_buckets[value]
            if bucket:
                ordered.append(bucket.pop(0))
    return tuple(ordered)


def _partition_for_graph(graph_index: int) -> str:
    if graph_index < GRAPH_TRAIN_COUNT:
        return "train"
    return "validation" if graph_index == GRAPH_TRAIN_COUNT else "test"


def _sbm_graph(graph_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(DATA_SEED + 7919 * graph_index)
    community = torch.arange(GRAPH_NODES, dtype=torch.int64) // (
        GRAPH_NODES // GRAPH_COMMUNITIES
    )
    same_block = community[:, None] == community[None, :]
    probability = torch.where(
        same_block,
        torch.full((GRAPH_NODES, GRAPH_NODES), 0.24),
        torch.full((GRAPH_NODES, GRAPH_NODES), 0.025),
    )
    draws = torch.rand((GRAPH_NODES, GRAPH_NODES), generator=generator)
    upper = torch.triu(draws < probability, diagonal=1)
    adjacency = upper | upper.transpose(0, 1)

    # A deterministic in-community ring prevents empty graph-local slots while
    # retaining stochastic SBM edges as the dominant topology.
    block_size = GRAPH_NODES // GRAPH_COMMUNITIES
    for block in range(GRAPH_COMMUNITIES):
        start = block * block_size
        for offset in range(block_size):
            left = start + offset
            right = start + (offset + 1) % block_size
            adjacency[left, right] = True
            adjacency[right, left] = True

    local = torch.arange(GRAPH_NODES, dtype=torch.float32)
    initial = (
        torch.sin((local + 3.0 * graph_index) * 0.17)
        + 0.35 * (community.to(torch.float32) - 1.5)
    )
    closed = adjacency | torch.eye(GRAPH_NODES, dtype=torch.bool)
    transition = closed.to(torch.float32) / closed.sum(dim=1, keepdim=True)
    diffusion = initial
    for _ in range(3):
        diffusion = transition @ diffusion
    lower = diffusion.min()
    upper_value = diffusion.max()
    diffusion = 2.0 * (diffusion - lower) / (upper_value - lower).clamp_min(1.0e-8) - 1.0
    diffusion = torch.round(diffusion * 16.0) / 16.0
    return adjacency, community, diffusion


def _graph_dataset(recipe: GraphSyntheticRecipe) -> RawDataset:
    all_adjacency = torch.zeros(
        GRAPH_COUNT * GRAPH_NODES,
        GRAPH_COUNT * GRAPH_NODES,
        dtype=torch.bool,
    )
    rows: list[str] = []
    graph_ids_by_row: dict[str, str] = {}
    signals: list[torch.Tensor] = []
    degrees: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for graph_index in range(GRAPH_COUNT):
        adjacency, community, diffusion = _sbm_graph(graph_index)
        offset = graph_index * GRAPH_NODES
        all_adjacency[
            offset : offset + GRAPH_NODES,
            offset : offset + GRAPH_NODES,
        ] = adjacency
        graph_id = f"sbm-{graph_index:02d}"
        graph_rows = tuple(f"{graph_id}-node-{node:03d}" for node in range(GRAPH_NODES))
        rows.extend(graph_rows)
        graph_ids_by_row.update({row_id: graph_id for row_id in graph_rows})
        local = torch.arange(GRAPH_NODES, dtype=torch.float32)
        signals.append(torch.cos((local + graph_index) * 0.11))
        degrees.append(adjacency.sum(dim=1).to(torch.float32) / GRAPH_NODES)
        targets.append(
            community.to(torch.float32)
            if recipe is GraphSyntheticRecipe.COMMUNITY
            else diffusion
        )

    target_spec = (
        FeatureSpec(
            name="community",
            kind=FeatureKind.CATEGORICAL,
            domain=tuple(f"community-{index}" for index in range(GRAPH_COMMUNITIES)),
            codebook_id="s1-sbm-community-v1",
            role=FeatureRole.RESPONSE,
        )
        if recipe is GraphSyntheticRecipe.COMMUNITY
        else FeatureSpec(
            name="diffusion",
            kind=FeatureKind.NUMERIC,
            role=FeatureRole.RESPONSE,
        )
    )
    values = torch.stack(
        (
            torch.cat(signals),
            torch.cat(degrees),
            torch.cat(targets),
        ),
        dim=1,
    )
    topology = GraphTopology(
        node_ids=tuple(rows),
        adjacency=all_adjacency,
        direction=GraphDirection.UNDIRECTED,
    )
    return RawDataset.from_values(
        dataset_id=f"s1-graph-sbm-{recipe.value}-v1",
        values=values,
        row_ids=tuple(rows),
        feature_specs=(
            FeatureSpec(name="node_signal"),
            FeatureSpec(name="closed_degree"),
            target_spec,
        ),
        graph_topology=topology,
        metadata={
            "data_seed": DATA_SEED,
            "generator": "deterministic_sbm_4block_diffusion_v1",
            "graph_count": GRAPH_COUNT,
            "nodes_per_graph": GRAPH_NODES,
            "graph_ids_by_row": graph_ids_by_row,
            "recipe": recipe.value,
            "scope": "S1_support_realizable_fit_not_generalization",
            "stage": "S1",
        },
    )


def _graph_target_masks(
    dataset: RawDataset,
    recipe: GraphSyntheticRecipe,
) -> dict[str, list[torch.Tensor]]:
    target_feature = 2
    by_partition: dict[str, list[torch.Tensor]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for graph_index in range(GRAPH_COUNT):
        offset = graph_index * GRAPH_NODES
        coordinates = tuple((offset + node, target_feature) for node in range(GRAPH_NODES))
        if recipe is GraphSyntheticRecipe.DIFFUSION:
            # Per-graph extrema are excluded so every target remains strictly
            # inside the train-support hull after artificial masking.
            coordinates = tuple(
                coordinate
                for coordinate in coordinates
                if abs(float(dataset.values[coordinate])) < 0.999
            )
        ordered = _balanced_order(
            coordinates,
            dataset.values,
            seed=DATA_SEED + graph_index,
            namespace=f"graph-{recipe.value}-{graph_index}",
        )
        selected = ordered[:GRAPH_TARGETS_PER_EPISODE]
        distinct = {float(dataset.values[coordinate]) for coordinate in selected}
        required_distinct = GRAPH_COMMUNITIES if recipe is GraphSyntheticRecipe.COMMUNITY else 4
        if len(selected) != GRAPH_TARGETS_PER_EPISODE or len(distinct) < required_distinct:
            raise AssertionError("graph target selection did not preserve target diversity")
        mask = torch.zeros(dataset.shape, dtype=torch.bool)
        for coordinate in selected:
            mask[coordinate] = True
        by_partition[_partition_for_graph(graph_index)].append(mask)
    return by_partition


def _graph_splits(
    dataset: RawDataset,
) -> tuple[GraphSplitManifest, SplitManifest, dict[str, SplitView]]:
    graph_ids = tuple(f"sbm-{index:02d}" for index in range(GRAPH_COUNT))
    typed_split = GraphSplitManifest(
        dataset_id=dataset.dataset_id,
        dataset_hash=dataset.dataset_hash,
        split_id="s1-graph-sbm-6-1-1-v1",
        fit_partition="train",
        strategy="fixed_graph_6_1_1",
        seed=SPLIT_SEED,
        scope=GraphSplitScope.GRAPH,
        partitions=(
            GraphPartition(
                name="train",
                elements=tuple(GraphElementId(graph_id=value) for value in graph_ids[:6]),
            ),
            GraphPartition(
                name="validation",
                elements=(GraphElementId(graph_id=graph_ids[6]),),
            ),
            GraphPartition(
                name="test",
                elements=(GraphElementId(graph_id=graph_ids[7]),),
            ),
        ),
    )
    # A single carrier view preserves each 128-node graph inside one block-
    # diagonal topology.  The typed graph split, not this carrier name, controls
    # statistics, SOURCE permission, and target membership.
    carrier_manifest = SplitManifest.create(
        dataset,
        {"train": dataset.row_ids},
        split_id="s1-graph-block-diagonal-carrier-v1",
        fit_partition="train",
        strategy="block_diagonal_all_graphs_typed_scope",
        seed=SPLIT_SEED,
    )
    views = {
        "train": SplitView(dataset=dataset, manifest=carrier_manifest, partition="train")
    }
    return typed_split, carrier_manifest, views


def _graph_train_row_mask(dataset: RawDataset) -> torch.Tensor:
    graph_ids = dataset.metadata["graph_ids_by_row"]
    if not isinstance(graph_ids, dict):
        graph_ids = dict(graph_ids)
    train_ids = {f"sbm-{index:02d}" for index in range(GRAPH_TRAIN_COUNT)}
    return torch.tensor(
        tuple(str(graph_ids[row_id]) in train_ids for row_id in dataset.row_ids),
        dtype=torch.bool,
    )


def _fit_normalizer(
    fit_view: SplitView,
    fit_value_mask: torch.Tensor,
    *,
    shared_numeric_groups: tuple[tuple[str, ...], ...] = (),
) -> NumericNormalizer:
    return NumericNormalizer.fit(
        fit_view,
        excluded_mask=~fit_value_mask,
        shared_numeric_groups=shared_numeric_groups,
    )


def _same_column_feasibility(
    compilation: object,
    *,
    family: FitTargetFamily = FitTargetFamily.COMPLETION,
) -> tuple[NumericNWTarget | CategoricalNWTarget, ...]:
    evidence = compilation.evidence
    truth = compilation.truth
    targets: list[NumericNWTarget | CategoricalNWTarget] = []
    width = evidence.forward_values.shape[1]
    for row, feature in torch.nonzero(truth.target_mask, as_tuple=False).tolist():
        source_rows = torch.nonzero(evidence.source_mask[:, feature], as_tuple=False).flatten()
        arm = NWSupportArm(
            arm_id="same_column",
            support_ids=tuple(int(source_row) * width + feature for source_row in source_rows),
            support_values=tuple(
                float(evidence.forward_values[source_row, feature]) for source_row in source_rows
            ),
        )
        target_id = f"{evidence.row_ids[row]}:{evidence.feature_names[feature]}"
        target_value = float(truth.target_values[row, feature])
        if evidence.feature_specs[feature].kind is FeatureKind.CATEGORICAL:
            targets.append(
                CategoricalNWTarget(
                    target_id=target_id,
                    family=family,
                    truth_code=int(target_value),
                    arms=(arm,),
                    max_nll=0.10,
                )
            )
        else:
            targets.append(
                NumericNWTarget(
                    target_id=target_id,
                    family=family,
                    truth_value=target_value,
                    arms=(arm,),
                )
            )
    return tuple(targets)


def _compile_graph_episodes(
    *,
    dataset: RawDataset,
    views: dict[str, SplitView],
    normalizer: NumericNormalizer,
    target_masks: dict[str, list[torch.Tensor]],
    recipe: GraphSyntheticRecipe,
) -> dict[str, tuple[CompiledFitEpisode, ...]]:
    source_view = views["train"]
    fit_view = views["train"]
    train_rows = _graph_train_row_mask(dataset)
    compiled: dict[str, tuple[CompiledFitEpisode, ...]] = {}
    for partition, masks in target_masks.items():
        episodes: list[CompiledFitEpisode] = []
        for ordinal, target_mask in enumerate(masks):
            roles = torch.full(source_view.shape, _RECEIVER, dtype=torch.uint8)
            roles[train_rows] = _SOURCE
            if partition != "train":
                target_rows = target_mask.any(dim=1)
                roles[target_rows, :2] = _SOURCE
            roles[target_mask] = _TARGET
            origins = source_view.origin_states
            origins[target_mask] = origin_code(OriginState.ARTIFICIAL_MASK)
            episode_recipe = EpisodeRecipe.create(
                source_view,
                fit_view,
                roles,
                origin_states=origins,
                recipe_id=f"s1-graph-{recipe.value}-{partition}-{ordinal:02d}-v1",
                metadata={
                    "partition": partition,
                    "recipe": recipe.value,
                    "target_feature": 2,
                },
            )
            result = compile_episode(
                source_view,
                episode_recipe,
                fit_view=fit_view,
                numeric_normalizer=normalizer,
            )
            feasibility = _same_column_feasibility(result)
            report = assess_nw_targets(
                feasibility,
                report_id=f"{episode_recipe.recipe_id}-feasibility",
            )
            report.require_ready()
            episodes.append(
                CompiledFitEpisode(
                    partition=partition,  # type: ignore[arg-type]
                    ordinal=ordinal,
                    recipe=episode_recipe,
                    compilation=result,
                    target_family_masks={recipe.value: target_mask},
                    feasibility_targets=feasibility,
                )
            )
        compiled[partition] = tuple(episodes)
    return compiled


def build_s1_graph_corpus(
    recipe: GraphSyntheticRecipe | str,
) -> FitEpisodeCorpus:
    """Build one of the two contract-distinct S1 TabU4Graph corpora."""

    resolved = GraphSyntheticRecipe(recipe)
    dataset = _graph_dataset(resolved)
    targets = _graph_target_masks(dataset, resolved)
    typed_split, carrier_manifest, views = _graph_splits(dataset)
    fit_view = views["train"]
    train_rows = _graph_train_row_mask(dataset)
    train_target_union = torch.zeros(dataset.shape, dtype=torch.bool)
    for target_mask in targets["train"]:
        train_target_union |= target_mask
    numeric = torch.tensor(
        tuple(spec.kind is FeatureKind.NUMERIC for spec in dataset.feature_specs),
        dtype=torch.bool,
    ).view(1, -1)
    fit_value_mask = (
        origin_value_mask(fit_view.origin_states)
        & train_rows.view(-1, 1)
        & numeric
        & ~train_target_union
    )
    normalizer = _fit_normalizer(fit_view, fit_value_mask)
    episodes = _compile_graph_episodes(
        dataset=dataset,
        views=views,
        normalizer=normalizer,
        target_masks=targets,
        recipe=resolved,
    )
    schedule = EpisodeSchedule(
        schedule_id=f"s1-graph-{resolved.value}-schedule-v1",
        sampling=ScheduleSampling.DETERMINISTIC_SHUFFLE,
        episode_count=GRAPH_COUNT,
        targets_per_episode=GRAPH_TARGETS_PER_EPISODE,
        target_families=(FitTargetFamily.COMPLETION,),
        target_origins=(FitTargetOrigin.ARTIFICIAL_MASK,),
        sampler_seed=DATA_SEED,
        order_seed=ORDER_SEED,
    )
    realization = EpisodeScheduleRealization.create(
        schedule,
        typed_split_hash=typed_split.content_hash,
        fit_value_mask_hash=normalizer.statistics.fit_value_mask_hash,
        train_recipe_hashes=tuple(episode.recipe_hash for episode in episodes["train"]),
        validation_recipe_hashes=tuple(
            episode.recipe_hash for episode in episodes["validation"]
        ),
        test_recipe_hashes=tuple(episode.recipe_hash for episode in episodes["test"]),
    )
    return FitEpisodeCorpus(
        dataset=dataset,
        typed_split=typed_split,
        carrier_manifest=carrier_manifest,
        carrier_views=views,
        fit_value_mask=fit_value_mask,
        numeric_normalizer=normalizer,
        train_episodes=episodes["train"],
        validation_episodes=episodes["validation"],
        test_episodes=episodes["test"],
        schedule=schedule,
        schedule_realization=realization,
        builder_options={
            "target_feature": 2,
            "unit_receiver_plan": "same_row_visible_cells",
            "recipe": resolved.value,
        },
    )


def _rec_latent_values(recipe: RecSyntheticRecipe) -> torch.Tensor:
    users = torch.arange(REC_USERS, dtype=torch.float32)
    items = torch.arange(REC_ITEMS, dtype=torch.float32)
    user_factors = torch.stack(
        (
            torch.sin((users + 1.0) * 0.37),
            torch.cos((users + 3.0) * 0.19),
            (users.remainder(7) - 3.0) / 3.0,
        ),
        dim=1,
    )
    item_factors = torch.stack(
        (
            torch.cos((items + 2.0) * 0.29),
            torch.sin((items + 5.0) * 0.13),
            (items.remainder(5) - 2.0) / 2.0,
        ),
        dim=1,
    )
    scores = (
        3.0
        + 1.30 * (user_factors @ item_factors.transpose(0, 1))
        + 0.30 * torch.sin(users[:, None] * 0.23)
        - 0.25 * torch.cos(items[None, :] * 0.31)
    )
    if recipe is RecSyntheticRecipe.RATING:
        return scores.round().clamp(1.0, 5.0)
    return (scores >= scores.median()).to(torch.float32)


def _rec_observed_mask() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(DATA_SEED)
    permutation = torch.randperm(REC_USERS * REC_ITEMS, generator=generator)
    mask = torch.zeros(REC_USERS * REC_ITEMS, dtype=torch.bool)
    mask[permutation[:REC_OBSERVED_INTERACTIONS]] = True
    return mask.view(REC_USERS, REC_ITEMS)


def _rec_interaction_partitions(
    observed: torch.Tensor,
) -> dict[str, tuple[tuple[int, int], ...]]:
    coordinates = [tuple(value) for value in torch.nonzero(observed, as_tuple=False).tolist()]
    ordered = _stable_coordinate_order(
        coordinates,
        seed=SPLIT_SEED,
        namespace="rec-observed-interaction-split",
    )
    train_count = int(0.8 * len(ordered))
    validation_count = (len(ordered) - train_count) // 2
    return {
        "train": ordered[:train_count],
        "validation": ordered[train_count : train_count + validation_count],
        "test": ordered[train_count + validation_count :],
    }


def _rec_has_equal_witnesses(
    coordinate: tuple[int, int],
    values: torch.Tensor,
    visible_train: set[tuple[int, int]],
) -> bool:
    user, item = coordinate
    target = float(values[user, item])
    user_arm = {
        float(values[other_user, item])
        for other_user in range(REC_USERS)
        if other_user != user and (other_user, item) in visible_train
    }
    item_arm = {
        float(values[user, other_item])
        for other_item in range(REC_ITEMS)
        if other_item != item and (user, other_item) in visible_train
    }
    return target in user_arm and target in item_arm


def _rec_target_batches(
    *,
    values: torch.Tensor,
    partitions: dict[str, tuple[tuple[int, int], ...]],
    recipe: RecSyntheticRecipe,
) -> dict[str, list[torch.Tensor]]:
    train = set(partitions["train"])
    episode_counts = {
        "train": REC_TRAIN_EPISODES,
        "validation": REC_VALIDATION_EPISODES,
        "test": REC_TEST_EPISODES,
    }
    result: dict[str, list[torch.Tensor]] = {name: [] for name in episode_counts}
    for partition, episode_count in episode_counts.items():
        unused = set(partitions[partition])
        for ordinal in range(episode_count):
            candidates = tuple(
                coordinate
                for coordinate in unused
                if _rec_has_equal_witnesses(coordinate, values, train - {coordinate})
            )
            ordered = _balanced_order(
                candidates,
                values,
                seed=DATA_SEED + 101 * ordinal,
                namespace=f"rec-{recipe.value}-{partition}-{ordinal}",
            )
            selected: list[tuple[int, int]] = []
            for coordinate in ordered:
                tentative = (*selected, coordinate)
                visible_train = train - set(tentative) if partition == "train" else train
                if all(
                    _rec_has_equal_witnesses(target, values, visible_train)
                    for target in tentative
                ):
                    selected.append(coordinate)
                if len(selected) == REC_TARGETS_PER_EPISODE:
                    break
            distinct = {float(values[coordinate]) for coordinate in selected}
            required_distinct = 2 if recipe is RecSyntheticRecipe.PREFERENCE else 4
            if len(selected) != REC_TARGETS_PER_EPISODE or len(distinct) < required_distinct:
                raise AssertionError(
                    f"insufficient dual-arm {recipe.value} targets in {partition} episode"
                )
            mask = torch.zeros((REC_USERS, REC_ITEMS), dtype=torch.bool)
            for coordinate in selected:
                mask[coordinate] = True
            result[partition].append(mask)
            unused.difference_update(selected)
    return result


def _rec_dataset(
    recipe: RecSyntheticRecipe,
) -> tuple[RawDataset, dict[str, tuple[tuple[int, int], ...]]]:
    values = _rec_latent_values(recipe)
    observed = _rec_observed_mask()
    partitions = _rec_interaction_partitions(observed)
    feature_specs = tuple(
        FeatureSpec(
            name=f"item-{item:02d}",
            kind=(
                FeatureKind.NUMERIC
                if recipe is RecSyntheticRecipe.RATING
                else FeatureKind.CATEGORICAL
            ),
            domain=() if recipe is RecSyntheticRecipe.RATING else ("dislike", "like"),
            codebook_id=None if recipe is RecSyntheticRecipe.RATING else "s1-rec-preference-v1",
            role=FeatureRole.RESPONSE,
        )
        for item in range(REC_ITEMS)
    )
    dataset = RawDataset.from_values(
        dataset_id=f"s1-rec-latent-{recipe.value}-v1",
        values=values,
        observed_mask=observed,
        row_ids=tuple(f"user-{user:02d}" for user in range(REC_USERS)),
        feature_specs=feature_specs,
        metadata={
            "data_seed": DATA_SEED,
            "generator": "deterministic_quantized_latent_factor_v1",
            "observed_interactions": REC_OBSERVED_INTERACTIONS,
            "observed_fraction": REC_OBSERVED_INTERACTIONS / (REC_USERS * REC_ITEMS),
            "recipe": recipe.value,
            "response_family": recipe.value,
            "scope": "S1_support_realizable_fit_not_generalization",
            "stage": "S1",
        },
    )
    return dataset, partitions


def _rec_splits(
    dataset: RawDataset,
    partitions: dict[str, tuple[tuple[int, int], ...]],
) -> tuple[InteractionSplitManifest, SplitManifest, dict[str, SplitView]]:
    typed_split = InteractionSplitManifest(
        dataset_id=dataset.dataset_id,
        dataset_hash=dataset.dataset_hash,
        split_id="s1-rec-observed-80-10-10-v1",
        fit_partition="train",
        strategy="stable_hash_observed_interactions",
        seed=SPLIT_SEED,
        partitions=tuple(
            InteractionPartition(
                name=partition,
                interactions=tuple(
                    InteractionId(
                        user_id=dataset.row_ids[user],
                        item_id=dataset.feature_names[item],
                    )
                    for user, item in coordinates
                ),
            )
            for partition, coordinates in partitions.items()
        ),
    )
    carrier_manifest = SplitManifest.create(
        dataset,
        {"train": dataset.row_ids},
        split_id="s1-rec-user-item-carrier-v1",
        fit_partition="train",
        strategy="all_users_typed_interaction_scope",
        seed=SPLIT_SEED,
    )
    views = {
        "train": SplitView(dataset=dataset, manifest=carrier_manifest, partition="train")
    }
    return typed_split, carrier_manifest, views


def _dual_arm_feasibility(
    compilation: object,
) -> tuple[NumericNWTarget | CategoricalNWTarget, ...]:
    evidence = compilation.evidence
    truth = compilation.truth
    width = evidence.forward_values.shape[1]
    targets: list[NumericNWTarget | CategoricalNWTarget] = []
    response = tuple(
        index
        for index, spec in enumerate(evidence.feature_specs)
        if spec.role is FeatureRole.RESPONSE
    )
    for row, feature in torch.nonzero(truth.target_mask, as_tuple=False).tolist():
        user_rows = torch.nonzero(evidence.source_mask[:, feature], as_tuple=False).flatten()
        item_features = tuple(
            item
            for item in response
            if item != feature and bool(evidence.source_mask[row, item])
        )
        arms = (
            NWSupportArm(
                arm_id="user",
                support_ids=tuple(int(source_row) * width + feature for source_row in user_rows),
                support_values=tuple(
                    float(evidence.forward_values[source_row, feature])
                    for source_row in user_rows
                ),
                arm_weight=0.5,
            ),
            NWSupportArm(
                arm_id="item",
                support_ids=tuple(row * width + item for item in item_features),
                support_values=tuple(
                    float(evidence.forward_values[row, item]) for item in item_features
                ),
                arm_weight=0.5,
            ),
        )
        target_id = f"{evidence.row_ids[row]}:{evidence.feature_names[feature]}"
        target_value = float(truth.target_values[row, feature])
        if evidence.feature_specs[feature].kind is FeatureKind.CATEGORICAL:
            targets.append(
                CategoricalNWTarget(
                    target_id=target_id,
                    family=FitTargetFamily.COMPLETION,
                    truth_code=int(target_value),
                    arms=arms,
                    max_nll=0.10,
                )
            )
        else:
            targets.append(
                NumericNWTarget(
                    target_id=target_id,
                    family=FitTargetFamily.COMPLETION,
                    truth_value=target_value,
                    arms=arms,
                )
            )
    return tuple(targets)


def _compile_rec_episodes(
    *,
    dataset: RawDataset,
    views: dict[str, SplitView],
    normalizer: NumericNormalizer,
    interaction_partitions: dict[str, tuple[tuple[int, int], ...]],
    target_masks: dict[str, list[torch.Tensor]],
    recipe: RecSyntheticRecipe,
) -> dict[str, tuple[CompiledFitEpisode, ...]]:
    source_view = views["train"]
    fit_view = views["train"]
    train_interactions = set(interaction_partitions["train"])
    compiled: dict[str, tuple[CompiledFitEpisode, ...]] = {}
    for partition, masks in target_masks.items():
        episodes: list[CompiledFitEpisode] = []
        for ordinal, target_mask in enumerate(masks):
            roles = torch.full(source_view.shape, _RECEIVER, dtype=torch.uint8)
            for coordinate in train_interactions:
                roles[coordinate] = _SOURCE
            roles[target_mask] = _TARGET
            origins = source_view.origin_states
            origins[target_mask] = origin_code(OriginState.ARTIFICIAL_MASK)
            episode_recipe = EpisodeRecipe.create(
                source_view,
                fit_view,
                roles,
                origin_states=origins,
                recipe_id=f"s1-rec-{recipe.value}-{partition}-{ordinal:02d}-v1",
                metadata={
                    "partition": partition,
                    "recipe": recipe.value,
                    "response_family": recipe.value,
                },
            )
            result = compile_episode(
                source_view,
                episode_recipe,
                fit_view=fit_view,
                numeric_normalizer=normalizer,
            )
            feasibility = _dual_arm_feasibility(result)
            report = assess_nw_targets(
                feasibility,
                report_id=f"{episode_recipe.recipe_id}-feasibility",
            )
            report.require_ready()
            episodes.append(
                CompiledFitEpisode(
                    partition=partition,  # type: ignore[arg-type]
                    ordinal=ordinal,
                    recipe=episode_recipe,
                    compilation=result,
                    target_family_masks={recipe.value: target_mask},
                    feasibility_targets=feasibility,
                )
            )
        compiled[partition] = tuple(episodes)
    return compiled


def build_s1_rec_corpus(
    recipe: RecSyntheticRecipe | str,
) -> FitEpisodeCorpus:
    """Build one of the two contract-distinct S1 TabU4Rec corpora."""

    resolved = RecSyntheticRecipe(recipe)
    dataset, interaction_partitions = _rec_dataset(resolved)
    target_masks = _rec_target_batches(
        values=_rec_latent_values(resolved),
        partitions=interaction_partitions,
        recipe=resolved,
    )
    typed_split, carrier_manifest, views = _rec_splits(dataset, interaction_partitions)
    fit_view = views["train"]
    train_scope = torch.zeros(dataset.shape, dtype=torch.bool)
    for coordinate in interaction_partitions["train"]:
        train_scope[coordinate] = True
    train_target_union = torch.zeros(dataset.shape, dtype=torch.bool)
    for target_mask in target_masks["train"]:
        train_target_union |= target_mask
    numeric = torch.tensor(
        tuple(spec.kind is FeatureKind.NUMERIC for spec in dataset.feature_specs),
        dtype=torch.bool,
    ).view(1, -1)
    fit_value_mask = (
        origin_value_mask(fit_view.origin_states)
        & train_scope
        & numeric
        & ~train_target_union
    )
    shared_groups = (
        (dataset.feature_names,)
        if resolved is RecSyntheticRecipe.RATING
        else ()
    )
    normalizer = _fit_normalizer(
        fit_view,
        fit_value_mask,
        shared_numeric_groups=shared_groups,
    )
    episodes = _compile_rec_episodes(
        dataset=dataset,
        views=views,
        normalizer=normalizer,
        interaction_partitions=interaction_partitions,
        target_masks=target_masks,
        recipe=resolved,
    )
    schedule = EpisodeSchedule(
        schedule_id=f"s1-rec-{resolved.value}-schedule-v1",
        sampling=ScheduleSampling.DETERMINISTIC_SHUFFLE,
        episode_count=(REC_TRAIN_EPISODES + REC_VALIDATION_EPISODES + REC_TEST_EPISODES),
        targets_per_episode=REC_TARGETS_PER_EPISODE,
        target_families=(FitTargetFamily.COMPLETION,),
        target_origins=(FitTargetOrigin.ARTIFICIAL_MASK,),
        sampler_seed=DATA_SEED,
        order_seed=ORDER_SEED,
    )
    realization = EpisodeScheduleRealization.create(
        schedule,
        typed_split_hash=typed_split.content_hash,
        fit_value_mask_hash=normalizer.statistics.fit_value_mask_hash,
        train_recipe_hashes=tuple(episode.recipe_hash for episode in episodes["train"]),
        validation_recipe_hashes=tuple(
            episode.recipe_hash for episode in episodes["validation"]
        ),
        test_recipe_hashes=tuple(episode.recipe_hash for episode in episodes["test"]),
    )
    return FitEpisodeCorpus(
        dataset=dataset,
        typed_split=typed_split,
        carrier_manifest=carrier_manifest,
        carrier_views=views,
        fit_value_mask=fit_value_mask,
        numeric_normalizer=normalizer,
        train_episodes=episodes["train"],
        validation_episodes=episodes["validation"],
        test_episodes=episodes["test"],
        schedule=schedule,
        schedule_realization=realization,
        builder_options={
            "recommendation_address_plan": "axis_address_bootstrap_v1",
            "rec_axis_summary_dim": 2,
            "rec_matched_residual_scale": 0.1,
            "response_family": resolved.value,
            "recipe": resolved.value,
        },
    )


__all__ = [
    "DATA_SEED",
    "GRAPH_COUNT",
    "GRAPH_NODES",
    "GRAPH_TARGETS_PER_EPISODE",
    "GRAPH_TRAIN_COUNT",
    "ORDER_SEED",
    "REC_ITEMS",
    "REC_OBSERVED_INTERACTIONS",
    "REC_TARGETS_PER_EPISODE",
    "REC_TEST_EPISODES",
    "REC_TRAIN_EPISODES",
    "REC_USERS",
    "REC_VALIDATION_EPISODES",
    "SPLIT_SEED",
    "GraphSyntheticRecipe",
    "RecSyntheticRecipe",
    "build_s1_graph_corpus",
    "build_s1_rec_corpus",
]
