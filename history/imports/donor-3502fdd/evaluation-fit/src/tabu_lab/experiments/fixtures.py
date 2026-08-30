"""Deterministic, support-realizable F0 fixtures for the buildable contracts.

The fixtures in this module are executable inputs, not evidence receipts.  Each
one follows the production split-before-compile boundary so tests and future
experiment runners exercise the same truth-isolation contract.  Dataset truth
is present only in :class:`RawDataset` and the compiled :class:`TruthSidecar`;
model-facing :class:`EvidenceEpisode` values are physically zero at targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import torch

from tabu_lab.compiler import (
    CompilationResult,
    NumericNormalizer,
    bind_split_view,
    compile_episode,
    split_dataset,
)
from tabu_lab.contracts import (
    EpisodeRecipe,
    EvidenceEpisode,
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
    TruthSidecar,
    canonical_hash,
    forward_role_mask,
    origin_code,
    origin_mask,
    to_canonical_data,
)

from .contracts import (
    EpisodeSchedule,
    FitTargetFamily,
    FitTargetOrigin,
    ScheduleSampling,
)
from .feasibility import (
    CategoricalNWTarget,
    NumericNWTarget,
    NWFeasibilityTarget,
    NWSupportArm,
)

DATA_SEED = 104729
SPLIT_SEED = 130363
MODEL_SEEDS = (1729, 2718, 31415)

BUILDABLE_CONTRACTS = (
    "tabuf",
    "tabu.unit_row",
    "tabu.unit_pair",
    "tabul",
    "tabufl",
    "tabu4graph",
    "tabu4rec",
    "tabu.cell.base",
)

_SOURCE = ForwardRole.RECEIVER | ForwardRole.SOURCE
_TARGET = ForwardRole.RECEIVER | ForwardRole.TARGET


def _frozen_options(value: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = dict(value)
    to_canonical_data(payload)
    return MappingProxyType(payload)


def _frozen_family_masks(
    value: Mapping[str, torch.Tensor],
    *,
    shape: tuple[int, int],
    target_mask: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    occupied = torch.zeros(shape, dtype=torch.bool)
    for raw_name, raw_mask in value.items():
        name = raw_name.strip()
        if not name or name in masks:
            raise ValueError("target-family names must be non-empty and unique")
        mask = torch.as_tensor(raw_mask).detach().clone().cpu().bool()
        if tuple(mask.shape) != shape:
            raise ValueError(f"target-family mask {name!r} must match fixture shape")
        if not bool(mask.any()):
            raise ValueError(f"target-family mask {name!r} cannot be empty")
        if bool((occupied & mask).any()):
            raise ValueError("target-family masks must be pairwise disjoint")
        occupied |= mask
        masks[name] = mask
    if not masks or not torch.equal(occupied, target_mask.cpu()):
        raise ValueError("target-family masks must exactly partition compiled targets")
    return MappingProxyType(masks)


@dataclass(frozen=True, slots=True)
class F0Fixture:
    """One fixed F0 episode plus all inputs needed to rebuild it."""

    fixture_id: str
    contract_id: str
    dataset: RawDataset
    split_manifest: SplitManifest
    source_view: SplitView
    fit_view: SplitView
    numeric_normalizer: NumericNormalizer
    recipe: EpisodeRecipe
    compilation: CompilationResult
    episode_schedule: EpisodeSchedule
    target_family_masks: Mapping[str, torch.Tensor]
    builder_options: Mapping[str, Any] = field(default_factory=dict)
    data_seed: int = DATA_SEED
    split_seed: int = SPLIT_SEED

    def __post_init__(self) -> None:
        fixture_id = self.fixture_id.strip()
        if not fixture_id:
            raise ValueError("fixture_id cannot be empty")
        if self.contract_id not in BUILDABLE_CONTRACTS:
            raise ValueError(f"unsupported F0 contract: {self.contract_id!r}")
        if self.data_seed != DATA_SEED or self.split_seed != SPLIT_SEED:
            raise ValueError("F0 fixtures use the frozen data and split seeds")
        if (
            self.source_view.dataset is not self.dataset
            or self.fit_view.dataset is not self.dataset
        ):
            raise ValueError("fixture views must reference the fixture dataset")
        if self.source_view.manifest != self.split_manifest:
            raise ValueError("source view must be bound to the fixture split manifest")
        if self.fit_view.manifest != self.split_manifest:
            raise ValueError("fit view must be bound to the fixture split manifest")
        if self.compilation.provenance.recipe_hash != self.recipe.recipe_hash:
            raise ValueError("compiled provenance must match the fixture recipe")
        if (
            self.compilation.provenance.numeric_normalizer_hash
            != self.numeric_normalizer.artifact_hash
        ):
            raise ValueError("compiled provenance must match the fixture normalizer")
        if self.compilation.provenance.dataset_hash != self.dataset.dataset_hash:
            raise ValueError("compiled provenance must match the fixture dataset")
        if self.compilation.provenance.split_manifest_hash != self.split_manifest.manifest_hash:
            raise ValueError("compiled provenance must match the fixture split")
        if self.episode_schedule.recipe_hashes != (self.recipe.recipe_hash,):
            raise ValueError("F0 schedule must bind exactly the fixture recipe")
        if self.episode_schedule.targets_per_episode != int(
            self.compilation.truth.target_mask.sum()
        ):
            raise ValueError("F0 schedule target count must match compiled truth")
        masks = _frozen_family_masks(
            self.target_family_masks,
            shape=self.source_view.shape,
            target_mask=self.compilation.truth.target_mask,
        )
        object.__setattr__(self, "fixture_id", fixture_id)
        object.__setattr__(self, "target_family_masks", masks)
        object.__setattr__(self, "builder_options", _frozen_options(self.builder_options))

    @property
    def fixture_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.f0-fixture.v1",
                "fixture_id": self.fixture_id,
                "contract_id": self.contract_id,
                "data_seed": self.data_seed,
                "split_seed": self.split_seed,
                "dataset_hash": self.dataset.dataset_hash,
                "split_manifest_hash": self.split_manifest.manifest_hash,
                "source_view_hash": self.source_view.view_hash,
                "fit_view_hash": self.fit_view.view_hash,
                "recipe_hash": self.recipe.recipe_hash,
                "numeric_normalizer_hash": self.numeric_normalizer.artifact_hash,
                "schedule_hash": self.episode_schedule.schedule_hash,
                "target_family_masks": self.target_family_masks,
                "builder_options": self.builder_options,
            }
        )

    @property
    def evidence(self) -> EvidenceEpisode:
        return self.compilation.evidence

    @property
    def truth(self) -> TruthSidecar:
        return self.compilation.truth


class InfeasibleReason(StrEnum):
    """Expected terminal-feasibility failure for a negative fixture."""

    NUMERIC_OUT_OF_HULL = "numeric_out_of_hull"
    MISSING_CATEGORICAL_CLASS = "missing_categorical_class"
    NO_SUPPORT = "no_support"


@dataclass(frozen=True, slots=True)
class InfeasibleF0Fixture:
    """One deliberately terminal-infeasible, truth-isolated F0 episode."""

    scenario_id: str
    reason: InfeasibleReason
    dataset: RawDataset
    split_manifest: SplitManifest
    recipe: EpisodeRecipe
    compilation: CompilationResult
    numeric_normalizer: NumericNormalizer
    target_coordinate: tuple[int, int]

    def __post_init__(self) -> None:
        scenario_id = self.scenario_id.strip()
        if not scenario_id:
            raise ValueError("scenario_id cannot be empty")
        row, feature = self.target_coordinate
        shape = tuple(self.compilation.evidence.forward_values.shape)
        if not (0 <= row < shape[0] and 0 <= feature < shape[1]):
            raise ValueError("target_coordinate is outside the fixture")
        if not bool(self.compilation.truth.target_mask[row, feature]):
            raise ValueError("target_coordinate must identify a truth target")
        if int(self.compilation.truth.target_mask.sum()) != 1:
            raise ValueError("negative F0 fixtures must contain exactly one target")
        if self.compilation.provenance.dataset_hash != self.dataset.dataset_hash:
            raise ValueError("negative fixture compilation must match its dataset")
        if self.compilation.provenance.split_manifest_hash != self.split_manifest.manifest_hash:
            raise ValueError("negative fixture compilation must match its split")
        if self.compilation.provenance.recipe_hash != self.recipe.recipe_hash:
            raise ValueError("negative fixture compilation must match its recipe")
        if (
            self.compilation.provenance.numeric_normalizer_hash
            != self.numeric_normalizer.artifact_hash
        ):
            raise ValueError("negative fixture compilation must match its normalizer")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "reason", InfeasibleReason(self.reason))

    @property
    def fixture_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.f0-infeasible-fixture.v1",
                "scenario_id": self.scenario_id,
                "reason": self.reason,
                "dataset_hash": self.dataset.dataset_hash,
                "split_manifest_hash": self.split_manifest.manifest_hash,
                "recipe_hash": self.recipe.recipe_hash,
                "numeric_normalizer_hash": self.numeric_normalizer.artifact_hash,
                "target_coordinate": self.target_coordinate,
            }
        )

    @property
    def evidence(self) -> EvidenceEpisode:
        return self.compilation.evidence

    @property
    def truth(self) -> TruthSidecar:
        return self.compilation.truth


def _roles_and_origins(
    dataset: RawDataset,
    *,
    artificial_targets: torch.Tensor | None = None,
    query_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = dataset.shape
    artificial = torch.zeros(shape, dtype=torch.bool)
    query = torch.zeros(shape, dtype=torch.bool)
    if artificial_targets is not None:
        artificial = torch.as_tensor(artificial_targets).cpu().bool()
    if query_targets is not None:
        query = torch.as_tensor(query_targets).cpu().bool()
    if tuple(artificial.shape) != shape or tuple(query.shape) != shape:
        raise ValueError("target masks must match the dataset")
    if bool((artificial & query).any()):
        raise ValueError("artificial and query targets must be disjoint")
    target = artificial | query
    observed = origin_mask(dataset.origin_states, OriginState.OBSERVED)
    if bool((target & ~observed).any()):
        raise ValueError("F0 targets must hide observed source truth")

    roles = torch.full(shape, int(ForwardRole.RECEIVER), dtype=torch.uint8)
    roles[observed] = int(_SOURCE)
    roles[target] = int(_TARGET)
    origins = dataset.origin_states.clone()
    origins[artificial] = origin_code(OriginState.ARTIFICIAL_MASK)
    origins[query] = origin_code(OriginState.QUERY)
    return roles, origins


def _compile(
    *,
    dataset: RawDataset,
    fixture_id: str,
    roles: torch.Tensor,
    origins: torch.Tensor,
    shared_numeric_groups: tuple[tuple[str, ...], ...] = (),
) -> tuple[
    SplitManifest,
    SplitView,
    SplitView,
    NumericNormalizer,
    EpisodeRecipe,
    CompilationResult,
]:
    manifest = split_dataset(
        dataset,
        {"train": tuple(range(dataset.shape[0]))},
        split_id=f"{fixture_id}-split-v1",
        fit_partition="train",
        strategy="fixed_f0",
        seed=SPLIT_SEED,
        metadata={"data_seed": DATA_SEED, "stage": "F0"},
    )
    source_view = bind_split_view(dataset, manifest, "train")
    fit_view = bind_split_view(dataset, manifest, "train")
    recipe = EpisodeRecipe.create(
        source_view,
        fit_view,
        roles,
        origin_states=origins,
        recipe_id=f"{fixture_id}-episode-v1",
        metadata={"data_seed": DATA_SEED, "split_seed": SPLIT_SEED, "stage": "F0"},
    )
    target_mask = forward_role_mask(roles, ForwardRole.TARGET)
    numeric_normalizer = NumericNormalizer.fit(
        fit_view,
        excluded_mask=target_mask,
        shared_numeric_groups=shared_numeric_groups,
    )
    result = compile_episode(
        source_view,
        recipe,
        fit_view=fit_view,
        numeric_normalizer=numeric_normalizer,
    )
    return manifest, source_view, fit_view, numeric_normalizer, recipe, result


def _fixed_schedule(
    *,
    fixture_id: str,
    recipe: EpisodeRecipe,
    target_count: int,
    target_families: tuple[FitTargetFamily, ...],
    target_origins: tuple[FitTargetOrigin, ...],
) -> EpisodeSchedule:
    return EpisodeSchedule(
        schedule_id=f"{fixture_id}-schedule-v1",
        sampling=ScheduleSampling.FIXED,
        episode_count=1,
        targets_per_episode=target_count,
        target_families=target_families,
        target_origins=target_origins,
        sampler_seed=DATA_SEED,
        order_seed=SPLIT_SEED,
        recipe_hashes=(recipe.recipe_hash,),
    )


def _completion_dataset_v1() -> RawDataset:
    rows = torch.arange(32, dtype=torch.float32)
    values = torch.stack(
        (
            rows.remainder(4),
            ((rows * 3 + 1).remainder(7) - 3) / 3,
            ((rows.remainder(5) * (torch.div(rows, 4, rounding_mode="floor") + 1)).remainder(9))
            / 4,
            rows.remainder(3),
        ),
        dim=1,
    )
    return RawDataset.from_values(
        dataset_id="f0-completion-mixed-v1",
        values=values,
        row_ids=tuple(f"row-{index:02d}" for index in range(32)),
        feature_specs=(
            FeatureSpec(name="numeric_cycle"),
            FeatureSpec(name="numeric_affine"),
            FeatureSpec(name="numeric_interaction"),
            FeatureSpec(
                name="category",
                kind=FeatureKind.CATEGORICAL,
                domain=("alpha", "beta", "gamma"),
                codebook_id="f0-completion-category-v1",
            ),
        ),
        metadata={
            "data_seed": DATA_SEED,
            "generator": "deterministic_mixed_cycles_v1",
            "stage": "F0",
        },
    )


def _build_completion_fixture_v1(contract_id: str) -> F0Fixture:
    dataset = _completion_dataset_v1()
    numeric = torch.zeros(dataset.shape, dtype=torch.bool)
    for row in range(8):
        numeric[row, row % 3] = True
    categorical = torch.zeros(dataset.shape, dtype=torch.bool)
    categorical[8:16, 3] = True
    targets = numeric | categorical
    roles, origins = _roles_and_origins(dataset, artificial_targets=targets)
    fixture_id = "F0-001-completion-shared-v1"
    manifest, source_view, fit_view, normalizer, recipe, result = _compile(
        dataset=dataset,
        fixture_id=fixture_id,
        roles=roles,
        origins=origins,
    )
    return F0Fixture(
        fixture_id=fixture_id,
        contract_id=contract_id,
        dataset=dataset,
        split_manifest=manifest,
        source_view=source_view,
        fit_view=fit_view,
        numeric_normalizer=normalizer,
        recipe=recipe,
        compilation=result,
        episode_schedule=_fixed_schedule(
            fixture_id=fixture_id,
            recipe=recipe,
            target_count=int(targets.sum()),
            target_families=(FitTargetFamily.COMPLETION,),
            target_origins=(FitTargetOrigin.ARTIFICIAL_MASK,),
        ),
        target_family_masks={"numeric": numeric, "categorical": categorical},
    )


def _completion_dataset_v2() -> RawDataset:
    """A minimal representation-identifiable mixed completion fixture.

    The three numeric columns expose the same four-level latent value.  When
    one is masked, either remaining numeric cell is a truth-free witness for
    the desired same-column support address.  The categorical column is a
    deterministic coarsening of that same visible latent value.  This isolates
    basic fit from the deliberately harder cyclic-routing stress fixture in
    v1; neither row IDs nor target truth are model inputs.
    """

    latent = torch.tensor((-0.75, -0.25, 0.25, 0.75), dtype=torch.float32).repeat(8)
    category = torch.tensor((0.0, 1.0, 1.0, 2.0), dtype=torch.float32).repeat(8)
    values = torch.stack((latent, latent, latent, category), dim=1)
    return RawDataset.from_values(
        dataset_id="f0-completion-mixed-identifiable-v2",
        values=values,
        row_ids=tuple(f"row-{index:02d}" for index in range(32)),
        feature_specs=(
            FeatureSpec(name="numeric_latent_a"),
            FeatureSpec(name="numeric_latent_b"),
            FeatureSpec(name="numeric_latent_c"),
            FeatureSpec(
                name="category",
                kind=FeatureKind.CATEGORICAL,
                domain=("alpha", "beta", "gamma"),
                codebook_id="f0-completion-category-v2",
            ),
        ),
        metadata={
            "data_seed": DATA_SEED,
            "generator": "deterministic_identifiable_latent_v2",
            "representation_witness": "other_visible_numeric_cell_same_latent",
            "scope": "fixed_episode_fit_not_generalization",
            "stage": "F0",
        },
    )


def _build_completion_fixture_v2(contract_id: str) -> F0Fixture:
    dataset = _completion_dataset_v2()
    numeric = torch.zeros(dataset.shape, dtype=torch.bool)
    for row in range(8):
        numeric[row, row % 3] = True
    categorical = torch.zeros(dataset.shape, dtype=torch.bool)
    categorical[8:16, 3] = True
    targets = numeric | categorical
    roles, origins = _roles_and_origins(dataset, artificial_targets=targets)
    fixture_id = "F0-001-completion-shared-v2"
    manifest, source_view, fit_view, normalizer, recipe, result = _compile(
        dataset=dataset,
        fixture_id=fixture_id,
        roles=roles,
        origins=origins,
        shared_numeric_groups=(
            (
                "numeric_latent_a",
                "numeric_latent_b",
                "numeric_latent_c",
            ),
        ),
    )
    return F0Fixture(
        fixture_id=fixture_id,
        contract_id=contract_id,
        dataset=dataset,
        split_manifest=manifest,
        source_view=source_view,
        fit_view=fit_view,
        numeric_normalizer=normalizer,
        recipe=recipe,
        compilation=result,
        episode_schedule=_fixed_schedule(
            fixture_id=fixture_id,
            recipe=recipe,
            target_count=int(targets.sum()),
            target_families=(FitTargetFamily.COMPLETION,),
            target_origins=(FitTargetOrigin.ARTIFICIAL_MASK,),
        ),
        target_family_masks={"numeric": numeric, "categorical": categorical},
    )


def _base_dataset_from(
    dataset: RawDataset,
    *,
    dataset_id: str,
    feature_specs: tuple[FeatureSpec, ...],
    values: torch.Tensor | None = None,
) -> RawDataset:
    """Create an independent Base asset without inheriting a legacy dataset id."""
    return RawDataset.from_values(
        dataset_id=dataset_id,
        values=dataset.values if values is None else values,
        row_ids=dataset.row_ids,
        feature_specs=feature_specs,
        metadata={**dict(dataset.metadata), "model_family": "tabu.cell.base@0.2.0"},
    )


def _build_base_completion_fixture() -> F0Fixture:
    source = _completion_dataset_v2()
    dataset = _base_dataset_from(
        source,
        dataset_id="f0-023-tabu-cell-base-completion-v1",
        feature_specs=source.feature_specs,
    )
    numeric = torch.zeros(dataset.shape, dtype=torch.bool)
    for row in range(8):
        numeric[row, row % 3] = True
    categorical = torch.zeros(dataset.shape, dtype=torch.bool)
    categorical[8:16, 3] = True
    targets = numeric | categorical
    roles, origins = _roles_and_origins(dataset, artificial_targets=targets)
    fixture_id = "F0-023-tabu-cell-base-completion-v1"
    manifest, source_view, fit_view, normalizer, recipe, result = _compile(
        dataset=dataset, fixture_id=fixture_id, roles=roles, origins=origins,
        shared_numeric_groups=(("numeric_latent_a", "numeric_latent_b", "numeric_latent_c"),),
    )
    return F0Fixture(
        fixture_id=fixture_id, contract_id="tabu.cell.base", dataset=dataset,
        split_manifest=manifest, source_view=source_view, fit_view=fit_view,
        numeric_normalizer=normalizer, recipe=recipe, compilation=result,
        episode_schedule=_fixed_schedule(
            fixture_id=fixture_id, recipe=recipe, target_count=int(targets.sum()),
            target_families=(FitTargetFamily.COMPLETION,),
            target_origins=(FitTargetOrigin.ARTIFICIAL_MASK,),
        ),
        target_family_masks={"numeric": numeric, "categorical": categorical},
        builder_options={"profile": "completion.artificial_mask.v1"},
    )


def _build_base_supervised_fixture(*, regression: bool) -> F0Fixture:
    source = _supervised_dataset_v2()
    response_index = 7 if regression else 6
    specs = tuple(source.feature_specs[:6]) + (source.feature_specs[response_index],)
    dataset = _base_dataset_from(
        source,
        dataset_id=(
            "f0-024-tabu-cell-base-supervised-regression-v1"
            if regression else "f0-025-tabu-cell-base-supervised-classification-v1"
        ),
        feature_specs=specs,
        values=torch.cat((source.values[:, :6], source.values[:, response_index:response_index + 1]), dim=1),
    )
    targets = torch.zeros(dataset.shape, dtype=torch.bool)
    targets[48:64, 6] = True
    roles, origins = _roles_and_origins(dataset, query_targets=targets)
    fixture_id = (
        "F0-024-tabu-cell-base-supervised-regression-v1"
        if regression else "F0-025-tabu-cell-base-supervised-classification-v1"
    )
    manifest, source_view, fit_view, normalizer, recipe, result = _compile(
        dataset=dataset, fixture_id=fixture_id, roles=roles, origins=origins,
        shared_numeric_groups=(("binary_a_witness_a", "binary_a_witness_b"),
                               ("binary_b_witness_a", "binary_b_witness_b"),
                               ("group_witness_a", "group_witness_b")),
    )
    family = "numeric" if regression else "categorical"
    return F0Fixture(
        fixture_id=fixture_id, contract_id="tabu.cell.base", dataset=dataset,
        split_manifest=manifest, source_view=source_view, fit_view=fit_view,
        numeric_normalizer=normalizer, recipe=recipe, compilation=result,
        episode_schedule=_fixed_schedule(
            fixture_id=fixture_id, recipe=recipe, target_count=int(targets.sum()),
            target_families=(FitTargetFamily.LABEL,), target_origins=(FitTargetOrigin.QUERY,),
        ),
        target_family_masks={family: targets},
        builder_options={
            "profile": "supervised.label_broadcast.v1",
            "label_columns": (6,),
        },
    )


def _supervised_dataset() -> RawDataset:
    rows = torch.arange(64, dtype=torch.float32)
    binary_a = rows.remainder(2)
    binary_b = torch.div(rows, 2, rounding_mode="floor").remainder(2)
    phase = rows.remainder(4)
    group = torch.div(rows, 4, rounding_mode="floor").remainder(4)
    xor_label = (binary_a.to(torch.int64) ^ binary_b.to(torch.int64)).to(torch.float32)
    bounded_numeric = binary_a + 2 * binary_b + group
    values = torch.stack(
        (binary_a, binary_b, phase, group, xor_label, bounded_numeric),
        dim=1,
    )
    return RawDataset.from_values(
        dataset_id="f0-supervised-xor-bounded-v1",
        values=values,
        row_ids=tuple(f"row-{index:02d}" for index in range(64)),
        feature_specs=(
            FeatureSpec(name="binary_a"),
            FeatureSpec(name="binary_b"),
            FeatureSpec(name="phase"),
            FeatureSpec(name="group"),
            FeatureSpec(
                name="xor_label",
                kind=FeatureKind.CATEGORICAL,
                domain=("equal", "different"),
                codebook_id="f0-xor-label-v1",
                role=FeatureRole.RESPONSE,
            ),
            FeatureSpec(name="bounded_numeric_label", role=FeatureRole.RESPONSE),
        ),
        metadata={
            "context_rows": 48,
            "data_seed": DATA_SEED,
            "generator": "deterministic_xor_bounded_v1",
            "query_rows": 16,
            "stage": "F0",
        },
    )


def _supervised_dataset_v2() -> RawDataset:
    """Duplicated predictors isolate F/L fit from address identifiability."""

    rows = torch.arange(64, dtype=torch.float32)
    binary_a = rows.remainder(2)
    binary_b = torch.div(rows, 2, rounding_mode="floor").remainder(2)
    group = torch.div(rows, 4, rounding_mode="floor").remainder(4)
    xor_label = (binary_a.to(torch.int64) ^ binary_b.to(torch.int64)).to(torch.float32)
    bounded_numeric = binary_a + 2 * binary_b + group
    values = torch.stack(
        (
            binary_a,
            binary_a,
            binary_b,
            binary_b,
            group,
            group,
            xor_label,
            bounded_numeric,
        ),
        dim=1,
    )
    return RawDataset.from_values(
        dataset_id="f0-supervised-xor-bounded-identifiable-v2",
        values=values,
        row_ids=tuple(f"row-{index:02d}" for index in range(64)),
        feature_specs=(
            FeatureSpec(name="binary_a_witness_a"),
            FeatureSpec(name="binary_a_witness_b"),
            FeatureSpec(name="binary_b_witness_a"),
            FeatureSpec(name="binary_b_witness_b"),
            FeatureSpec(name="group_witness_a"),
            FeatureSpec(name="group_witness_b"),
            FeatureSpec(
                name="xor_label",
                kind=FeatureKind.CATEGORICAL,
                domain=("equal", "different"),
                codebook_id="f0-xor-label-v2",
                role=FeatureRole.RESPONSE,
            ),
            FeatureSpec(name="bounded_numeric_label", role=FeatureRole.RESPONSE),
        ),
        metadata={
            "context_rows": 48,
            "data_seed": DATA_SEED,
            "generator": "deterministic_xor_bounded_duplicate_predictors_v2",
            "query_rows": 16,
            "representation_witness": "visible_duplicate_for_each_completion_predictor",
            "scope": "fixed_episode_fit_not_generalization",
            "stage": "F0",
        },
    )


def _build_supervised_fixture_v2(contract_id: str) -> F0Fixture:
    if contract_id not in {"tabul", "tabufl"}:
        raise ValueError("supervised v2 fixture requires tabul or tabufl")
    dataset = _supervised_dataset_v2()
    label_targets = torch.zeros(dataset.shape, dtype=torch.bool)
    label_targets[48:64, 6:8] = True
    feature_targets = torch.zeros(dataset.shape, dtype=torch.bool)
    if contract_id == "tabufl":
        for row in range(16):
            feature_targets[row, row % 6] = True
    roles, origins = _roles_and_origins(
        dataset,
        artificial_targets=feature_targets,
        query_targets=label_targets,
    )
    fixture_id = (
        "F0-002-tabul-query-label-v2"
        if contract_id == "tabul"
        else "F0-003-tabufl-joint-ledgers-v2"
    )
    manifest, source_view, fit_view, normalizer, recipe, result = _compile(
        dataset=dataset,
        fixture_id=fixture_id,
        roles=roles,
        origins=origins,
        shared_numeric_groups=(
            ("binary_a_witness_a", "binary_a_witness_b"),
            ("binary_b_witness_a", "binary_b_witness_b"),
            ("group_witness_a", "group_witness_b"),
        ),
    )
    target_families = (
        (FitTargetFamily.LABEL,)
        if contract_id == "tabul"
        else (FitTargetFamily.COMPLETION, FitTargetFamily.LABEL)
    )
    target_origins = (
        (FitTargetOrigin.QUERY,)
        if contract_id == "tabul"
        else (FitTargetOrigin.ARTIFICIAL_MASK, FitTargetOrigin.QUERY)
    )
    family_masks = (
        {"L": label_targets}
        if contract_id == "tabul"
        else {"F": feature_targets, "L": label_targets}
    )
    return F0Fixture(
        fixture_id=fixture_id,
        contract_id=contract_id,
        dataset=dataset,
        split_manifest=manifest,
        source_view=source_view,
        fit_view=fit_view,
        numeric_normalizer=normalizer,
        recipe=recipe,
        compilation=result,
        episode_schedule=_fixed_schedule(
            fixture_id=fixture_id,
            recipe=recipe,
            target_count=int((feature_targets | label_targets).sum()),
            target_families=target_families,
            target_origins=target_origins,
        ),
        target_family_masks=family_masks,
        builder_options={
            "label_columns": (6, 7),
            "label_address_plan": "predictor_only_per_label_v1",
        },
    )


def _tabufl_dataset_v4() -> RawDataset:
    """Joint fixture with an identifiable F latent and compositional L rules."""

    rows = torch.arange(64, dtype=torch.float32)
    binary_a = rows.remainder(2)
    binary_b = torch.div(rows, 2, rounding_mode="floor").remainder(2)
    group = torch.div(rows, 4, rounding_mode="floor").remainder(4)
    latent = (binary_a + 2.0 * binary_b - 1.5) / 2.0
    xor_label = (binary_a.to(torch.int64) ^ binary_b.to(torch.int64)).to(torch.float32)
    bounded_numeric = binary_a + 2.0 * binary_b + group
    values = torch.stack(
        (
            latent,
            latent,
            latent,
            binary_a,
            binary_b,
            group,
            xor_label,
            bounded_numeric,
        ),
        dim=1,
    )
    return RawDataset.from_values(
        dataset_id="f0-tabufl-completion-latent-label-composition-v4",
        values=values,
        row_ids=tuple(f"row-{index:02d}" for index in range(64)),
        feature_specs=(
            FeatureSpec(name="completion_latent_a"),
            FeatureSpec(name="completion_latent_b"),
            FeatureSpec(name="completion_latent_c"),
            FeatureSpec(name="label_binary_a"),
            FeatureSpec(name="label_binary_b"),
            FeatureSpec(name="label_group"),
            FeatureSpec(
                name="xor_label",
                kind=FeatureKind.CATEGORICAL,
                domain=("equal", "different"),
                codebook_id="f0-xor-label-v4",
                role=FeatureRole.RESPONSE,
            ),
            FeatureSpec(name="bounded_numeric_label", role=FeatureRole.RESPONSE),
        ),
        metadata={
            "context_rows": 48,
            "data_seed": DATA_SEED,
            "generator": "deterministic_completion_latent_label_composition_v4",
            "query_rows": 16,
            "representation_witness": "three_shared_four_level_completion_latents",
            "scope": "fixed_episode_fit_not_generalization",
            "stage": "F0",
        },
    )


def _build_tabufl_fixture_v4() -> F0Fixture:
    dataset = _tabufl_dataset_v4()
    feature_targets = torch.zeros(dataset.shape, dtype=torch.bool)
    # Exercise each of the three completion Features at every one of the four
    # attainable latent values.  The balanced 3x4 ledger avoids the target
    # frequency asymmetry that made an earlier rotating 16-mask probe unstable.
    for feature in range(3):
        for level in range(4):
            feature_targets[feature * 4 + level, feature] = True
    label_targets = torch.zeros(dataset.shape, dtype=torch.bool)
    label_targets[48:64, 6:8] = True
    roles, origins = _roles_and_origins(
        dataset,
        artificial_targets=feature_targets,
        query_targets=label_targets,
    )
    fixture_id = "F0-003-tabufl-joint-ledgers-v4"
    manifest, source_view, fit_view, normalizer, recipe, result = _compile(
        dataset=dataset,
        fixture_id=fixture_id,
        roles=roles,
        origins=origins,
        shared_numeric_groups=(
            (
                "completion_latent_a",
                "completion_latent_b",
                "completion_latent_c",
            ),
        ),
    )
    return F0Fixture(
        fixture_id=fixture_id,
        contract_id="tabufl",
        dataset=dataset,
        split_manifest=manifest,
        source_view=source_view,
        fit_view=fit_view,
        numeric_normalizer=normalizer,
        recipe=recipe,
        compilation=result,
        episode_schedule=_fixed_schedule(
            fixture_id=fixture_id,
            recipe=recipe,
            target_count=int((feature_targets | label_targets).sum()),
            target_families=(FitTargetFamily.COMPLETION, FitTargetFamily.LABEL),
            target_origins=(
                FitTargetOrigin.ARTIFICIAL_MASK,
                FitTargetOrigin.QUERY,
            ),
        ),
        target_family_masks={"F": feature_targets, "L": label_targets},
        builder_options={
            "label_columns": (6, 7),
            "label_address_plan": "predictor_unit_linked_per_label_v2",
        },
    )


def _build_tabul_fixture() -> F0Fixture:
    dataset = _supervised_dataset()
    label_targets = torch.zeros(dataset.shape, dtype=torch.bool)
    label_targets[48:64, 4:6] = True
    roles, origins = _roles_and_origins(dataset, query_targets=label_targets)
    fixture_id = "F0-002-tabul-query-label-v1"
    manifest, source_view, fit_view, normalizer, recipe, result = _compile(
        dataset=dataset,
        fixture_id=fixture_id,
        roles=roles,
        origins=origins,
    )
    return F0Fixture(
        fixture_id=fixture_id,
        contract_id="tabul",
        dataset=dataset,
        split_manifest=manifest,
        source_view=source_view,
        fit_view=fit_view,
        numeric_normalizer=normalizer,
        recipe=recipe,
        compilation=result,
        episode_schedule=_fixed_schedule(
            fixture_id=fixture_id,
            recipe=recipe,
            target_count=int(label_targets.sum()),
            target_families=(FitTargetFamily.LABEL,),
            target_origins=(FitTargetOrigin.QUERY,),
        ),
        target_family_masks={"L": label_targets},
        builder_options={"label_columns": (4, 5)},
    )


def _build_tabufl_fixture() -> F0Fixture:
    dataset = _supervised_dataset()
    feature_targets = torch.zeros(dataset.shape, dtype=torch.bool)
    for row in range(16):
        feature_targets[row, row % 4] = True
    label_targets = torch.zeros(dataset.shape, dtype=torch.bool)
    label_targets[48:64, 4:6] = True
    roles, origins = _roles_and_origins(
        dataset,
        artificial_targets=feature_targets,
        query_targets=label_targets,
    )
    fixture_id = "F0-003-tabufl-joint-ledgers-v1"
    manifest, source_view, fit_view, normalizer, recipe, result = _compile(
        dataset=dataset,
        fixture_id=fixture_id,
        roles=roles,
        origins=origins,
    )
    return F0Fixture(
        fixture_id=fixture_id,
        contract_id="tabufl",
        dataset=dataset,
        split_manifest=manifest,
        source_view=source_view,
        fit_view=fit_view,
        numeric_normalizer=normalizer,
        recipe=recipe,
        compilation=result,
        episode_schedule=_fixed_schedule(
            fixture_id=fixture_id,
            recipe=recipe,
            target_count=int((feature_targets | label_targets).sum()),
            target_families=(FitTargetFamily.COMPLETION, FitTargetFamily.LABEL),
            target_origins=(FitTargetOrigin.ARTIFICIAL_MASK, FitTargetOrigin.QUERY),
        ),
        target_family_masks={"F": feature_targets, "L": label_targets},
        builder_options={"label_columns": (4, 5)},
    )


def _grid_topology() -> GraphTopology:
    node_ids = tuple(f"node-{index:02d}" for index in range(64))
    adjacency = torch.zeros(64, 64, dtype=torch.bool)
    for row in range(8):
        for column in range(8):
            node = row * 8 + column
            for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor_row = row + delta_row
                neighbor_column = column + delta_column
                if 0 <= neighbor_row < 8 and 0 <= neighbor_column < 8:
                    neighbor = neighbor_row * 8 + neighbor_column
                    adjacency[node, neighbor] = True
    return GraphTopology(
        node_ids=node_ids,
        adjacency=adjacency,
        direction=GraphDirection.UNDIRECTED,
    )


def _build_graph_fixture() -> F0Fixture:
    topology = _grid_topology()
    rows = torch.arange(8, dtype=torch.float32).repeat_interleave(8)
    columns = torch.arange(8, dtype=torch.float32).repeat(8)
    context = (rows - columns) / 7
    tau = (2 * rows + columns).remainder(4)
    dataset = RawDataset.from_values(
        dataset_id="f0-graph-grid-v1",
        values=torch.stack((context, tau), dim=1),
        row_ids=topology.node_ids,
        feature_specs=(
            FeatureSpec(name="position_context"),
            FeatureSpec(
                name="tau",
                kind=FeatureKind.CATEGORICAL,
                domain=("community-0", "community-1", "community-2", "community-3"),
                codebook_id="f0-graph-tau-v1",
            ),
        ),
        graph_topology=topology,
        metadata={
            "data_seed": DATA_SEED,
            "generator": "deterministic_8x8_grid_v1",
            "stage": "F0",
        },
    )
    tau_targets = torch.zeros(dataset.shape, dtype=torch.bool)
    for row in range(8):
        for column in range(8):
            if (row + column) % 4 == 0:
                tau_targets[row * 8 + column, 1] = True
    roles, origins = _roles_and_origins(dataset, artificial_targets=tau_targets)
    fixture_id = "F0-004-tabu4graph-grid-v1"
    manifest, source_view, fit_view, normalizer, recipe, result = _compile(
        dataset=dataset,
        fixture_id=fixture_id,
        roles=roles,
        origins=origins,
    )
    return F0Fixture(
        fixture_id=fixture_id,
        contract_id="tabu4graph",
        dataset=dataset,
        split_manifest=manifest,
        source_view=source_view,
        fit_view=fit_view,
        numeric_normalizer=normalizer,
        recipe=recipe,
        compilation=result,
        episode_schedule=_fixed_schedule(
            fixture_id=fixture_id,
            recipe=recipe,
            target_count=int(tau_targets.sum()),
            target_families=(FitTargetFamily.COMPLETION,),
            target_origins=(FitTargetOrigin.ARTIFICIAL_MASK,),
        ),
        target_family_masks={"tau": tau_targets},
        builder_options={"target_feature": 1},
    )


def _recommendation_values() -> torch.Tensor:
    users = torch.arange(16, dtype=torch.int64)
    items = torch.arange(12, dtype=torch.int64)
    user_factors = torch.stack(
        (((users * 7 + 3).remainder(11) - 5), ((users * 5 + 1).remainder(13) - 6)),
        dim=1,
    )
    item_factors = torch.stack(
        (((items * 3 + 2).remainder(7) - 3), ((items * 7 + 4).remainder(11) - 5)),
        dim=1,
    )
    scores = user_factors @ item_factors.transpose(0, 1)
    return torch.clamp(torch.round(3 + scores.to(torch.float32) / 6), 1, 5)


def _recommendation_values_v2() -> torch.Tensor:
    """Quantized additive rank-two ratings with axis-identifiable structure."""

    users = torch.arange(16, dtype=torch.float32).remainder(4).unsqueeze(1)
    items = torch.arange(12, dtype=torch.float32).remainder(2).unsqueeze(0)
    return 1.0 + users + items


def _recommendation_observed_mask() -> torch.Tensor:
    # 193 is prime and 73 is coprime to it.  The first 192 residues form a
    # deterministic near-permutation; this threshold gives exactly 134/192
    # observed interactions (69.79%) for DATA_SEED % 100 == 29.
    flat = torch.arange(16 * 12, dtype=torch.int64)
    residues = (flat * 73 + DATA_SEED % 100).remainder(193)
    observed = (residues < 134).reshape(16, 12)
    if int(observed.sum()) != 134:
        raise AssertionError("F0 recommendation mask must contain 134 observations")
    return observed


def _recommendation_targets(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    coordinates: list[tuple[int, int]] = []
    row_counts = [0] * values.shape[0]
    column_counts = [0] * values.shape[1]

    def dual_arm_feasible(selected: list[tuple[int, int]]) -> bool:
        remaining = observed.clone()
        for selected_user, selected_item in selected:
            remaining[selected_user, selected_item] = False
        for selected_user, selected_item in selected:
            row_values = values[selected_user, remaining[selected_user]]
            column_values = values[remaining[:, selected_item], selected_item]
            if row_values.numel() == 0 or column_values.numel() == 0:
                return False
            # Each active arm is independently normalized, then the contract
            # assigns both arms weight 0.5.  The reachable interval is the
            # weighted Minkowski sum, not the hull of their union.
            lower = 0.5 * (row_values.min() + column_values.min())
            upper = 0.5 * (row_values.max() + column_values.max())
            truth = values[selected_user, selected_item]
            if not bool((lower <= truth) & (truth <= upper)):
                return False
        return True

    # Cycle ratings rather than walking coordinates alone: the 24 targets span
    # all five rating values and cannot be passed by a constant prediction.
    for target_rating in (1, 2, 3, 4, 5) * 20:
        for flat_index in range(values.numel()):
            user, item = divmod(flat_index, values.shape[1])
            coordinate = (user, item)
            if coordinate in coordinates or not bool(observed[user, item]):
                continue
            if int(values[user, item]) != target_rating:
                continue
            if row_counts[user] >= 2 or column_counts[item] >= 2:
                continue
            if not dual_arm_feasible([*coordinates, coordinate]):
                continue
            coordinates.append(coordinate)
            row_counts[user] += 1
            column_counts[item] += 1
            break
        if len(coordinates) == 24:
            break
    if len(coordinates) != 24:
        raise AssertionError("failed to construct 24 recommendation F0 targets")

    targets = torch.zeros_like(observed)
    for user, item in coordinates:
        targets[user, item] = True
    if not dual_arm_feasible(coordinates):
        raise AssertionError("recommendation target truth left the dual-arm reachable interval")
    return targets


def _recommendation_targets_v2(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Choose targets retaining an equal-value witness in each Rec arm."""

    coordinates: list[tuple[int, int]] = []
    row_counts = [0] * values.shape[0]
    column_counts = [0] * values.shape[1]

    def both_arms_have_equal_witness(selected: list[tuple[int, int]]) -> bool:
        remaining = observed.clone()
        for user, item in selected:
            remaining[user, item] = False
        for user, item in selected:
            truth = values[user, item]
            user_arm = values[remaining[:, item], item]
            item_arm = values[user, remaining[user]]
            if user_arm.numel() == 0 or item_arm.numel() == 0:
                return False
            if not bool((user_arm == truth).any()) or not bool((item_arm == truth).any()):
                return False
        return True

    for target_rating in (1, 2, 3, 4, 5) * 20:
        for flat_index in range(values.numel()):
            user, item = divmod(flat_index, values.shape[1])
            coordinate = (user, item)
            if coordinate in coordinates or not bool(observed[user, item]):
                continue
            if int(values[user, item]) != target_rating:
                continue
            if row_counts[user] >= 2 or column_counts[item] >= 2:
                continue
            if not both_arms_have_equal_witness([*coordinates, coordinate]):
                continue
            coordinates.append(coordinate)
            row_counts[user] += 1
            column_counts[item] += 1
            break
        if len(coordinates) == 24:
            break
    if len(coordinates) != 24 or not both_arms_have_equal_witness(coordinates):
        raise AssertionError("failed to retain equal-value witnesses for 24 Rec targets")
    targets = torch.zeros_like(observed)
    for user, item in coordinates:
        targets[user, item] = True
    return targets


def _build_rec_fixture() -> F0Fixture:
    values = _recommendation_values()
    observed = _recommendation_observed_mask()
    dataset = RawDataset.from_values(
        dataset_id="f0-recommendation-low-rank-v1",
        values=values,
        observed_mask=observed,
        row_ids=tuple(f"user-{index:02d}" for index in range(16)),
        feature_specs=tuple(
            FeatureSpec(name=f"item-{index:02d}", role=FeatureRole.RESPONSE) for index in range(12)
        ),
        metadata={
            "data_seed": DATA_SEED,
            "generator": "deterministic_quantized_rank2_v1",
            "observed_interactions": 134,
            "stage": "F0",
        },
    )
    rating_targets = _recommendation_targets(values, observed)
    roles, origins = _roles_and_origins(dataset, artificial_targets=rating_targets)
    fixture_id = "F0-005-tabu4rec-dual-arm-v1"
    manifest, source_view, fit_view, normalizer, recipe, result = _compile(
        dataset=dataset,
        fixture_id=fixture_id,
        roles=roles,
        origins=origins,
        shared_numeric_groups=(dataset.feature_names,),
    )
    return F0Fixture(
        fixture_id=fixture_id,
        contract_id="tabu4rec",
        dataset=dataset,
        split_manifest=manifest,
        source_view=source_view,
        fit_view=fit_view,
        numeric_normalizer=normalizer,
        recipe=recipe,
        compilation=result,
        episode_schedule=_fixed_schedule(
            fixture_id=fixture_id,
            recipe=recipe,
            target_count=int(rating_targets.sum()),
            target_families=(FitTargetFamily.COMPLETION,),
            target_origins=(FitTargetOrigin.ARTIFICIAL_MASK,),
        ),
        target_family_masks={"rating": rating_targets},
    )


def _build_rec_fixture_v2() -> F0Fixture:
    values = _recommendation_values_v2()
    observed = _recommendation_observed_mask()
    dataset = RawDataset.from_values(
        dataset_id="f0-recommendation-axis-identifiable-v2",
        values=values,
        observed_mask=observed,
        row_ids=tuple(f"user-{index:02d}" for index in range(16)),
        feature_specs=tuple(
            FeatureSpec(name=f"item-{index:02d}", role=FeatureRole.RESPONSE) for index in range(12)
        ),
        metadata={
            "data_seed": DATA_SEED,
            "generator": "deterministic_quantized_additive_rank2_v2",
            "observed_interactions": 134,
            "representation_witness": "equal_value_visible_in_each_user_item_arm",
            "scope": "fixed_episode_fit_not_generalization",
            "stage": "F0",
        },
    )
    rating_targets = _recommendation_targets_v2(values, observed)
    roles, origins = _roles_and_origins(dataset, artificial_targets=rating_targets)
    fixture_id = "F0-005-tabu4rec-dual-arm-v2"
    manifest, source_view, fit_view, normalizer, recipe, result = _compile(
        dataset=dataset,
        fixture_id=fixture_id,
        roles=roles,
        origins=origins,
        shared_numeric_groups=(dataset.feature_names,),
    )
    return F0Fixture(
        fixture_id=fixture_id,
        contract_id="tabu4rec",
        dataset=dataset,
        split_manifest=manifest,
        source_view=source_view,
        fit_view=fit_view,
        numeric_normalizer=normalizer,
        recipe=recipe,
        compilation=result,
        episode_schedule=_fixed_schedule(
            fixture_id=fixture_id,
            recipe=recipe,
            target_count=int(rating_targets.sum()),
            target_families=(FitTargetFamily.COMPLETION,),
            target_origins=(FitTargetOrigin.ARTIFICIAL_MASK,),
        ),
        target_family_masks={"rating": rating_targets},
        builder_options={
            "recommendation_address_plan": "axis_address_bootstrap_v1",
            "rec_axis_summary_dim": 2,
            "rec_matched_residual_scale": 0.1,
        },
    )


def build_f0_fixture(
    contract_id: str,
    *,
    fixture_version: str = "v1",
) -> F0Fixture:
    """Build one deterministic positive F0 fixture by contract and version.

    ``v1`` remains available for historical preregistration and failure
    replay.  Completion-family ``v2`` is the first explicitly
    support-realizable/trainability-oriented repair and never overwrites the
    original fixture identity.
    """

    if contract_id in {"tabuf", "tabu.unit_row", "tabu.unit_pair"}:
        if fixture_version == "v1":
            return _build_completion_fixture_v1(contract_id)
        if fixture_version == "v2":
            return _build_completion_fixture_v2(contract_id)
        raise ValueError(f"unsupported completion F0 fixture version: {fixture_version!r}")
    if contract_id in {"tabul", "tabufl"}:
        if fixture_version == "v1":
            return _build_tabul_fixture() if contract_id == "tabul" else _build_tabufl_fixture()
        if fixture_version == "v2":
            return _build_supervised_fixture_v2(contract_id)
        if fixture_version == "v4" and contract_id == "tabufl":
            return _build_tabufl_fixture_v4()
        raise ValueError(f"{contract_id} has no F0 fixture version {fixture_version!r}")
    if contract_id == "tabu4graph":
        if fixture_version not in {"v1", "v2"}:
            raise ValueError(f"tabu4graph has no F0 fixture version {fixture_version!r}")
        # v2 repairs the executable dynamics plan, not the frozen data or
        # target ledger.  Reusing the exact episode makes the architectural
        # comparison identifiable while the experiment id and code hash
        # remain distinct.
        return _build_graph_fixture()
    if contract_id == "tabu4rec":
        if fixture_version == "v1":
            return _build_rec_fixture()
        if fixture_version == "v2":
            return _build_rec_fixture_v2()
        raise ValueError(f"tabu4rec has no F0 fixture version {fixture_version!r}")
    if contract_id == "tabu.cell.base":
        if fixture_version in {"v1", "completion", "completion-v1"}:
            return _build_base_completion_fixture()
        if fixture_version in {"supervised_regression", "regression"}:
            return _build_base_supervised_fixture(regression=True)
        if fixture_version in {"supervised_classification", "classification"}:
            return _build_base_supervised_fixture(regression=False)
        raise ValueError(f"tabu.cell.base has no F0 fixture version {fixture_version!r}")
    if fixture_version != "v1":
        raise ValueError(f"{contract_id} has no F0 fixture version {fixture_version!r}")
    raise KeyError(f"unknown buildable F0 contract: {contract_id!r}")


def build_f0_fixture_for_dataset(contract_id: str, dataset_id: str) -> F0Fixture:
    """Resolve an immutable fixture from the preregistered dataset identity."""

    versions_by_contract = {
        "tabuf": ("v1", "v2"),
        "tabu.unit_row": ("v1", "v2"),
        "tabu.unit_pair": ("v1", "v2"),
        "tabul": ("v1", "v2"),
        "tabufl": ("v1", "v2", "v4"),
        "tabu4graph": ("v1", "v2"),
        "tabu4rec": ("v1", "v2"),
        "tabu.cell.base": ("completion", "supervised_regression", "supervised_classification"),
    }
    versions = versions_by_contract.get(contract_id, ("v1",))
    for version in versions:
        fixture = build_f0_fixture(contract_id, fixture_version=version)
        if fixture.dataset.dataset_id == dataset_id:
            return fixture
    raise KeyError(f"no {contract_id} F0 fixture is registered for dataset {dataset_id!r}")


def build_all_f0_fixtures(*, fixture_version: str = "v1") -> tuple[F0Fixture, ...]:
    """Return the seven fixtures in frozen family-progression order."""

    if fixture_version != "v1":
        raise ValueError("only the complete seven-model v1 fixture matrix is registered")
    return tuple(
        build_f0_fixture(contract_id, fixture_version=fixture_version)
        for contract_id in BUILDABLE_CONTRACTS
    )


def _single_target_negative(
    *,
    scenario_id: str,
    reason: InfeasibleReason,
    values: torch.Tensor,
    feature_spec: FeatureSpec,
) -> InfeasibleF0Fixture:
    dataset = RawDataset.from_values(
        dataset_id=f"f0-infeasible-{scenario_id}-v1",
        values=values,
        row_ids=tuple(f"row-{index}" for index in range(values.shape[0])),
        feature_specs=(feature_spec,),
        metadata={
            "data_seed": DATA_SEED,
            "expected_reason": reason.value,
            "generator": "deterministic_terminal_negative_v1",
            "stage": "F0",
        },
    )
    target = torch.zeros(dataset.shape, dtype=torch.bool)
    target[0, 0] = True
    roles, origins = _roles_and_origins(dataset, artificial_targets=target)
    manifest, _, _, normalizer, recipe, result = _compile(
        dataset=dataset,
        fixture_id=f"F0-negative-{scenario_id}-v1",
        roles=roles,
        origins=origins,
    )
    return InfeasibleF0Fixture(
        scenario_id=scenario_id,
        reason=reason,
        dataset=dataset,
        split_manifest=manifest,
        recipe=recipe,
        compilation=result,
        numeric_normalizer=normalizer,
        target_coordinate=(0, 0),
    )


def build_infeasible_f0_fixtures() -> tuple[InfeasibleF0Fixture, ...]:
    """Build out-of-hull, missing-class, and no-support negative controls."""

    return (
        _single_target_negative(
            scenario_id="numeric-out-of-hull",
            reason=InfeasibleReason.NUMERIC_OUT_OF_HULL,
            values=torch.tensor([[10.0], [0.0], [1.0]]),
            feature_spec=FeatureSpec(name="numeric"),
        ),
        _single_target_negative(
            scenario_id="missing-categorical-class",
            reason=InfeasibleReason.MISSING_CATEGORICAL_CLASS,
            values=torch.tensor([[2.0], [0.0], [1.0], [0.0]]),
            feature_spec=FeatureSpec(
                name="category",
                kind=FeatureKind.CATEGORICAL,
                domain=("zero", "one", "two"),
                codebook_id="f0-negative-category-v1",
            ),
        ),
        _single_target_negative(
            scenario_id="no-support",
            reason=InfeasibleReason.NO_SUPPORT,
            values=torch.tensor([[1.0]]),
            feature_spec=FeatureSpec(name="numeric"),
        ),
    )


def assert_truth_isolated(fixture: F0Fixture | InfeasibleF0Fixture) -> None:
    """Fail if target truth is physically present in model-facing evidence."""

    target = fixture.truth.target_mask
    if not bool(target.any()):
        raise AssertionError("fixture must contain at least one scored truth target")
    if not bool((fixture.evidence.forward_values[target] == 0).all()):
        raise AssertionError("target values must be physically zero in EvidenceEpisode")
    if bool(forward_role_mask(fixture.evidence.forward_roles, ForwardRole.SOURCE)[target].any()):
        raise AssertionError("targets must never carry the SOURCE role")
    if not bool(
        forward_role_mask(fixture.evidence.forward_roles, ForwardRole.TARGET)[target].all()
    ):
        raise AssertionError("truth targets must carry the TARGET role")


def _target_family(
    fixture: F0Fixture | InfeasibleF0Fixture,
    *,
    row: int,
    feature: int,
) -> FitTargetFamily:
    if isinstance(fixture, F0Fixture):
        label_mask = fixture.target_family_masks.get("L")
        if label_mask is not None and bool(label_mask[row, feature]):
            return FitTargetFamily.LABEL
    return FitTargetFamily.COMPLETION


def _support_arm(
    fixture: F0Fixture | InfeasibleF0Fixture,
    *,
    arm_id: str,
    mask: torch.Tensor,
) -> NWSupportArm:
    width = fixture.evidence.forward_values.shape[1]
    coordinates = torch.nonzero(mask, as_tuple=False)
    support_ids = tuple(int(row) * width + int(feature) for row, feature in coordinates.tolist())
    support_values = tuple(
        float(fixture.evidence.forward_values[row, feature])
        for row, feature in coordinates.tolist()
    )
    return NWSupportArm(
        arm_id=arm_id,
        support_ids=support_ids,
        support_values=support_values,
    )


def build_f0_feasibility_targets(
    fixture: F0Fixture | InfeasibleF0Fixture,
    *,
    categorical_max_nll: float = 0.05,
) -> tuple[NWFeasibilityTarget, ...]:
    """Materialize the exact NW support ledger for a fixture's targets."""

    evidence = fixture.evidence
    truth = fixture.truth
    source = evidence.source_mask.clone()
    contract_id = fixture.contract_id if isinstance(fixture, F0Fixture) else "tabuf"
    if contract_id in {"tabul", "tabufl"}:
        query_rows = origin_mask(evidence.origin_states, OriginState.QUERY).any(dim=1)
        source &= ~query_rows.unsqueeze(1)

    targets: list[NWFeasibilityTarget] = []
    for row, feature in torch.nonzero(truth.target_mask, as_tuple=False).tolist():
        target_id = f"{evidence.row_ids[row]}:{evidence.feature_names[feature]}"
        if contract_id == "tabu4rec":
            user_mask = torch.zeros_like(source)
            user_mask[:, feature] = source[:, feature]
            item_mask = torch.zeros_like(source)
            item_mask[row, :] = source[row, :]
            arms = (
                _support_arm(fixture, arm_id="user", mask=user_mask),
                _support_arm(fixture, arm_id="item", mask=item_mask),
            )
        else:
            same_column = torch.zeros_like(source)
            same_column[:, feature] = source[:, feature]
            arms = (_support_arm(fixture, arm_id="same_column", mask=same_column),)

        family = _target_family(fixture, row=row, feature=feature)
        spec = evidence.feature_specs[feature]
        target_value = float(truth.target_values[row, feature])
        if spec.kind is FeatureKind.CATEGORICAL:
            targets.append(
                CategoricalNWTarget(
                    target_id=target_id,
                    family=family,
                    truth_code=int(target_value),
                    arms=arms,
                    max_nll=categorical_max_nll,
                )
            )
        else:
            targets.append(
                NumericNWTarget(
                    target_id=target_id,
                    family=family,
                    truth_value=target_value,
                    arms=arms,
                )
            )
    return tuple(targets)


__all__ = [
    "BUILDABLE_CONTRACTS",
    "DATA_SEED",
    "MODEL_SEEDS",
    "SPLIT_SEED",
    "F0Fixture",
    "InfeasibleF0Fixture",
    "InfeasibleReason",
    "assert_truth_isolated",
    "build_all_f0_fixtures",
    "build_f0_feasibility_targets",
    "build_f0_fixture",
    "build_f0_fixture_for_dataset",
    "build_infeasible_f0_fixtures",
]
