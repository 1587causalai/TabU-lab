"""Synthetic-to-real transfer contracts, deterministic priors, and arm helpers.

The module is data-source agnostic: OpenML bytes are never fetched here.  It
owns the versioned S1/R1 boundary, cache/world identities, paired statistics,
and a small in-memory training loop that can be embedded by the receipt-aware
runner.  A loop result is not formal evidence until an existing ``RunIdentity``
and receipt verifier bind it to immutable artifacts.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Literal

import torch
from pydantic import Field, field_validator, model_validator

from tabu_lab.contracts.canonical import canonical_hash, require_sha256
from tabu_lab.evidence.schemas import EvidenceSchema

ROOT_SEEDS = (1729, 2718, 31415)
SCALE_WORLD_COUNTS = {"S0": 2_048, "S1": 20_000, "S2": 100_000}
CHECKPOINT_LADDER = (0, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)
R0_LEARNING_RATES = (1.0e-4, 3.0e-4, 1.0e-3)
R0_MAX_UPDATES = (400, 1_200)
R1_CLASSIFICATION_TASKS = (43, 3, 14969, 45)
R1_REGRESSION_TASKS = (361249, 361234, 361260, 361267)
R0_DEVELOPMENT_DATASETS = ("openml-task-7592:adult", "sklearn:load_diabetes")
R2_CAPACITY_BLOCKED_CC18 = (
    3481,
    3573,
    9910,
    9964,
    9976,
    9981,
    14970,
    146825,
    167121,
    167124,
    167125,
)
R2_CLASSIFICATION_TASKS = (
    6,
    11,
    12,
    14,
    15,
    16,
    18,
    22,
    23,
    28,
    29,
    31,
    32,
    37,
    49,
    53,
    219,
    2074,
    2079,
    3021,
    3022,
    3549,
    3560,
    3902,
    3903,
    3904,
    3913,
    3917,
    3918,
    9946,
    9952,
    9957,
    9960,
    9971,
    9977,
    9978,
    9985,
    10093,
    10101,
    14952,
    14954,
    14965,
    125920,
    125922,
    146195,
    146800,
    146817,
    146819,
    146820,
    146821,
    146822,
    146824,
    167119,
    167120,
    167140,
    167141,
)
R2_REGRESSION_TASKS = (
    361235,
    361236,
    361237,
    361241,
    361242,
    361243,
    361244,
    361247,
    361250,
    361251,
    361252,
    361253,
    361254,
    361255,
    361256,
    361257,
    361258,
    361259,
    361261,
    361264,
    361266,
    361268,
    361269,
    361272,
    361616,
    361617,
    361618,
    361619,
    361621,
    361622,
    361623,
)


class SyntheticFamily(StrEnum):
    SMOOTH_SPARSE_SCM = "smooth_sparse_scm"
    TREE_THRESHOLD = "tree_threshold"
    LATENT_FACTOR = "latent_factor_low_rank"
    HETEROSCEDASTIC_SHIFT = "heteroscedastic_missingness_shift"


class TransferArm(StrEnum):
    PRETRAINED = "synthetic_pretrained"
    SCRATCH = "matched_scratch"


class TransferModality(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TransferMetric(StrEnum):
    NORMALIZED_NLL = "normalized_nll"
    SCALED_RMSE = "scaled_rmse"


class EvidenceLineageRelation(StrEnum):
    INITIALIZED_FROM = "initialized_from"
    RESUMES_FROM = "resumes_from"


class SyntheticPriorSpec(EvidenceSchema):
    """Immutable generator mixture for one pretraining lineage."""

    schema_version: Literal["tabu.synthetic-prior.v1"] = "tabu.synthetic-prior.v1"
    prior_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    generator_families: tuple[SyntheticFamily, ...] = Field(min_length=1)
    heldout_generator_families: tuple[SyntheticFamily, ...] = ()
    mixture_weights: dict[SyntheticFamily, float]
    target_modalities: tuple[TransferModality, ...] = (
        TransferModality.CLASSIFICATION,
        TransferModality.REGRESSION,
    )
    target_types: tuple[Literal["numeric", "binary", "ordinal", "categorical"], ...] = (
        "numeric",
        "binary",
        "ordinal",
        "categorical",
    )
    world_count: int = Field(gt=0)
    schema_version_label: str = Field(default="synthetic-schema-v1", min_length=1)
    cache_manifest_sha256: str
    sampler_seed: int = Field(ge=0)
    split_seed: int = Field(ge=0)
    split_policy: Literal["world_hash_80_10_10_with_heldout_family"] = (
        "world_hash_80_10_10_with_heldout_family"
    )
    adaptive_curriculum: Literal[False] = False

    @field_validator("cache_manifest_sha256")
    @classmethod
    def _cache_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="cache_manifest_sha256")

    @field_validator("generator_families", "heldout_generator_families")
    @classmethod
    def _unique_families(cls, values: tuple[SyntheticFamily, ...]) -> tuple[SyntheticFamily, ...]:
        if len(values) != len(set(values)):
            raise ValueError("synthetic generator families must be unique")
        return values

    @model_validator(mode="after")
    def _mixture_is_closed(self) -> SyntheticPriorSpec:
        train = set(self.generator_families)
        heldout = set(self.heldout_generator_families)
        required_train = {
            SyntheticFamily.SMOOTH_SPARSE_SCM,
            SyntheticFamily.TREE_THRESHOLD,
            SyntheticFamily.LATENT_FACTOR,
        }
        if train != required_train:
            raise ValueError("synthetic-prior-v1 requires the three frozen training families")
        if heldout != {SyntheticFamily.HETEROSCEDASTIC_SHIFT}:
            raise ValueError("synthetic-prior-v1 requires heteroscedastic_shift as held-out family")
        if train & heldout:
            raise ValueError("train and heldout generator families must be disjoint")
        if set(self.mixture_weights) != train:
            raise ValueError("mixture_weights must cover exactly the train families")
        if any(
            not math.isfinite(weight) or weight <= 0.0 for weight in self.mixture_weights.values()
        ):
            raise ValueError("mixture weights must be finite and positive")
        if not math.isclose(sum(self.mixture_weights.values()), 1.0, abs_tol=1e-12):
            raise ValueError("mixture weights must sum to one")
        if any(
            not math.isclose(weight, 1.0 / 3.0, abs_tol=1e-12)
            for weight in self.mixture_weights.values()
        ):
            raise ValueError("synthetic-prior-v1 uses equal-weight family sampling")
        if len(self.target_types) != len(set(self.target_types)):
            raise ValueError("target_types must be unique")
        if not self.target_types:
            raise ValueError("at least one target type is required")
        if not train:
            raise ValueError("at least one train generator family is required")
        return self

    @property
    def spec_hash(self) -> str:
        return self.content_hash


class TransferTrainingConfig(EvidenceSchema):
    """Shared training settings for pretraining and both real-data arms."""

    schema_version: Literal["tabu.transfer-training.v1"] = "tabu.transfer-training.v1"
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    gradient_clip_norm: float = Field(gt=0.0)
    max_updates: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    scheduler: Literal["none"] = "none"
    exact_resume: Literal[True] = True


class ComputeEnvelope(EvidenceSchema):
    """Portable, coarse execution disclosure for a formal transfer run."""

    schema_version: Literal["tabu.compute-envelope.v1"] = "tabu.compute-envelope.v1"
    host_class: Literal["cuda", "mps", "cpu"]
    accelerator: Literal["cuda", "mps", "cpu"]
    dtype: Literal["float32"] = "float32"
    max_wall_clock_minutes: int = Field(gt=0)
    artifact_locator: str = Field(min_length=1)

    @model_validator(mode="after")
    def _accelerator_matches_host(self) -> ComputeEnvelope:
        if self.host_class != self.accelerator:
            raise ValueError("compute envelope host_class and accelerator must match")
        return self


class DatasetPassport(EvidenceSchema):
    schema_version: Literal["tabu.dataset-passport.v1"] = "tabu.dataset-passport.v1"
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    source_uri: str = Field(min_length=1)
    source_sha256: str
    dataset_sha256: str
    split_sha256: str
    license_id: str = Field(min_length=1)
    raw_files_in_git: Literal[False] = False
    access_status: Literal["verified", "blocked_license_or_access"]

    @field_validator("source_sha256", "dataset_sha256", "split_sha256")
    @classmethod
    def _passport_hash(cls, value: str, info: object) -> str:
        return require_sha256(value, field_name=getattr(info, "field_name", "sha256"))

    @field_validator("license_id")
    @classmethod
    def _license_is_explicit(cls, value: str) -> str:
        if value.strip().lower() in {"public", "unknown", "unspecified", "none"}:
            raise ValueError("license_id must be an explicit SPDX or publisher identifier")
        return value


class TransferSplitManifest(EvidenceSchema):
    """Resolved OpenML task split used before any episode compilation."""

    schema_version: Literal["tabu.transfer-split-manifest.v1"] = (
        "tabu.transfer-split-manifest.v1"
    )
    task_id: int = Field(gt=0)
    repeat: Literal[0] = 0
    test_row_ids: tuple[str, ...] = Field(min_length=1)
    validation_row_ids: tuple[str, ...] = Field(min_length=1)
    train_row_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _disjoint_partitions(self) -> TransferSplitManifest:
        partitions = {
            "train": set(self.train_row_ids),
            "validation": set(self.validation_row_ids),
            "test": set(self.test_row_ids),
        }
        if any(
            len(values) != len(getattr(self, f"{name}_row_ids"))
            for name, values in partitions.items()
        ):
            raise ValueError("split row identifiers must be unique within each partition")
        if partitions["train"] & (partitions["validation"] | partitions["test"]):
            raise ValueError("train rows must be disjoint from validation and test")
        if partitions["validation"] & partitions["test"]:
            raise ValueError("validation and test rows must be disjoint")
        return self


def resolve_task_provided_split(
    task_id: int,
    folds: Mapping[int, Mapping[str, Sequence[str]]],
    *,
    repeat: int = 0,
) -> TransferSplitManifest:
    """Resolve repeat-0 OpenML folds with fail-closed train intersection.

    ``folds[0]`` and ``folds[1]`` must each expose ``train`` and ``test`` row
    identifiers. Fold 0 is the sealed test, fold 1 is validation, and train is
    the intersection of both fold training sets. No rows are downloaded or
    compiled by this pure resolver.
    """

    if type(task_id) is not int or task_id <= 0:
        raise ValueError("task_id must be a positive integer")
    if repeat != 0:
        raise ValueError("transfer v1 only accepts OpenML repeat 0")
    if set(folds) != {0, 1}:
        raise ValueError("task-provided split must contain exactly folds 0 and 1")
    normalized: dict[int, dict[str, tuple[str, ...]]] = {}
    for fold, payload in folds.items():
        if set(payload) != {"train", "test"}:
            raise ValueError("each task fold must contain exactly train and test rows")
        values = {
            name: tuple(str(row) for row in payload[name])
            for name in ("train", "test")
        }
        if any(not row for rows in values.values() for row in rows):
            raise ValueError("task split row identifiers cannot be empty")
        if any(len(rows) != len(set(rows)) for rows in values.values()):
            raise ValueError("task split rows must be unique within a fold")
        if set(values["train"]) & set(values["test"]):
            raise ValueError("task split train and test rows overlap within a fold")
        normalized[fold] = values
    train = tuple(sorted(set(normalized[0]["train"]) & set(normalized[1]["train"])))
    validation = tuple(sorted(normalized[1]["test"]))
    test = tuple(sorted(normalized[0]["test"]))
    return TransferSplitManifest(
        task_id=task_id,
        repeat=0,
        train_row_ids=train,
        validation_row_ids=validation,
        test_row_ids=test,
    )


class PretrainExperimentSpec(EvidenceSchema):
    schema_version: Literal["tabu.pretrain-experiment.v1"] = "tabu.pretrain-experiment.v1"
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    model_id: Literal["tabul"] = "tabul"
    model_spec_hash: str
    prior: SyntheticPriorSpec
    training: TransferTrainingConfig
    root_seeds: tuple[int, ...] = ROOT_SEEDS
    scale: Literal["S0", "S1", "S2"] = "S1"
    active_seeds: tuple[int, ...] | None = None
    checkpoint_updates: tuple[int, ...] = Field(min_length=1)
    compute_envelope: ComputeEnvelope = ComputeEnvelope(
        host_class="cpu",
        accelerator="cpu",
        max_wall_clock_minutes=60,
        artifact_locator="artifact://pending",
    )
    resume_policy: Literal["exact_state_only"] = "exact_state_only"
    resume_relation: Literal["resumes_from"] = "resumes_from"
    resumes_from: str | None = None
    source_identity_sha256: str | None = None

    @field_validator("model_spec_hash", "source_identity_sha256")
    @classmethod
    def _hashes(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return require_sha256(value, field_name=getattr(info, "field_name", "sha256"))

    @field_validator("root_seeds")
    @classmethod
    def _frozen_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if values != ROOT_SEEDS:
            raise ValueError("transfer v1 freezes root seeds to 1729, 2718, and 31415")
        return values

    @field_validator("checkpoint_updates")
    @classmethod
    def _checkpoint_order(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        allowed = set(CHECKPOINT_LADDER)
        if any(value not in allowed for value in values) or tuple(sorted(set(values))) != values:
            raise ValueError(
                "checkpoint_updates must use the frozen 0/2k/5k/10k/20k/50k/100k ladder"
            )
        if values[0] != 0:
            raise ValueError("checkpoint_updates must begin with checkpoint 0")
        return values

    @model_validator(mode="after")
    def _scale_budget(self) -> PretrainExperimentSpec:
        expected_worlds = SCALE_WORLD_COUNTS[self.scale]
        if self.prior.world_count != expected_worlds:
            raise ValueError(
                f"{self.scale} requires exactly {expected_worlds} synthetic episodes/worlds"
            )
        active = self.active_seeds or (
            (self.root_seeds[0],) if self.scale == "S0" else self.root_seeds
        )
        if any(seed not in self.root_seeds for seed in active) or len(set(active)) != len(active):
            raise ValueError("active_seeds must be a unique subset of root_seeds")
        if self.scale == "S0" and len(active) != 1:
            raise ValueError("S0 uses exactly one active seed")
        if self.scale in {"S1", "S2"} and tuple(active) != self.root_seeds:
            raise ValueError(f"{self.scale} uses all three frozen root seeds")
        if self.scale == "S2" and not self.resumes_from:
            raise ValueError("S2 must declare the S1 pretraining lineage it resumes_from")
        if self.scale != "S2" and self.resumes_from is not None:
            raise ValueError("only S2 may declare a pretraining resumes_from lineage")
        if max(self.checkpoint_updates) > self.training.max_updates:
            raise ValueError("checkpoint_updates cannot exceed training.max_updates")
        return self


class FineTuneExperimentSpec(EvidenceSchema):
    schema_version: Literal["tabu.finetune-experiment.v1"] = "tabu.finetune-experiment.v1"
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    model_id: Literal["tabul"] = "tabul"
    model_spec_hash: str
    arm: TransferArm
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    subset_manifest_sha256: str | None = None
    dataset_passport: DatasetPassport | None = None
    label_budget: int = Field(gt=0)
    training: TransferTrainingConfig
    root_seed: int
    initialized_from: str | None = None
    resumes_from: str | None = None
    parameter_policy: Literal["full"] = "full"
    optimizer_reset: Literal[True] = True
    validation_checkpoint_selection: Literal["lowest_primary_loss_tie_earliest_update"] = (
        "lowest_primary_loss_tie_earliest_update"
    )
    test_access: Literal["sealed"] = "sealed"

    @field_validator(
        "model_spec_hash",
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "subset_manifest_sha256",
    )
    @classmethod
    def _required_hashes(cls, value: str, info: object) -> str:
        return require_sha256(value, field_name=getattr(info, "field_name", "sha256"))

    @field_validator("root_seed")
    @classmethod
    def _root_seed(cls, value: int) -> int:
        if value not in ROOT_SEEDS:
            raise ValueError("root_seed must be one of the frozen transfer seeds")
        return value

    @model_validator(mode="after")
    def _lineage_is_closed(self) -> FineTuneExperimentSpec:
        if (
            self.dataset_passport is not None
            and self.dataset_passport.dataset_id != self.dataset_id
        ):
            raise ValueError("dataset_passport.dataset_id must match dataset_id")
        if self.arm is TransferArm.PRETRAINED:
            if not self.initialized_from or self.resumes_from is not None:
                raise ValueError("pretrained fine-tuning requires initialized_from only")
        elif self.initialized_from is not None or self.resumes_from is not None:
            raise ValueError("scratch fine-tuning cannot declare a checkpoint lineage")
        return self


class TransferComparisonSpec(EvidenceSchema):
    schema_version: Literal["tabu.transfer-comparison.v1"] = "tabu.transfer-comparison.v1"
    comparison_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    pretrained_experiment_ids: tuple[str, ...] = Field(min_length=1)
    scratch_experiment_ids: tuple[str, ...] = Field(min_length=1)
    metric: TransferMetric
    primary_budget: Literal[128] = 128
    secondary_budgets: tuple[Literal[512, 2048], ...] = (512, 2048)
    max_catastrophic_negative_transfer_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    claim_boundary: Literal["tabul_synthetic_initialization_transfer_only"] = (
        "tabul_synthetic_initialization_transfer_only"
    )

    @model_validator(mode="after")
    def _paired_ids(self) -> TransferComparisonSpec:
        if len(self.pretrained_experiment_ids) != len(self.scratch_experiment_ids):
            raise ValueError("pretrained and scratch experiment lists must be paired")
        if len(set(self.pretrained_experiment_ids + self.scratch_experiment_ids)) != (
            len(self.pretrained_experiment_ids) + len(self.scratch_experiment_ids)
        ):
            raise ValueError("paired experiment ids must be unique")
        return self


class IclArm(StrEnum):
    ICL_PRETRAINED = "icl_pretrained"
    ICL_NO_PRETRAIN = "icl_no_pretrain"
    FINETUNE_SCRATCH = "finetune_scratch"


class IclHarnessSpec(EvidenceSchema):
    """Link 5 protocol: in-context learning after synthetic pretraining.

    The harness measures held-out-task accuracy as a function of context size
    and attributes any in-context ability to pretraining through the two
    control arms.  It is a probe specification: no receipt, result, or claim
    exists until a reviewed preregistration binds a pretrained checkpoint to
    a formal run.
    """

    schema_version: Literal["tabu.icl-harness.v1"] = "tabu.icl-harness.v1"
    harness_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    model_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    pretrain_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sizes: tuple[int, ...]
    arms: tuple[IclArm, ...]
    heldout_world_families: tuple[str, ...] = Field(min_length=1)
    heldout_real_task_ids: tuple[int, ...] = ()
    root_seed: int = Field(ge=0)
    claim_boundary: str = Field(min_length=1)

    @field_validator("context_sizes")
    @classmethod
    def _context_sizes_ascending(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(values) < 2:
            raise ValueError("context_sizes must contain at least two sweep points")
        if any(value < 0 for value in values):
            raise ValueError("context_sizes entries must be non-negative")
        if any(left >= right for left, right in pairwise(values)):
            raise ValueError("context_sizes must be strictly ascending")
        return values

    @model_validator(mode="after")
    def _arms_are_closed(self) -> IclHarnessSpec:
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("icl harness arms must be unique")
        if set(self.arms) != set(IclArm):
            raise ValueError(
                "icl harness requires the closed arm set "
                f"{sorted(arm.value for arm in IclArm)}"
            )
        return self


@dataclass(frozen=True, slots=True)
class SyntheticWorld:
    world_id: str
    family: SyntheticFamily
    features: torch.Tensor
    target: torch.Tensor
    modality: TransferModality
    target_type: Literal["numeric", "binary", "ordinal", "categorical"]
    missing_mask: torch.Tensor | None = None

    @property
    def content_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.synthetic-world.v1",
                "world_id": self.world_id,
                "family": self.family,
                "features": self.features,
                "target": self.target,
                "modality": self.modality,
                "target_type": self.target_type,
                "missing_mask": self.missing_mask,
            }
        )


def derive_seed(root_seed: int, namespace: str) -> int:
    """Derive a stable named RNG seed without depending on Python hash()."""

    if root_seed < 0 or not namespace.strip():
        raise ValueError("root_seed must be non-negative and namespace non-empty")
    digest = hashlib.sha256(f"tabu.transfer-seed.v1|{root_seed}|{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def stable_world_partition(world_id: str, *, heldout_family: bool = False) -> str:
    """Return the frozen 80/10/10 world partition."""

    if not world_id.strip():
        raise ValueError("world_id cannot be empty")
    if heldout_family:
        return "heldout_family"
    bucket = int(hashlib.sha256(world_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def materialize_synthetic_world(
    prior: SyntheticPriorSpec,
    family: SyntheticFamily,
    world_id: str,
    *,
    target_type: Literal["numeric", "binary", "ordinal", "categorical"] | None = None,
) -> SyntheticWorld:
    """Materialize one finite, deterministic mixed-type synthetic world."""

    if family not in set(prior.generator_families) | set(prior.heldout_generator_families):
        raise ValueError(f"family {family.value!r} is not declared by the prior")
    generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(prior.sampler_seed, f"world:{world_id}:{family.value}")
    )
    width = 4 + int(torch.randint(0, 12, (), generator=generator))
    rows = 128 + int(torch.randint(0, 128, (), generator=generator))
    features = torch.randn((rows, width), generator=generator, dtype=torch.float32)
    if target_type is None:
        target_type = prior.target_types[
            int(torch.randint(0, len(prior.target_types), (), generator=generator))
        ]
    if target_type not in prior.target_types:
        raise ValueError(f"target_type {target_type!r} is not declared by the prior")
    if family is SyntheticFamily.SMOOTH_SPARSE_SCM:
        target = 0.7 * features[:, 0] - 0.25 * features[:, 1].square()
        target = target + 0.2 * torch.tanh(features[:, 2])
    elif family is SyntheticFamily.TREE_THRESHOLD:
        target = (features[:, 0] > 0).float() + (features[:, 1] > 0.5).float()
        target = target + 0.2 * features[:, 2]
    elif family is SyntheticFamily.LATENT_FACTOR:
        latent = 0.8 * features[:, 0] + 0.4 * features[:, 1]
        target = latent + 0.3 * torch.sin(features[:, 2] + latent)
    else:
        noise = torch.randn((rows,), generator=generator) * (0.1 + features[:, 0].abs() * 0.2)
        target = features[:, 0] + noise
    missing_mask = None
    if family is SyntheticFamily.HETEROSCEDASTIC_SHIFT:
        missing_mask = torch.rand(features.shape, generator=generator) < 0.05
        features = features.masked_fill(missing_mask, 0.0)
    modality = (
        TransferModality.CLASSIFICATION if target_type != "numeric" else TransferModality.REGRESSION
    )
    if target_type == "binary":
        target = (target > target.median()).to(torch.int64)
    elif target_type == "ordinal":
        q1, q2, q3 = torch.quantile(target, torch.tensor((0.25, 0.5, 0.75)))
        target = torch.bucketize(target, torch.stack((q1, q2, q3))).to(torch.int64)
    elif target_type == "categorical":
        # A stable four-class projection exercises categorical token/readout paths.
        quantiles = torch.quantile(target, torch.tensor((0.25, 0.5, 0.75)))
        target = torch.bucketize(target, quantiles).to(torch.int64)
    return SyntheticWorld(world_id, family, features, target, modality, target_type, missing_mask)


def load_pretrained_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> torch.nn.Module:
    """Load only model tensors from a TabU training checkpoint.

    This is intentionally distinct from :meth:`Trainer.load_checkpoint`: a
    fine-tune arm imports weights but must not inherit optimizer, scheduler,
    step counter, or RNG state from pretraining.
    """

    from safetensors import safe_open

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        keys = checkpoint.keys()
        model_keys = tuple(key for key in keys if key.startswith("model."))
        if not model_keys:
            raise ValueError("pretrained checkpoint contains no model tensors")
        state = {key.removeprefix("model."): checkpoint.get_tensor(key) for key in model_keys}
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("pretrained model tensor keys do not match the target model")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError(f"pretrained model tensor {name!r} is incompatible")
    model.load_state_dict(state, strict=True)
    return model


def build_finetune_optimizer(
    model: torch.nn.Module,
    training: TransferTrainingConfig,
) -> torch.optim.Optimizer:
    """Construct a fresh optimizer, guaranteeing an optimizer reset."""

    return torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )


@dataclass(frozen=True, slots=True)
class SyntheticWorldBank:
    prior: SyntheticPriorSpec
    worlds: tuple[SyntheticWorld, ...]

    @classmethod
    def create(cls, prior: SyntheticPriorSpec, world_ids: Iterable[str]) -> SyntheticWorldBank:
        ids = tuple(world_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("world_ids must be non-empty and unique")
        families = tuple(prior.generator_families)
        worlds = tuple(
            materialize_synthetic_world(
                prior,
                families[index % len(families)],
                world_id,
                target_type=prior.target_types[index % len(prior.target_types)],
            )
            for index, world_id in enumerate(ids)
        )
        return cls(prior=prior, worlds=worlds)

    @property
    def content_hash(self) -> str:
        return canonical_hash(
            {
                "prior": self.prior,
                "worlds": tuple(world.content_hash for world in self.worlds),
            }
        )

    def partition(self, name: str) -> tuple[SyntheticWorld, ...]:
        if name not in {"train", "validation", "test", "heldout_family"}:
            raise ValueError(f"unknown synthetic partition: {name}")
        return tuple(
            world
            for world in self.worlds
            if stable_world_partition(
                world.world_id,
                heldout_family=world.family in set(self.prior.heldout_generator_families),
            )
            == name
        )


def build_synthetic_cache_manifest(
    prior: SyntheticPriorSpec, world_ids: Iterable[str]
) -> dict[str, object]:
    """Build a content-addressed, raw-data-free world cache manifest."""

    identifiers = tuple(world_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("world_ids must be non-empty and unique")
    if any(not identifier.strip() for identifier in identifiers):
        raise ValueError("world_ids cannot contain empty identifiers")
    payload: dict[str, object] = {
        "schema_version": "tabu.synthetic-world-cache-manifest.v1",
        "prior_id": prior.prior_id,
        "prior_spec_hash": prior.spec_hash,
        "world_ids": tuple(sorted(identifiers)),
    }
    return payload | {"manifest_sha256": canonical_hash(payload)}


@dataclass(frozen=True, slots=True)
class RealTransferTask:
    task_id: int
    modality: TransferModality
    dataset_id: str
    stratum: str
    development: bool = False
    lineage_family: str | None = None

    def __post_init__(self) -> None:
        if type(self.task_id) is not int or self.task_id <= 0:
            raise ValueError("real transfer task_id must be a positive integer")
        if not self.dataset_id.strip() or not self.stratum.strip():
            raise ValueError("real transfer task dataset_id and stratum are required")


@dataclass(frozen=True, slots=True)
class TransferPanel:
    panel_id: str
    tasks: tuple[RealTransferTask, ...]
    sealed_test: bool = True

    def __post_init__(self) -> None:
        if not self.panel_id.strip() or not self.tasks:
            raise ValueError("transfer panels require an id and at least one task")
        ids = tuple(task.task_id for task in self.tasks)
        if len(ids) != len(set(ids)):
            raise ValueError("transfer panel task ids must be unique")

    @property
    def content_hash(self) -> str:
        return canonical_hash(
            {
                "panel_id": self.panel_id,
                "tasks": self.tasks,
                "sealed_test": self.sealed_test,
            }
        )

    def require_task(self, task_id: int) -> RealTransferTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)


def r1_panel() -> TransferPanel:
    return TransferPanel(
        panel_id="tabu-r1-real-transfer-v1",
        tasks=(
            RealTransferTask(
                43, TransferModality.CLASSIFICATION, "spambase", "binary_numeric", development=True
            ),
            RealTransferTask(
                3, TransferModality.CLASSIFICATION, "kr-vs-kp", "binary_mixed", development=True
            ),
            RealTransferTask(
                14969,
                TransferModality.CLASSIFICATION,
                "GesturePhaseSegmentationProcessed",
                "multiclass_numeric",
                development=True,
            ),
            RealTransferTask(
                45, TransferModality.CLASSIFICATION, "splice", "multiclass_mixed", development=True
            ),
            RealTransferTask(
                361249, TransferModality.REGRESSION, "white_wine", "small_numeric", development=True
            ),
            RealTransferTask(
                361234, TransferModality.REGRESSION, "abalone", "small_mixed", development=True
            ),
            RealTransferTask(
                361260,
                TransferModality.REGRESSION,
                "miami_housing",
                "medium_numeric",
                development=True,
            ),
            RealTransferTask(
                361267,
                TransferModality.REGRESSION,
                "brazilian_houses",
                "medium_mixed",
                development=True,
            ),
        ),
    )


def r2_panel() -> TransferPanel:
    tasks = tuple(
        RealTransferTask(
            task_id,
            TransferModality.CLASSIFICATION,
            f"openml-{task_id}",
            "cc18",
            lineage_family="cc18-derived",
        )
        for task_id in R2_CLASSIFICATION_TASKS
    ) + tuple(
        RealTransferTask(
            task_id,
            TransferModality.REGRESSION,
            f"openml-{task_id}",
            "ctr23",
            lineage_family="ctr23-derived",
        )
        for task_id in R2_REGRESSION_TASKS
    )
    return TransferPanel(panel_id="tabu-r2-real-transfer-v1", tasks=tasks)


def stable_row_order(
    row_ids: Sequence[str], *, labels: Sequence[object] | None = None, classification: bool = False
) -> tuple[int, ...]:
    """Return a deterministic, prefix-preserving row order for label budgets."""

    if not row_ids or len(row_ids) != len(set(row_ids)):
        raise ValueError("row_ids must be non-empty and unique")
    if labels is not None and len(labels) != len(row_ids):
        raise ValueError("labels must align with row_ids")
    keyed = sorted(
        range(len(row_ids)),
        key=lambda index: hashlib.sha256(f"tabu-real-row-v1|{row_ids[index]}".encode()).hexdigest(),
    )
    if not classification or labels is None:
        return tuple(keyed)
    buckets: dict[str, list[int]] = {}
    for index in keyed:
        buckets.setdefault(str(labels[index]), []).append(index)
    result: list[int] = []
    for index in range(max(len(bucket) for bucket in buckets.values())):
        for label in sorted(buckets):
            if index < len(buckets[label]):
                result.append(buckets[label][index])
    return tuple(result)


def stable_label_subset(
    row_ids: Sequence[str],
    budget: int,
    *,
    labels: Sequence[object] | None = None,
    classification: bool = False,
) -> tuple[int, ...]:
    """Return the first ``budget`` rows of the frozen hash order.

    The prefix property makes budgets 128, 512, and 2048 comparable.
    Classification uses the round-robin order and fails closed if a requested
    budget cannot include every observed class.
    """

    if type(budget) is not int or budget <= 0 or budget > len(row_ids):
        raise ValueError("label budget must be a positive integer within the row count")
    if classification and labels is None:
        raise ValueError("classification label subsets require labels")
    order = stable_row_order(row_ids, labels=labels, classification=classification)
    if classification and labels is not None:
        classes = {str(value) for value in labels}
        selected_classes = {str(labels[index]) for index in order[:budget]}
        if len(classes) > budget or selected_classes != classes:
            raise ValueError("label budget cannot cover every observed classification class")
    return tuple(order[:budget])


def normalized_nll(nll: float, classes: int) -> float:
    if not math.isfinite(nll) or nll < 0.0 or classes < 2:
        raise ValueError("nll must be finite and classes must be at least two")
    return nll / math.log(classes)


def scaled_rmse(rmse: float, train_target_std: float) -> float:
    if (
        not math.isfinite(rmse)
        or rmse < 0.0
        or not math.isfinite(train_target_std)
        or train_target_std <= 0.0
    ):
        raise ValueError("rmse must be finite and target standard deviation must be positive")
    return rmse / train_target_std


def select_r0_schedule(
    candidate_validation_losses: Mapping[tuple[float, int], Sequence[float]],
) -> tuple[float, int]:
    """Choose one global R0 schedule with the frozen deterministic tie-break."""

    expected = {(lr, updates) for lr in R0_LEARNING_RATES for updates in R0_MAX_UPDATES}
    if set(candidate_validation_losses) != expected:
        raise ValueError("R0 schedule candidates must cover all 3 learning rates x 2 update caps")
    scores: dict[tuple[float, int], float] = {}
    for candidate, losses in candidate_validation_losses.items():
        values = tuple(float(value) for value in losses)
        if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("R0 validation losses must be finite and non-negative")
        scores[candidate] = statistics.fmean(values)
    return min(scores, key=lambda candidate: (scores[candidate], candidate[1], candidate[0]))


def exact_positive_sign_pvalue(gains: Sequence[float]) -> float:
    """Exact one-sided sign-test p-value, dropping zero differences."""

    nonzero = [gain for gain in gains if not math.isclose(gain, 0.0, abs_tol=1e-15)]
    if not nonzero:
        return 1.0
    positives = sum(gain > 0.0 for gain in nonzero)
    n = len(nonzero)
    return sum(math.comb(n, k) for k in range(positives, n + 1)) / (2.0**n)


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in pvalues.values()):
        raise ValueError("p-values must be finite and in [0, 1]")
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def paired_cluster_bootstrap_median(
    gains_by_cluster: Mapping[str, Sequence[float]],
    *,
    replicates: int = 2_000,
    seed: int = ROOT_SEEDS[0],
) -> tuple[float, float, float]:
    """Return ``(median, lower_95, upper_95)`` with cluster-level resampling.

    A derivative dataset family contributes one cluster, so near-duplicate
    tables cannot be counted as independent observations.  The implementation
    is deterministic and uses a local generator; it never touches process RNG.
    """

    if not gains_by_cluster or any(not values for values in gains_by_cluster.values()):
        raise ValueError("cluster bootstrap requires non-empty clusters")
    if type(replicates) is not int or replicates < 100:
        raise ValueError("replicates must be an integer of at least 100")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    clusters = tuple(sorted(gains_by_cluster))
    observed = tuple(value for name in clusters for value in gains_by_cluster[name])
    point = statistics.median(observed)
    generator = torch.Generator(device="cpu").manual_seed(derive_seed(seed, "cluster-bootstrap"))
    samples: list[float] = []
    cluster_count = len(clusters)
    for _ in range(replicates):
        sampled = []
        choices = torch.randint(cluster_count, (cluster_count,), generator=generator)
        for choice in choices.tolist():
            sampled.extend(gains_by_cluster[clusters[choice]])
        samples.append(statistics.median(sampled))
    samples.sort()
    lower = samples[max(0, int(0.025 * replicates) - 1)]
    upper = samples[min(replicates - 1, int(0.975 * replicates))]
    return point, lower, upper


class TransferObservation(EvidenceSchema):
    task_id: int = Field(gt=0)
    modality: TransferModality
    budget: int = Field(gt=0)
    seed: int
    scratch_loss: float = Field(ge=0.0)
    pretrained_loss: float = Field(ge=0.0)
    completed: Literal[True] = True

    @field_validator("scratch_loss", "pretrained_loss")
    @classmethod
    def _finite_loss(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("transfer losses must be finite")
        return value

    @field_validator("seed")
    @classmethod
    def _seed(cls, value: int) -> int:
        if value not in ROOT_SEEDS:
            raise ValueError("observation seed must be one of the frozen transfer seeds")
        return value

    @property
    def gain(self) -> float:
        return self.scratch_loss - self.pretrained_loss

    @property
    def relative_deterioration(self) -> float:
        """Positive values mean pretrained is worse, relative to scratch."""

        if self.scratch_loss == 0.0:
            return 0.0 if self.pretrained_loss == 0.0 else math.inf
        return (self.pretrained_loss - self.scratch_loss) / self.scratch_loss


@dataclass(frozen=True, slots=True)
class TransferGateResult:
    gate: str
    passed: bool
    reasons: tuple[str, ...]
    statistics: Mapping[str, float | int | None]


@dataclass(frozen=True, slots=True)
class TransferTrainingResult:
    """In-memory result of one paired arm training slice.

    This helper intentionally does not issue a receipt or claim formal
    evidence.  Formal execution must bind the returned model/checkpoint to the
    existing :class:`RunIdentity` and receipt writer.  Keeping the loop here
    nevertheless gives both arms one auditable optimizer/reset implementation.
    """

    arm: TransferArm
    seed: int
    updates: int
    losses: tuple[float, ...]
    initial_model_hash: str
    final_model_hash: str
    initialized_from: str | None = None

    @property
    def initial_loss(self) -> float:
        return self.losses[0] if self.losses else math.nan

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else math.nan


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def train_pretrain_batches(
    model: torch.nn.Module,
    batches: Iterable[tuple[object, object]],
    training: TransferTrainingConfig,
    *,
    seed: int,
    max_updates: int | None = None,
) -> TransferTrainingResult:
    """Train a synthetic prior slice with deterministic named seed derivation."""

    return _train_transfer_batches(
        model,
        batches,
        training,
        arm=TransferArm.PRETRAINED,
        seed=seed,
        max_updates=max_updates,
    )


def train_finetune_batches(
    model: torch.nn.Module,
    batches: Iterable[tuple[object, object]],
    training: TransferTrainingConfig,
    *,
    seed: int,
    arm: TransferArm,
    checkpoint_path: str | None = None,
    max_updates: int | None = None,
) -> TransferTrainingResult:
    """Run one full-parameter fine-tune arm with an explicit lineage boundary."""

    if arm is TransferArm.PRETRAINED and checkpoint_path is None:
        raise ValueError("synthetic_pretrained fine-tuning requires initialized_from checkpoint")
    if arm is TransferArm.SCRATCH and checkpoint_path is not None:
        raise ValueError("matched_scratch fine-tuning cannot load a checkpoint")
    if checkpoint_path is not None:
        load_pretrained_weights(model, checkpoint_path)
    return _train_transfer_batches(
        model,
        batches,
        training,
        arm=arm,
        seed=seed,
        max_updates=max_updates,
        initialized_from=checkpoint_path,
    )


def _train_transfer_batches(
    model: torch.nn.Module,
    batches: Iterable[tuple[object, object]],
    training: TransferTrainingConfig,
    *,
    arm: TransferArm,
    seed: int,
    max_updates: int | None,
    initialized_from: str | None = None,
) -> TransferTrainingResult:
    if seed not in ROOT_SEEDS:
        raise ValueError("transfer training seed must be one of the frozen root seeds")
    update_budget = training.max_updates if max_updates is None else max_updates
    if type(update_budget) is not int or not 0 < update_budget <= training.max_updates:
        raise ValueError("max_updates must be in [1, training.max_updates]")
    materialized = tuple(batches)
    if not materialized:
        raise ValueError("transfer training requires at least one evidence/TruthSidecar batch")
    torch.manual_seed(derive_seed(seed, f"{arm.value}:model"))
    initial_hash = _model_state_hash(model)
    optimizer = build_finetune_optimizer(model, training)
    from tabu_lab.training import Trainer

    trainer = Trainer(
        model,
        optimizer=optimizer,
        learning_rate=training.learning_rate,
        max_gradient_norm=training.gradient_clip_norm,
    )
    losses: list[float] = []
    for update in range(update_budget):
        evidence, truth = materialized[update % len(materialized)]
        step = trainer.train_step(evidence, truth)
        loss = float(step.loss.total.detach().cpu())
        if not math.isfinite(loss):
            raise ValueError(f"non-finite transfer loss at update {step.step}")
        losses.append(loss)
    return TransferTrainingResult(
        arm=arm,
        seed=seed,
        updates=len(losses),
        losses=tuple(losses),
        initial_model_hash=initial_hash,
        final_model_hash=_model_state_hash(model),
        initialized_from=initialized_from,
    )


def evaluate_r1_gate(observations: Sequence[TransferObservation]) -> TransferGateResult:
    _validate_observation_pairs(observations, panel=r1_panel())
    by_task_budget: dict[tuple[int, int], list[float]] = {}
    for observation in observations:
        by_task_budget.setdefault((observation.task_id, observation.budget), []).append(
            observation.gain
        )
    task_gains = {
        task_id: sum(gains) / len(gains)
        for (task_id, budget), gains in by_task_budget.items()
        if budget == 128 and len(gains) == len(ROOT_SEEDS)
    }
    r1 = r1_panel()
    cls_ids = {
        task.task_id for task in r1.tasks if task.modality is TransferModality.CLASSIFICATION
    }
    reg_ids = {task.task_id for task in r1.tasks if task.modality is TransferModality.REGRESSION}
    wins = sum(value > 0.0 for value in task_gains.values())
    cls_wins = sum(task_gains.get(task_id, 0.0) > 0.0 for task_id in cls_ids)
    reg_wins = sum(task_gains.get(task_id, 0.0) > 0.0 for task_id in reg_ids)
    b512_task_gains = {
        task_id: statistics.median(gains)
        for (task_id, budget), gains in by_task_budget.items()
        if budget == 512 and len(gains) == len(ROOT_SEEDS)
    }
    reasons: list[str] = []
    if len(task_gains) != 8:
        reasons.append("incomplete_r1_task_seed_coverage")
    if wins < 6:
        reasons.append("fewer_than_6_of_8_tasks_win_at_budget_128")
    if cls_wins < 3 or reg_wins < 3:
        reasons.append("each_modality_requires_3_of_4_wins")
    if cls_ids and not any(task_id in task_gains for task_id in cls_ids):
        reasons.append("classification_panel_missing")
    if reg_ids and not any(task_id in task_gains for task_id in reg_ids):
        reasons.append("regression_panel_missing")
    if any(budget == 512 for _, budget in by_task_budget) and len(b512_task_gains) != 8:
        reasons.append("incomplete_r1_budget_512_seed_coverage")
    if b512_task_gains and statistics.median(b512_task_gains.values()) < 0.0:
        reasons.append("budget_512_median_gain_is_negative")
    return TransferGateResult(
        gate="R1",
        passed=not reasons,
        reasons=tuple(reasons),
        statistics={
            "tasks": len(task_gains),
            "wins": wins,
            "classification_wins": cls_wins,
            "regression_wins": reg_wins,
            "budget_512_median_gain": (
                statistics.median(b512_task_gains.values()) if b512_task_gains else None
            ),
        },
    )


def evaluate_r2_gate(observations: Sequence[TransferObservation]) -> TransferGateResult:
    panel = r2_panel()
    _validate_observation_pairs(observations, panel=panel)
    task_modality = {task.task_id: task.modality for task in panel.tasks}
    task_gains: dict[tuple[int, int], list[float]] = {}
    for observation in observations:
        if observation.task_id not in task_modality:
            continue
        task_gains.setdefault((observation.task_id, observation.budget), []).append(
            observation.gain
        )
    primary = {
        task_id: sum(gains) / len(gains)
        for (task_id, budget), gains in task_gains.items()
        if budget == 128 and len(gains) == len(ROOT_SEEDS)
    }
    modality_gains = {
        modality: [gain for task_id, gain in primary.items() if task_modality[task_id] is modality]
        for modality in TransferModality
    }
    pvalues = {
        modality.value: exact_positive_sign_pvalue(gains)
        for modality, gains in modality_gains.items()
    }
    adjusted = holm_adjust(pvalues)
    reasons: list[str] = []
    required = {TransferModality.CLASSIFICATION: 51, TransferModality.REGRESSION: 28}
    for modality, gains in modality_gains.items():
        completed = len(gains)
        if completed < required[modality]:
            reasons.append(f"{modality.value}_coverage_below_required")
        if not gains or statistics.median(gains) <= 0.0:
            reasons.append(f"{modality.value}_median_gain_not_positive")
        if adjusted[modality.value] >= 0.05:
            reasons.append(f"{modality.value}_holm_adjusted_sign_test_not_significant")
    task_primary_pairs: dict[int, list[tuple[float, float]]] = {}
    for observation in observations:
        if observation.task_id in task_modality and observation.budget == 128:
            task_primary_pairs.setdefault(observation.task_id, []).append(
                (observation.scratch_loss, observation.pretrained_loss)
            )
    task_relative_deterioration = []
    for pairs in task_primary_pairs.values():
        if len(pairs) != len(ROOT_SEEDS):
            continue
        scratch = statistics.fmean(item[0] for item in pairs)
        pretrained = statistics.fmean(item[1] for item in pairs)
        task_relative_deterioration.append(
            math.inf if scratch == 0.0 and pretrained > 0.0 else (
                0.0 if scratch == 0.0 else (pretrained - scratch) / scratch
            )
        )
    catastrophic = sum(value > 0.10 for value in task_relative_deterioration) / max(
        len(task_relative_deterioration), 1
    )
    if catastrophic > 0.10:
        reasons.append("catastrophic_negative_transfer_fraction_exceeds_10_percent")
    return TransferGateResult(
        gate="R2",
        passed=not reasons,
        reasons=tuple(reasons),
        statistics={
            "completed": len(primary),
            "classification_completed": len(modality_gains[TransferModality.CLASSIFICATION]),
            "regression_completed": len(modality_gains[TransferModality.REGRESSION]),
            "classification_p_holm": adjusted["classification"],
            "regression_p_holm": adjusted["regression"],
            "negative_transfer_fraction": catastrophic,
        },
    )


def _validate_observation_pairs(
    observations: Sequence[TransferObservation], *, panel: TransferPanel
) -> None:
    task_modality = {task.task_id: task.modality for task in panel.tasks}
    seen: set[tuple[int, int, int]] = set()
    for observation in observations:
        key = (observation.task_id, observation.budget, observation.seed)
        if key in seen:
            raise ValueError("duplicate transfer observation for task/budget/seed")
        seen.add(key)
        expected = task_modality.get(observation.task_id)
        if expected is not None and expected is not observation.modality:
            raise ValueError("observation modality does not match frozen transfer panel")


def transfer_manifest() -> dict[str, object]:
    """Return a public, raw-data-free panel manifest for catalog generation."""

    def panel_payload(panel: TransferPanel) -> dict[str, object]:
        return {
            "panel_id": panel.panel_id,
            "sealed_test": panel.sealed_test,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "modality": task.modality.value,
                    "dataset_id": task.dataset_id,
                    "stratum": task.stratum,
                    "development": task.development,
                    "lineage_family": task.lineage_family,
                }
                for task in panel.tasks
            ],
        }

    return {
        "schema_version": "tabu.transfer-panel-manifest.v1",
        "r0_development_datasets": R0_DEVELOPMENT_DATASETS,
        "r1": panel_payload(r1_panel()),
        "r2": panel_payload(r2_panel()),
        "capacity_blocked_cc18": R2_CAPACITY_BLOCKED_CC18,
        "label_budgets": (128, 512, 2048),
        "root_seeds": ROOT_SEEDS,
        "scale_world_counts": SCALE_WORLD_COUNTS,
        "checkpoint_ladder": CHECKPOINT_LADDER,
        "split_policy": "task_provided_repeat0_fold0_test_fold1_validation_train_intersection",
        "claim_boundary": "tabul_synthetic_initialization_transfer_only",
        "raw_files_in_git": False,
    }


__all__ = [
    "CHECKPOINT_LADDER",
    "R0_DEVELOPMENT_DATASETS",
    "R0_LEARNING_RATES",
    "R0_MAX_UPDATES",
    "R1_CLASSIFICATION_TASKS",
    "R1_REGRESSION_TASKS",
    "R2_CAPACITY_BLOCKED_CC18",
    "R2_CLASSIFICATION_TASKS",
    "R2_REGRESSION_TASKS",
    "ROOT_SEEDS",
    "SCALE_WORLD_COUNTS",
    "ComputeEnvelope",
    "DatasetPassport",
    "EvidenceLineageRelation",
    "FineTuneExperimentSpec",
    "PretrainExperimentSpec",
    "RealTransferTask",
    "SyntheticFamily",
    "SyntheticPriorSpec",
    "SyntheticWorld",
    "SyntheticWorldBank",
    "TransferArm",
    "TransferComparisonSpec",
    "TransferGateResult",
    "TransferMetric",
    "TransferModality",
    "TransferObservation",
    "TransferPanel",
    "TransferSplitManifest",
    "TransferTrainingConfig",
    "TransferTrainingResult",
    "build_synthetic_cache_manifest",
    "derive_seed",
    "evaluate_r1_gate",
    "evaluate_r2_gate",
    "exact_positive_sign_pvalue",
    "holm_adjust",
    "materialize_synthetic_world",
    "normalized_nll",
    "paired_cluster_bootstrap_median",
    "r1_panel",
    "r2_panel",
    "resolve_task_provided_split",
    "scaled_rmse",
    "select_r0_schedule",
    "stable_label_subset",
    "stable_row_order",
    "stable_world_partition",
    "train_finetune_batches",
    "train_pretrain_batches",
    "transfer_manifest",
]
