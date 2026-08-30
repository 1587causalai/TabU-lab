#!/usr/bin/env python3
"""Create the canonical TabUL x sklearn-Diabetes R1 preregistration."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tabu_lab.contracts import canonical_hash
from tabu_lab.experiments.contracts import (
    BaselineRole,
    BaselineSpec,
    DatasetAdapterSpec,
    DatasetOrigin,
    EpisodeSchedule,
    FitDatasetSpec,
    FitExecutionConfig,
    FitExperimentSpec,
    FitPassGate,
    FitSeedConfig,
    FitStage,
    FitTrainingConfig,
    ModelSemanticConfig,
    RedistributionPolicy,
    ScheduleSampling,
)
from tabu_lab.experiments.splits import RowPartition, RowSplitManifest
from tabu_lab.registry import get_model_spec

EXPERIMENT_ID = "R1-001-tabul-sklearn-diabetes-regression-v1"
DATASET_HASH = "9de43a4c610d77e0bda81bfd524c92a8ebb7c9bf53deb56f2e84d9f0511e187c"
SOURCE_HASH = "0ea9e8f5a8f0c85548add40324bdab3d49e73fe335689556e233fbf09b3f2d3a"


def build_spec() -> FitExperimentSpec:
    model_spec = get_model_spec("tabul")
    split = RowSplitManifest(
        dataset_id="sklearn-diabetes",
        dataset_hash=DATASET_HASH,
        split_id="R1-001-tabul-sklearn-diabetes-regression-v1-split",
        fit_partition="train",
        strategy="frozen_diabetes_split_v1",
        seed=130363,
        require_complete=True,
        partitions=(
            RowPartition(name="train", row_ids=tuple(str(i) for i in range(256))),
            RowPartition(name="validation", row_ids=tuple(str(i) for i in range(256, 320))),
            RowPartition(name="test", row_ids=tuple(str(i) for i in range(320, 442))),
        ),
    )
    semantic = ModelSemanticConfig(
        reference={"backend": "dense_reference_v0"},
        label_columns=(-1,),
        label_address_plan="matched_uf",
        augmented_readout_geometry="matched_uf",
    )
    return FitExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        stage=FitStage.R1,
        contract_id="tabul",
        contract_version=model_spec.contract_version,
        model_spec=model_spec,
        model_spec_hash=canonical_hash(model_spec),
        dataset=FitDatasetSpec(
            dataset_id="sklearn-diabetes",
            origin=DatasetOrigin.CLASSIC,
            source_uri="sklearn.datasets.load_diabetes",
            source_sha256=SOURCE_HASH,
            dataset_hash=DATASET_HASH,
            license_id="BSD-3-Clause",
            redistribution=RedistributionPolicy.METADATA_ONLY,
            adapter=DatasetAdapterSpec(
                adapter_id="sklearn-diabetes-frozen",
                adapter_version="1.0.0",
            ),
        ),
        split=split,
        episode_schedule=EpisodeSchedule(
            schedule_id="R1-001-tabul-sklearn-diabetes-regression-v1-schedule",
            sampling=ScheduleSampling.DETERMINISTIC_SHUFFLE,
            episode_count=1,
            targets_per_episode=1,
            target_families=("label",),
            target_origins=("query",),
            sampler_seed=104729,
            order_seed=130363,
        ),
        semantic=semantic,
        training=FitTrainingConfig(
            learning_rate=1.0e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            max_updates=10_000,
            max_epochs=100,
            wall_clock_budget_minutes=120,
            exact_resume=True,
        ),
        execution=FitExecutionConfig(device="cpu", device_index=None),
        seeds=FitSeedConfig(
            model_seeds=(1729, 2718, 31415),
            data_seed=104729,
            split_seed=130363,
            episode_order_seed=130363,
        ),
        target_families=("label",),
        baselines=(
            BaselineSpec(baseline_id="train-mean", role=BaselineRole.TRIVIAL),
            BaselineSpec(baseline_id="standardized-ridge", role=BaselineRole.DIAGNOSTIC),
        ),
        pass_gate=FitPassGate(
            stage=FitStage.R1,
            max_loss_ratio=0.25,
            max_trivial_baseline_ratio=0.80,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/fit-first/R1") / f"{EXPERIMENT_ID}/preregistration.yaml",
    )
    args = parser.parse_args()
    spec = build_spec()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(
            spec.model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=False,
            width=120,
        ),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
