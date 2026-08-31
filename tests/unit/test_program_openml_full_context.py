from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tabu_lab.contracts import canonical_hash
from tabu_lab.evolution import EvidenceStatus, file_sha256
from tabu_lab.evolution.models import ProgramArtifact
from tabu_lab.evolution.openml_full_context import (
    ArmDatasetResult,
    ArmSplitResult,
    BaselineSplitResult,
    DatasetExecution,
    ProgramOpenMLDatasetReceipt,
    aggregate_program_openml_full_context,
    load_openml_data_panel,
    load_program_openml_full_context_request,
)
from tabu_lab.experiments.query_row_openml_full_context import _forward_query_response_only
from tabu_lab.experiments.tabubase_real_benchmark import RealDataset
from tabu_lab.experiments.tabubase_real_icl import (
    build_real_icl_episode,
    prepare_real_icl_split,
)
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig

ROOT = Path(__file__).resolve().parents[2]
DATA_PANEL = ROOT / "experiments/evolution/openml-new6-data-panel-1.0.0.yaml"
REQUEST = ROOT / "experiments/evolution/v3.1-best-frozen-full-context-new6-1.0.0.yaml"


def _toy_episode(task: str) -> tuple[Any, Any]:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 3)).astype(np.float32)
    if task == "classification":
        response = np.asarray([index % 2 for index in range(40)], dtype=np.int64)
    else:
        response = rng.normal(size=40).astype(np.float32)
    split = prepare_real_icl_split(
        RealDataset("toy", task, features, response, "unit-test"),
        split_seed=1729,
    )
    evidence, _ = build_real_icl_episode(
        split,
        context_size=len(split.train_indices),
        query_indices=split.query_indices,
        shuffled_context=False,
    )
    return split, evidence


def test_checked_in_request_binds_independent_current_query_siblings() -> None:
    panel = load_openml_data_panel(DATA_PANEL).panel
    request = load_program_openml_full_context_request(REQUEST)

    assert request.evidence_status is EvidenceStatus.LOCAL_UNISSUED
    assert request.data_panel.canonical_payload_sha256 == panel.panel_hash
    assert tuple(item.dataset_id for item in panel.datasets) == (
        "banknote_authentication",
        "segment",
        "spambase",
        "airfoil_self_noise",
        "concrete_compressive_strength",
        "qsar_fish_toxicity",
    )
    assert [(arm.arm_id, arm.model_contract_ref, arm.checkpoint_step) for arm in request.arms] == [
        ("tabu_base", "tabu.query.base@0.1.0", 500),
        ("tabu_row", "tabu.query.row@0.2.0", 1500),
    ]


@pytest.mark.parametrize("model_id", ["tabu.query.base", "tabu.query.row"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_response_only_adapter_matches_dense_current_query_family(
    model_id: str,
    task: str,
) -> None:
    split, evidence = _toy_episode(task)
    options: dict[str, Any] = {
        "config": ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=4,
            max_features=256,
        ),
        "profile": "supervised.label_broadcast.v1",
    }
    if model_id == "tabu.query.row":
        options["row_token_count"] = 4
    model = build_model(model_id, **options).eval()
    context_rows = len(split.train_indices)
    with torch.inference_mode():
        dense = model._forward_dense(evidence, emit_trace=False)
        probabilities, predicted = _forward_query_response_only(
            model,
            evidence,
            context_rows=context_rows,
            classes=split.classes,
            query_chunk_rows=7,
            device=torch.device("cpu"),
        )
    if task == "classification":
        values = dense.entries["distribution"].values
        assert values is not None and probabilities is not None
        direct = values[context_rows:, -1, : split.classes].numpy()
        direct /= direct.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(probabilities[0], direct, rtol=1.0e-5, atol=1.0e-5)
    else:
        raw = dense.auxiliaries["numeric_raw_prediction"]
        assert predicted is not None
        np.testing.assert_allclose(
            predicted[0],
            raw[context_rows:, -1].numpy(),
            rtol=1.0e-5,
            atol=1.0e-5,
        )


def _dataset_receipt(dataset_id: str) -> ProgramOpenMLDatasetReceipt:
    loaded = load_openml_data_panel(DATA_PANEL)
    request = load_program_openml_full_context_request(REQUEST)
    identity = next(item for item in loaded.panel.datasets if item.dataset_id == dataset_id)
    primary = "normalized_nll" if identity.task == "classification" else "scaled_rmse"
    linear_value = 0.5 if identity.task == "classification" else 0.7
    arm_values = {
        "tabu_base": 0.4 if identity.task == "classification" else 0.8,
        "tabu_row": 0.45 if identity.task == "classification" else 0.6,
    }
    split_manifests = tuple(
        {"dataset_id": dataset_id, "split_seed": seed}
        for seed in request.split_protocol.split_seeds
    )
    baselines = tuple(
        BaselineSplitResult(
            split_seed=seed,
            metrics={primary: linear_value},
            fit={"estimator": "linear", "fit_rows": 10},
        )
        for seed in request.split_protocol.split_seeds
    )
    arm_results: list[ArmDatasetResult] = []
    for frozen in request.arms:
        prediction_hash = canonical_hash({"dataset": dataset_id, "arm": frozen.arm_id})
        splits = tuple(
            ArmSplitResult(
                split_seed=seed,
                metrics={primary: arm_values[frozen.arm_id]},
                prediction_sha256=prediction_hash,
                truth_sidecar_sha256=canonical_hash(
                    {"dataset": dataset_id, "split": seed, "truth": "original"}
                ),
                truth_substitution_checked=(index == 0),
                substituted_truth_sidecar_sha256=(
                    canonical_hash({"dataset": dataset_id, "truth": "substituted"})
                    if index == 0
                    else None
                ),
                truth_substitution_prediction_sha256=(
                    prediction_hash if index == 0 else None
                ),
                truth_substitution_prediction_unchanged=True if index == 0 else None,
            )
            for index, seed in enumerate(request.split_protocol.split_seeds)
        )
        arm_results.append(
            ArmDatasetResult(
                arm_id=frozen.arm_id,
                program_ref=frozen.program_ref,
                snapshot_hash=frozen.snapshot_hash,
                run_identity_hash=frozen.run_identity_hash,
                checkpoint=ProgramArtifact(
                    name=frozen.checkpoint_name,
                    sha256=frozen.checkpoint_sha256,
                    size_bytes=1,
                ),
                checkpoint_sidecar=ProgramArtifact(
                    name=Path(frozen.checkpoint_name).with_suffix(".program.json").name,
                    sha256=frozen.checkpoint_sidecar_sha256,
                    size_bytes=1,
                ),
                checkpoint_metadata_hash=frozen.checkpoint_metadata_hash,
                selection_evaluation_receipt_hash=frozen.selection_evaluation_receipt_hash,
                model_state_hash_before="a" * 64,
                model_state_hash_after="a" * 64,
                splits=splits,
            )
        )
    source_manifest: dict[str, Any] = {"dataset_id": dataset_id, "source": "unit-test"}
    execution = DatasetExecution(
        evaluation_source_revision="b" * 40,
        evaluation_source_archive_sha256="c" * 64,
        source_tree_sha256="d" * 64,
        evaluation_tool=ProgramArtifact(name="tool.py", sha256="e" * 64, size_bytes=1),
        environment={"device": "cpu"},
    )
    payload: dict[str, Any] = {
        "schema_version": "tabu.program-openml-full-context-dataset-receipt.v1",
        "evidence_status": EvidenceStatus.LOCAL_UNISSUED,
        "claim_boundary": request.claim_boundary,
        "request": request,
        "request_hash": request.request_hash,
        "source_repository_hash": request.source_repository_hash,
        "data_panel": loaded.panel,
        "data_panel_hash": loaded.panel.panel_hash,
        "data_panel_artifact": ProgramArtifact(
            name=DATA_PANEL.name,
            sha256=file_sha256(DATA_PANEL),
            size_bytes=DATA_PANEL.stat().st_size,
        ),
        "dataset_id": dataset_id,
        "task": identity.task,
        "primary_metric": primary,
        "source_manifest": source_manifest,
        "source_manifest_sha256": canonical_hash(source_manifest),
        "split_manifests": split_manifests,
        "linear_baseline": baselines,
        "arms": tuple(arm_results),
        "execution": execution,
    }
    return ProgramOpenMLDatasetReceipt(**payload, receipt_hash=canonical_hash(payload))


def _write_dataset_receipt(path: Path, receipt: ProgramOpenMLDatasetReceipt) -> None:
    path.write_text(
        json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_aggregate_requires_all_six_and_computes_task_macros(tmp_path: Path) -> None:
    request = load_program_openml_full_context_request(REQUEST)
    panel = load_openml_data_panel(DATA_PANEL).panel
    paths: list[Path] = []
    for item in panel.datasets:
        path = tmp_path / f"{item.dataset_id}.json"
        _write_dataset_receipt(path, _dataset_receipt(item.dataset_id))
        paths.append(path)

    output = tmp_path / "panel.json"
    receipt = aggregate_program_openml_full_context(
        request=request,
        receipt_paths=tuple(paths),
        output=output,
    )

    assert output.is_file()
    assert receipt.arm_panel_success == {"tabu_base": False, "tabu_row": True}
    assert receipt.task_macros["classification"]["arms"]["tabu_base"]["dataset_wins"] == 3
    assert receipt.task_macros["regression"]["arms"]["tabu_base"]["dataset_wins"] == 0
    assert receipt.task_macros["regression"]["arms"]["tabu_row"]["beats_linear"] is True

    with pytest.raises(ValueError, match="refusing to overwrite"):
        aggregate_program_openml_full_context(
            request=request,
            receipt_paths=tuple(paths),
            output=output,
        )
    with pytest.raises(ValueError, match="ordered unique receipts"):
        aggregate_program_openml_full_context(
            request=request,
            receipt_paths=tuple(paths[:-1]),
            output=tmp_path / "missing.json",
        )
