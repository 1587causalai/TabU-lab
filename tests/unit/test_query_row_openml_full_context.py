from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig
from tabu_lab.experiments.query_row_openml_full_context import (
    _forward_query_response_only,
    load_query_openml_full_context_panel_manifest,
)
from tabu_lab.experiments.tabubase_real_benchmark import RealDataset
from tabu_lab.experiments.tabubase_real_icl import (
    build_real_icl_episode,
    prepare_real_icl_split,
)


def _toy_episode(task: str):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 3)).astype(np.float32)
    if task == "classification":
        response = np.asarray([index % 2 for index in range(40)], dtype=np.int64)
    else:
        response = rng.normal(size=40).astype(np.float32)
    dataset = RealDataset("toy", task, features, response, "unit-test")
    split = prepare_real_icl_split(dataset, split_seed=1729)
    evidence, _ = build_real_icl_episode(
        split,
        context_size=len(split.train_indices),
        query_indices=split.query_indices,
        shuffled_context=False,
    )
    return split, evidence


def test_full_context_manifest_is_strictly_bound() -> None:
    manifest = load_query_openml_full_context_panel_manifest(
        Path("experiments/transfer-query-v2/openml-full-context-2026-08-31.yaml")
    )
    assert manifest["payload"]["evaluation_design"]["query_policy"] == "all_heldout_rows"
    assert manifest["payload"]["evaluation_design"]["context_policy"] == "full_train"


def test_query_response_readout_matches_dense_terminal_for_full_context() -> None:
    for task in ("classification", "regression"):
        split, evidence = _toy_episode(task)
        model = build_model(
            "tabu.query.row",
            config=ReferenceConfig(
                d_model=8,
                n_heads=2,
                d_ff=16,
                n_blocks=1,
                inducing_slots=2,
                matched_slots=4,
                max_features=256,
            ),
            profile="supervised.label_broadcast.v1",
            row_token_count=4,
        ).eval()
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
            np.testing.assert_allclose(
                predicted[0],
                raw[context_rows:, -1].numpy(),
                rtol=1.0e-5,
                atol=1.0e-5,
            )
