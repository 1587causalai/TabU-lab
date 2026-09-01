from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import tabu_lab.experiments.query_row_supervised_synthetic_v3_categorical as categorical_module
from tabu_lab.contracts import FeatureKind
from tabu_lab.experiments.query_row_supervised_synthetic_v3_capped import (
    make_query_row_supervised_synthetic_v3_capped_episode,
)
from tabu_lab.models import ReferenceConfig, build_model
from tabu_lab.training import Objective

EPISODE_OPTIONS = {
    "root_seed": 2718,
    "world_id": "categorical-response-contract",
    "width": 8,
    "rows": 32,
    "context_rows": 16,
    "family": "sparse_additive",
    "predictor_regime": "gaussian",
    "noise_level": "low",
}


def test_categorical_response_generator_is_typed_masked_and_deterministic() -> None:
    episode = categorical_module.make_query_row_supervised_synthetic_v3_categorical_episode(
        **EPISODE_OPTIONS
    )
    replay = categorical_module.make_query_row_supervised_synthetic_v3_categorical_episode(
        **EPISODE_OPTIONS
    )

    response = episode.evidence.feature_specs[-1]
    assert response.kind is FeatureKind.CATEGORICAL
    assert 2 <= len(response.domain) <= 8
    assert response.codebook_id is not None
    assert episode.evidence.metadata["response_schema_source"] == "visible_context_only"
    assert episode.evidence.evidence_hash == replay.evidence.evidence_hash
    assert episode.sidecar.truth_hash == replay.sidecar.truth_hash
    assert bool((episode.evidence.forward_values[episode.sidecar.target_mask] == 0).all())
    codes = episode.sidecar.target_values[episode.sidecar.target_mask]
    assert bool((codes == codes.round()).all())
    assert int(codes.min()) >= 0
    assert int(codes.max()) < len(response.domain)
    context_codes = episode.evidence.forward_values[: episode.context_rows, -1]
    assert int(torch.unique(context_codes).numel()) == len(response.domain)


def test_query_truth_substitution_cannot_change_categorical_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = make_query_row_supervised_synthetic_v3_capped_episode(**EPISODE_OPTIONS)
    poisoned_values = parent.sidecar.target_values.clone()
    poisoned_values[parent.sidecar.target_mask] += 10_000.0
    poisoned_parent = replace(
        parent,
        sidecar=replace(parent.sidecar, target_values=poisoned_values),
    )

    clean = categorical_module.make_query_row_supervised_synthetic_v3_categorical_episode(
        **EPISODE_OPTIONS
    )
    monkeypatch.setattr(
        categorical_module,
        "make_query_row_supervised_synthetic_v3_capped_episode",
        lambda **_kwargs: poisoned_parent,
    )
    poisoned = categorical_module.make_query_row_supervised_synthetic_v3_categorical_episode(
        **EPISODE_OPTIONS
    )

    assert clean.evidence.evidence_hash == poisoned.evidence.evidence_hash
    assert torch.equal(clean.evidence.forward_values, poisoned.evidence.forward_values)
    assert not torch.equal(clean.sidecar.target_values, poisoned.sidecar.target_values)


@pytest.mark.parametrize(
    ("model_id", "model_options"),
    (
        ("tabu.query.base", {}),
        (
            "tabu.query.row",
            {
                "row_token_count": 4,
                "row_readout_mode": "anchored",
                "anchored_gamma_initial": 0.01,
            },
        ),
    ),
)
def test_categorical_response_activates_nll_and_backpropagates(
    model_id: str,
    model_options: dict[str, object],
) -> None:
    episode = categorical_module.make_query_row_supervised_synthetic_v3_categorical_episode(
        **EPISODE_OPTIONS
    )
    torch.manual_seed(1729)
    model = build_model(
        model_id,
        config=ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=4,
            max_features=64,
            dropout=0.0,
        ),
        profile="supervised.label_broadcast.v1",
        **model_options,
    )
    prediction = model(episode.evidence)
    loss = Objective(
        numeric_target_coordinate="context_standardized",
        include_categorical=True,
    )(prediction, episode.sidecar, evidence=episode.evidence)

    assert loss.counts["numeric_scored_targets"] == 0
    assert loss.counts["categorical_scored_targets"] == episode.sidecar.target_count
    assert loss.components["categorical_nll"].item() > 0.0
    assert torch.isfinite(loss.components["categorical_normalized_nll"])
    assert torch.isfinite(loss.components["categorical_context_prior_nll"])
    assert torch.isfinite(loss.components["categorical_skill_vs_context_prior"])
    assert 0.0 <= loss.components["categorical_balanced_accuracy"].item() <= 1.0
    assert loss.metadata["label_active_types"] == ("categorical",)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.parameters()
    )
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
