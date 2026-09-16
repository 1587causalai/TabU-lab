"""Explicit state weights change optimization, never the reported raw errors."""

from pathlib import Path

import pytest
import torch
import yaml

from tabu_lab import restoration_curriculum_fit as curriculum
from tabu_lab.models.restoration import (
    ColumnSchema,
    LossConfig,
    RestorationModel,
    make_episode,
    prepare_episode,
    score_episode,
    score_prepared_episode,
)
from tabu_lab.models.restoration.end_to_end_checks import small_config

OBJECTIVE = {"kind": "state_weighted", "retained_weight": .05, "query_weight": .95}


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        yield
    torch.set_num_threads(threads)


def _table():
    return curriculum.CurriculumTable(
        "mixed", "old120", curriculum.SYNTHETIC,
        (
            torch.tensor([0., 1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64),
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
            torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
        ),
        (ColumnSchema("x", "numeric"), ColumnSchema("c", "nominal", 2),
         ColumnSchema("o", "ordinal", 3)),
        tuple(range(8)), 2,
    )


def _episode():
    table = _table()
    query = torch.zeros((8, 3), dtype=torch.bool)
    # Unequal retained/query and numeric/discrete counts expose pooled weighting.
    query[-2:, 0] = True
    query[-1, 1] = True
    query[-3:, 2] = True
    return make_episode(table.schema, table.values, torch.ones_like(query), query, code_seed=17)


def _type_state_means(score, episode):
    targets, truth = episode[1].targets, episode[2]
    states = truth.states[targets[:, 0], targets[:, 1]]
    numeric = targets[:, 1] == 0
    return {
        state: sum(score.per_target[kind & (states == state)].mean()
                   for kind in (numeric, ~numeric))
        for state in (0, 1)
    }


def test_weighted_objective_uses_state_means_per_type_and_keeps_gradients():
    model = RestorationModel(small_config()).double()
    episode = _episode()
    config = curriculum._loss_config({"objective": OBJECTIVE})
    assert config.state_weights == (.05, .95, 0., 0.)
    weighted = score_episode(model, *episode, loss_config=config)
    means = _type_state_means(weighted, episode)
    torch.testing.assert_close(weighted.loss, .05 * means[0] + .95 * means[1])

    states = episode[2].states[episode[1].targets[:, 0], episode[1].targets[:, 1]]
    numeric = episode[1].targets[:, 1] == 0
    weights = torch.where(states == 0, .05, .95)
    pooled = sum((weighted.per_target[kind] * weights[kind]).mean()
                 for kind in (numeric, ~numeric))
    assert not torch.isclose(weighted.loss, pooled, rtol=1e-3, atol=1e-8)
    assert weighted.by_state["retained"]["count"] == 18
    assert weighted.by_state["query"]["count"] == 6
    weighted.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(value).all()) for value in gradients)
    assert sum(float(value.square().sum()) for value in gradients) > 0


def test_legacy_objective_remains_type_internal_observed_cell_mean():
    config = curriculum._loss_config({})
    assert config == LossConfig(None)
    model = RestorationModel(small_config()).double()
    episode = _episode()
    score = score_episode(model, *episode, loss_config=config)
    numeric = episode[1].targets[:, 1] == 0
    expected = score.per_target[numeric].mean() + score.per_target[~numeric].mean()
    torch.testing.assert_close(score.loss, expected)


@pytest.mark.parametrize("spec", [{}, {"objective": OBJECTIVE}])
def test_logged_loss_contributions_sum_to_optimized_loss(spec):
    model = RestorationModel(small_config()).double()
    episode = _episode()
    config = curriculum._loss_config(spec)
    prepared = prepare_episode(model, *episode)
    score = score_prepared_episode(model, prepared, loss_config=config)
    terms = curriculum._loss_terms(score, prepared, config)
    means = _type_state_means(score, episode)
    assert terms["retained_loss"] == pytest.approx(float(means[0].detach()), rel=1e-12)
    assert terms["query_loss"] == pytest.approx(float(means[1].detach()), rel=1e-12)
    assert terms["retained_loss_contribution"] + terms["query_loss_contribution"] == (
        pytest.approx(float(score.loss.detach()), rel=1e-12)
    )
    if config.state_weights is not None:
        assert terms["retained_loss_contribution"] == pytest.approx(.05 * terms["retained_loss"])
        assert terms["query_loss_contribution"] == pytest.approx(.95 * terms["query_loss"])


@pytest.mark.parametrize("objective", [
    None, [], "state_weighted", {},
    {"kind": "other", "retained_weight": .05, "query_weight": .95},
    {"kind": "state_weighted", "retained_weight": .05},
    {"kind": "state_weighted", "query_weight": .95},
    {"kind": "state_weighted", "retained_weight": 0., "query_weight": 1.},
    {"kind": "state_weighted", "retained_weight": -.05, "query_weight": 1.05},
    {"kind": "state_weighted", "retained_weight": .05, "query_weight": 0.},
    {"kind": "state_weighted", "retained_weight": float("nan"), "query_weight": .95},
    {"kind": "state_weighted", "retained_weight": .05, "query_weight": float("inf")},
    {"kind": "state_weighted", "retained_weight": True, "query_weight": .95},
    {"kind": "state_weighted", "retained_weight": ".05", "query_weight": .95},
    {"kind": "state_weighted", "retained_weight": .05, "query_weight": .90},
])
def test_explicit_invalid_objective_is_rejected(objective):
    with pytest.raises(ValueError):
        curriculum._loss_config({"objective": objective})


def test_evaluation_uses_weighted_loss_but_preserves_raw_metrics():
    config = small_config()
    model = RestorationModel(config).double()
    table = _table()
    stage = {"name": "old120", "mask": "random_cell", "mask_fraction": .25}
    seeds = {"masks": 19, "windows": 23}
    loss_config = curriculum._loss_config({"objective": OBJECTIVE})
    args = (model, [table], stage, seeds, config, "cpu")
    legacy = curriculum.evaluate(*args, masks=2)
    explicit_legacy = curriculum.evaluate(*args, masks=2, loss_config=LossConfig(None))
    weighted = curriculum.evaluate(*args, masks=2, loss_config=loss_config)
    assert legacy == explicit_legacy
    assert {key: value for key, value in weighted.items() if key != "loss"} == {
        key: value for key, value in legacy.items() if key != "loss"
    }
    expected = []
    with torch.no_grad():
        for index in range(2):
            episode, _ = curriculum._episode_for(
                table, index, stage, seeds, config, "cpu", evaluation=True,
            )
            score = score_episode(model, *episode, loss_config=loss_config)
            expected.append(float(score.loss))
    assert weighted["loss"] == pytest.approx(sum(expected) / len(expected), rel=1e-12)
    assert weighted["loss"] != pytest.approx(legacy["loss"])


def test_checkpoint_rejects_changed_objective_even_with_same_preregistration(tmp_path):
    prereg = tmp_path / "preregistration.yaml"
    prereg.write_text("frozen fixture\n")
    model = RestorationModel(small_config()).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    spec = {"model": model.config.as_dict()}
    identity = curriculum._identity(spec, prereg, [_table()], {"source": "fixture"})
    changed = curriculum._identity(
        spec | {"objective": OBJECTIVE}, prereg, [_table()], {"source": "fixture"},
    )
    assert identity != changed
    path = tmp_path / "checkpoint.pt"
    curriculum._checkpoint(
        path, model, optimizer, identity, 0, 0, 0, 0., stage_name="old120",
    )
    loaded = curriculum._load_checkpoint(path, model, optimizer, identity)
    assert loaded["identity"] == identity
    with pytest.raises(ValueError, match="identity drift"):
        curriculum._load_checkpoint(path, model, optimizer, changed)


def test_query95_recipe_is_distinct_and_historical_recipe_stays_legacy():
    experiments = Path(__file__).resolve().parents[3] / "experiments" / "local" / "restoration"
    legacy = yaml.safe_load(
        (experiments / "curriculum-small128" / "preregistration.yaml").read_text()
    )
    weighted = yaml.safe_load(
        (experiments / "curriculum-small128-query95" / "preregistration.yaml").read_text()
    )
    assert curriculum._loss_config(legacy).state_weights is None
    assert curriculum._loss_config(weighted).state_weights == (.05, .95, 0., 0.)
    assert weighted["experiment_id"] != legacy["experiment_id"]
