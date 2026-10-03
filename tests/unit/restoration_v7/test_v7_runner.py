"""V7 update-step reduction, strict checkpoints and frozen trajectory reports."""

from __future__ import annotations

import copy

import pytest
import torch

from tabu_lab.models.restoration.contracts import ColumnSchema, make_episode
from tabu_lab.models.restoration_v7 import (
    V7Config,
    V7Model,
    V7Task,
    checkpoint_state,
    evaluate_task,
    load_checkpoint,
    make_optimizer,
    prepare_episode,
    reference_values,
    save_checkpoint,
    score_rounds,
    state_loss,
    train_step,
)

SCHEMA = (
    ColumnSchema("x", "numeric"),
    ColumnSchema("c", "nominal", domain_size=4),
    ColumnSchema("y", "numeric"),
)
MANIFEST = {"data": "synthetic-unit-test", "split": "rows 8..", "donor_bank": [0, 1]}


@pytest.fixture(autouse=True)
def deterministic_cpu():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        yield


def small_config(**kwargs) -> V7Config:
    values = dict(
        backbone=dict(width=64, layers=1, heads=2, ff_width=64, slots=4),
        rounds=3,
        coupling_hidden=(32, 32),
        center_chunk_size=5,
    )
    values.update(kwargs)
    return V7Config(**values)


def task(seed: int, *, target: int = 2, n: int = 12, donor_seed: int = 0, **overrides) -> V7Task:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=generator, dtype=torch.float64)
    c = torch.randint(3, (n,), generator=generator)
    y = 2 * x + 0.1 * torch.randn(n, generator=generator, dtype=torch.float64)
    values = dict(x=x, c=c, y=y) | overrides
    observed = torch.ones(n, len(SCHEMA), dtype=torch.bool)
    query = torch.zeros_like(observed)
    query[8:, target] = True
    inputs, _, truth = make_episode(
        SCHEMA, tuple(values[s.key] for s in SCHEMA), observed, query, code_seed=seed
    )
    return V7Task(inputs, truth, donor_seed)


def episode_loss(model: V7Model, item: V7Task) -> torch.Tensor:
    episode = prepare_episode(item.inputs, donor_seed=item.donor_seed)
    output = model(episode, decode=False)
    return score_rounds(output, episode, reference_values(episode, item.truth), model.config).loss


def test_step_loss_is_task_equal_mean_and_one_update_follows_the_full_backward():
    model = V7Model(small_config())
    reference = copy.deepcopy(model)
    tasks = [task(1), task(2, n=20)]
    expected = torch.stack([episode_loss(reference, item) for item in tasks]).mean()
    expected.backward()
    optimizer = make_optimizer(model)
    record = train_step(model, optimizer, tasks)
    assert record.loss == pytest.approx(float(expected), rel=1e-12)
    assert len(record.round_losses) == 2 and all(len(r) == 3 for r in record.round_losses)
    for (name, p), q in zip(model.named_parameters(), reference.parameters(), strict=True):
        if q.grad is None:
            assert p.grad is None or not bool(p.grad.any()), name
        else:
            assert torch.allclose(p.grad, q.grad, rtol=1e-10, atol=1e-14), name
    moved = [
        not torch.equal(p, q)
        for p, q in zip(model.parameters(), reference.parameters(), strict=True)
    ]
    assert any(moved)


def test_optimizer_defaults_follow_the_example_spec():
    group = make_optimizer(V7Model(small_config())).param_groups[0]
    assert (group["lr"], group["betas"], group["eps"], group["weight_decay"]) == (
        1e-4,
        (0.9, 0.999),
        1e-8,
        0.0,
    )


def test_empty_step_is_rejected():
    model = V7Model(small_config())
    with pytest.raises(ValueError, match="at least one episode"):
        train_step(model, make_optimizer(model), [])


def test_checkpoint_resume_reproduces_the_next_update_exactly(tmp_path):
    model = V7Model(small_config())
    optimizer = make_optimizer(model)
    train_step(model, optimizer, [task(1)])
    path = tmp_path / "step1.pt"
    save_checkpoint(path, checkpoint_state(model, optimizer, step=1, manifest=MANIFEST))
    with pytest.raises(FileExistsError):
        save_checkpoint(path, checkpoint_state(model, optimizer, step=1, manifest=MANIFEST))

    resumed = V7Model(small_config())
    resumed_optimizer = make_optimizer(resumed)
    assert load_checkpoint(path, resumed, resumed_optimizer, manifest=MANIFEST) == 1
    first = train_step(model, optimizer, [task(2)])
    second = train_step(resumed, resumed_optimizer, [task(2)])
    assert first.loss == second.loss
    for p, q in zip(model.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(p, q)


def test_checkpoint_rejects_manifest_and_model_spec_drift(tmp_path):
    model = V7Model(small_config())
    optimizer = make_optimizer(model)
    path = tmp_path / "step0.pt"
    save_checkpoint(path, checkpoint_state(model, optimizer, step=0, manifest=MANIFEST))
    with pytest.raises(ValueError, match="manifest"):
        load_checkpoint(path, V7Model(small_config()), manifest=MANIFEST | {"split": "other"})
    with pytest.raises(ValueError, match="sharing mode"):
        load_checkpoint(path, V7Model(small_config(share_rounds=False)), manifest=MANIFEST)


def test_checkpoint_capture_survives_later_updates_and_manifest_mutation(tmp_path):
    model = V7Model(small_config())
    optimizer = make_optimizer(model)
    train_step(model, optimizer, [task(1)])
    manifest = copy.deepcopy(MANIFEST)
    state = checkpoint_state(model, optimizer, step=1, manifest=manifest)
    expected = copy.deepcopy(state)
    next_record = train_step(model, optimizer, [task(2)])
    manifest["donor_bank"].append(999)
    path = tmp_path / "delayed-step1.pt"
    save_checkpoint(path, state)

    resumed = V7Model(small_config())
    resumed_optimizer = make_optimizer(resumed)
    assert load_checkpoint(path, resumed, resumed_optimizer, manifest=MANIFEST) == 1
    assert state["manifest"] == MANIFEST
    assert torch.equal(torch.get_rng_state(), expected["torch_cpu_rng"])
    for key, tensor in resumed.state_dict().items():
        assert torch.equal(tensor, expected["model"][key]), key
    restored = resumed_optimizer.state_dict()
    assert restored["param_groups"] == expected["optimizer"]["param_groups"]
    for key, values in restored["state"].items():
        for name, tensor in values.items():
            assert torch.equal(tensor, expected["optimizer"]["state"][key][name])
    record = train_step(resumed, resumed_optimizer, [task(2)])
    assert record.loss == next_record.loss
    for p, q in zip(model.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(p, q)


def test_checkpoint_rejects_inconsistent_stored_manifest_before_loading_weights(tmp_path):
    model = V7Model(small_config())
    state = checkpoint_state(model, make_optimizer(model), step=0, manifest=MANIFEST)
    state["manifest"]["split"] = "changed without updating its digest"
    path = tmp_path / "inconsistent.pt"
    save_checkpoint(path, state)
    resumed = V7Model(small_config())
    before = copy.deepcopy(resumed.state_dict())
    with pytest.raises(ValueError, match="stored manifest"):
        load_checkpoint(path, resumed, manifest=MANIFEST)
    for key, tensor in resumed.state_dict().items():
        assert torch.equal(tensor, before[key]), key


def test_trajectory_reports_every_round_and_matches_the_frozen_forward():
    model = V7Model(small_config())
    item = task(3)
    report = evaluate_task(model, item)
    episode = prepare_episode(item.inputs, donor_seed=item.donor_seed, admission="inference")
    with torch.no_grad():
        output = model(episode)
    assert report.status == "ok" and report.kind == "numeric"
    assert len(report.code_losses) == 4 and len(report.state_changes) == 3
    assert len(report.predictions) == 3
    assert torch.equal(report.predictions[-1], output.decoded)
    column = episode.codec.columns[episode.target]
    truth_codes = column.encode(reference_values(episode, item.truth)).to(output.initial)
    seed_loss = float(state_loss(output.initial, truth_codes, model.config.chi_numeric))
    donor_codes = episode.observed[episode.donor_rows, episode.target].to(truth_codes)
    donor_loss = float(state_loss(donor_codes, truth_codes, model.config.chi_numeric))
    assert torch.equal(
        output.initial,
        model.query_seed("numeric").detach().to(output.initial).expand_as(output.initial),
    )
    assert report.code_losses[0] == pytest.approx(seed_loss)
    assert report.baselines["donor"]["code_loss"] == pytest.approx(donor_loss)
    assert torch.allclose(report.donor, column.decode(donor_codes))
    assert set(report.metrics) == {"mae", "rmse", "standardized_rmse"}
    assert set(report.baselines["support_marginal"]) == {
        "mae",
        "rmse",
        "standardized_rmse",
        "code_mean_loss",
    }


def test_trajectory_predictions_do_not_read_hidden_truth():
    model = V7Model(small_config())
    item = task(4)
    altered = list(item.truth.values)
    altered[2] = altered[2].clone()
    altered[2][8:] += 100.0
    changed = V7Task(item.inputs, type(item.truth)(tuple(altered), item.truth.states), 0)
    first, second = evaluate_task(model, item), evaluate_task(model, changed)
    for a, b in zip(first.predictions, second.predictions, strict=True):
        assert torch.equal(a, b)
    assert first.metrics != second.metrics


def test_unseen_hidden_category_is_reported_not_dropped():
    c = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 0, 1])
    report = evaluate_task(V7Model(small_config()), task(5, target=1, c=c))
    assert report.status == "no-answer-code" and report.code_losses is None
    assert 0.0 <= report.metrics["accuracy"] <= 0.5
    assert "code_loss" not in report.baselines["donor"]


def test_short_synthetic_training_run_stays_finite():
    model = V7Model(small_config())
    optimizer = make_optimizer(model)
    records = [train_step(model, optimizer, [task(100 + step)]) for step in range(5)]
    assert all(torch.isfinite(torch.tensor(r.loss)) for r in records)
    assert evaluate_task(model, task(999)).status == "ok"
