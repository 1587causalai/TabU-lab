"""Check episode changes, truth isolation, codebooks and deterministic replay."""

from dataclasses import replace

import pytest
import torch

from tabu_lab.models.tar import TabUTARModel, TARConfig, TARFeature
from tabu_lab.models.tar.episodes import episode_seed, sample_supervised_episode
from tabu_lab.models.tar.inference import predict_supervised


@pytest.fixture
def model():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield TabUTARModel(
        TARConfig(width=16, heads=4, ff_width=32, blocks=2, semantic_slots=3, inducing_slots=8)
    )
    torch.set_num_threads(old)


def fixture(index=0, namespace="training", values=None):
    if values is None:
        values = torch.tensor([[r, r % 2, r % 2] for r in range(8)], dtype=torch.float64)
    features = (
        TARFeature(column_id=0),
        TARFeature("nominal", ("a", "b"), 1),
        TARFeature("nominal", ("no", "yes"), 2),
    )
    return sample_supervised_episode(
        values,
        features,
        row_ids=list(range(100, 108)),
        context_size=6,
        seed=73,
        namespace=namespace,
        episode_id=index,
    )


def test_fresh_roles_and_actual_codebooks_with_replay(model):
    rng = torch.get_rng_state().clone()
    a, ta, ra = fixture(0)
    b, _, rb = fixture(1)
    again, truth, replay = fixture(0)
    assert ra == replay and ta == truth
    assert ra["context_row_ids"] != rb["context_row_ids"]
    assert ra["codebook_seed"] != rb["codebook_seed"]
    assert torch.equal(a.values, again.values)
    ca, cb = model.compile_episode(a), model.compile_episode(b)
    for column in (1, 2):
        assert not torch.equal(ca[4][column], cb[4][column])
        torch.testing.assert_close(ca[4][column].norm(dim=-1), torch.ones(2))
        # The same visible category has exactly the same vector inside an episode.
        ids = ((a.values[:, column] == 0) & a.visible[:, column]).nonzero().flatten()
        assert len(ids) >= 2
        torch.testing.assert_close(ca[0][ids[0], column], ca[0][ids[1], column], rtol=0, atol=0)
    # Freeze row roles: book changes are independently capable of changing vectors.
    different_book = model.compile_episode(replace(a, codebook_seed=b.codebook_seed))
    assert not torch.equal(ca[4][1], different_book[4][1])
    assert torch.equal(rng, torch.get_rng_state())


def test_query_truth_is_not_used_for_sampling_or_encoding(model):
    ep, truth, record = fixture()
    raw = torch.tensor([[r, r % 2, r % 2] for r in range(8)], dtype=torch.float64)
    for global_id in record["query_row_ids"]:
        raw[global_id - 100, -1] = 999999
    changed, new_truth, new_record = fixture(values=raw)
    assert new_record == record and new_truth != truth
    assert torch.equal(ep.values, changed.values)
    assert torch.count_nonzero(changed.values[changed.queries]) == 0
    a, b = model.compile_episode(ep), model.compile_episode(changed)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    assert set(record["context_row_ids"]).isdisjoint(record["query_row_ids"])


def test_stream_and_episode_namespaces_are_independent():
    seeds = {
        episode_seed(73, ns, i, stream)
        for ns in ("train", "fit-eval", "held-eval")
        for i in range(100)
        for stream in ("row_roles", "codebook")
    }
    assert len(seeds) == 600
    # Sampling is stateless: starting at a later episode needs no earlier RNG draws.
    _, _, expected = fixture(19)
    fixture(50)
    assert fixture(19)[2] == expected
    assert fixture(19, "fit-evaluation")[2]["codebook_seed"] != expected["codebook_seed"]


def test_inference_has_independent_books_and_stable_ids(model, monkeypatch):
    ep, _, _ = fixture()
    original = model.forward
    seen = []

    def record(item):
        out = original(item)
        seen.append((item.codebook_seed, out.codebooks[1].clone()))
        return out

    monkeypatch.setattr(model, "forward", record)
    train, test = ep.values[:6], ep.values[6:]
    vm, tm = torch.ones_like(train, dtype=torch.bool), torch.ones_like(test, dtype=torch.bool)
    first = predict_supervised(
        model, train, vm, test, tm, ep.features, codebook_seed=9, episode_ids=[91, 92]
    )
    second = predict_supervised(
        model, train, vm, test.flip(0), tm, ep.features, codebook_seed=9, episode_ids=[92, 91]
    )
    assert seen[0][0] != seen[1][0]
    assert not torch.equal(seen[0][1], seen[1][1])
    assert seen[0][0] == seen[3][0]
    torch.testing.assert_close(first[0].probabilities, second[1].probabilities, rtol=0, atol=0)
    with pytest.raises(ValueError, match="unique"):
        predict_supervised(model, train, vm, test, tm, ep.features, episode_ids=[1, 1])


def test_joint_prediction_uses_all_train_and_test_in_one_forward(model, monkeypatch):
    from tabu_lab.models.tar import predict_joint_supervised

    ep, _, _ = fixture()
    train, test = ep.values[:6], ep.values[6:].clone()
    vm, tm = torch.ones_like(train, dtype=torch.bool), torch.ones_like(test, dtype=torch.bool)
    seen = []
    forward = model.forward

    def capture(item):
        seen.append(item)
        return forward(item)

    monkeypatch.setattr(model, "forward", capture)
    first = predict_joint_supervised(
        model, train, vm, test, tm, ep.features, codebook_seed=9, episode_id=4
    )
    assert len(seen) == 1 and len(first) == len(test)
    assert seen[0].values.shape[0] == len(train) + len(test)
    assert seen[0].visible[: len(train), -1].all()
    assert not seen[0].visible[len(train) :, -1].any()
    assert seen[0].queries[len(train) :, -1].all()
    assert not seen[0].values[len(train) :, -1].any()
    test[:, -1] = 987654321
    second = predict_joint_supervised(
        model, train, vm, test, tm, ep.features, codebook_seed=9, episode_id=4
    )
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a.probabilities, b.probabilities, rtol=0, atol=0)
    assert torch.equal(seen[0].values, seen[1].values)
    assert model.training
    vm[0, -1] = False
    with pytest.raises(ValueError, match="all training labels"):
        predict_joint_supervised(model, train, vm, test, tm, ep.features)
