from dataclasses import replace

import pytest
import torch

from tabu_lab.models.tar import TabUTARModel, TARConfig, TAREpisode
from tabu_lab.models.tar.checkpoint import load_checkpoint, save_checkpoint
from tabu_lab.models.tar.encoding import constant_weight_codebook
from tabu_lab.models.tar.types import TARFeature


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(mode):
    return TARConfig(width=128, heads=4, blocks=1, ff_width=32,
                     semantic_slots=2, inducing_slots=2, value_encoding=mode)


def episode():
    values = torch.tensor([[1., 0., 0.], [2., 1., 2.], [3., 0., 0.], [9., 2., 1.]])
    visible = torch.ones_like(values, dtype=torch.bool)
    visible[-1] = False
    return TAREpisode.from_table(values, visible, ~visible, (
        TARFeature('numeric', column_id=0),
        TARFeature('nominal', ('a', 'b', 'hidden'), column_id=1),
        TARFeature('ordinal', ('low', 'middle', 'high'), column_id=2),
    ), codebook_seed=7)


def test_code_geometry_reproducibility_and_rng():
    state = torch.get_rng_state()
    b = constant_weight_codebook(range(100), seed=7, column_id=1)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(b, constant_weight_codebook(range(100), seed=7, column_id=1))
    assert not torch.equal(b, constant_weight_codebook(range(100), seed=8, column_id=1))
    assert not torch.equal(b, constant_weight_codebook(range(100), seed=7, column_id=2))
    assert torch.equal(b.sum(-1), torch.full((100,), 8., dtype=b.dtype))
    assert len(b.unique(dim=0)) == 100
    distances = torch.pdist(b).square()
    assert distances.min() >= 2 - 1e-10 and distances.max() <= 16 + 1e-10
    u = constant_weight_codebook(range(100), seed=7, column_id=1, unit_norm=True)
    torch.testing.assert_close(u.norm(dim=-1), torch.ones(100, dtype=u.dtype))


@pytest.mark.parametrize('mode', ['constant_weight', 'unified_constant_weight'])
def test_encoding_boundaries_gradients_and_checkpoint(mode, tmp_path):
    model = TabUTARModel(config(mode), dtype=torch.float64)
    ep = episode()
    h, _, null, _, books, classes = model.compile_episode(ep)
    assert classes[1] == (0, 1)  # hidden category never enters the book
    assert torch.equal(h[0, 1], h[2, 1])
    assert (h[null] == 0).all()
    assert sum(p.numel() for p in model.parameters()) == model.config.parameter_count
    replay = replace(ep, codebooks=books, codebook_classes=classes)
    torch.testing.assert_close(h, model.compile_episode(replay)[0])
    reordered = replace(ep, values=ep.values.flip(0), visible=ep.visible.flip(0),
                        queries=ep.queries.flip(0))
    torch.testing.assert_close(h[:4].flip(0), model.compile_episode(reordered)[0][:4])
    changed = replace(ep, codebook_seed=8)
    assert not torch.equal(books[1], model.compile_episode(changed)[4][1])
    if mode == 'unified_constant_weight':
        assert classes[2] == (0, 2)
        assert model.frequencies.numel() == 64
        torch.testing.assert_close(h[1, 2], (books[2][1] + 1) @ model.W_enc.T)
        # Each type backpropagates to the very same projection parameter.
        for col in range(3):
            model.zero_grad()
            model.compile_episode(ep)[0][:3, col].square().sum().backward()
            assert model.W_enc.grad.norm() > 0
    output = model(ep)
    output.responses.square().sum().backward()
    assert torch.isfinite(model.continuous.grad).all()
    save_checkpoint(model, tmp_path / 'model')
    restored = load_checkpoint(tmp_path / 'model', expected_config=model.config)
    torch.testing.assert_close(output.responses, restored(ep).responses)
    broken = books[1].clone()
    broken[1] = broken[0]
    with pytest.raises(ValueError, match='unique 8-of-128'):
        model.compile_episode(replace(replay, codebooks={**books, 1: broken}))


def test_unified_exact_lifts_rank_recovery_and_singleton():
    model = TabUTARModel(config('unified_constant_weight'), dtype=torch.float64)
    with torch.no_grad():
        model.W_enc.copy_(torch.eye(128))
    ep = episode()
    h, _, _, _, books, _ = model.compile_episode(ep)
    ranks = torch.tensor([0., 1., 0.], dtype=h.dtype)
    torch.testing.assert_close(h[:3, 2].mean(-1) - 8 / 128, ranks)
    torch.testing.assert_close(h[1, 2] - ranks[1], books[2][1])
    delta = h[0, 2] - h[1, 2]
    identity_distance = (books[2][0] - books[2][1]).square().sum()
    torch.testing.assert_close(delta.square().sum(), identity_distance + 128)
    torch.testing.assert_close(h[:3, 0].square().sum(-1), torch.full((3,), 64., dtype=h.dtype))
    singleton = TAREpisode.from_table(torch.zeros(2, 1), torch.ones(2, 1, dtype=torch.bool),
                                    torch.zeros(2, 1, dtype=torch.bool),
                                    (TARFeature('ordinal', ('only',)),))
    hs, _, _, _, bs, _ = model.compile_episode(singleton)
    torch.testing.assert_close(hs[0, 0], bs[0][0])


def test_nominal_only_preserves_other_parameters_and_carriers():
    legacy = TabUTARModel(config('legacy'))
    candidate = TabUTARModel(config('constant_weight'))
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name])
    a = legacy.compile_episode(episode())[0]
    b = candidate.compile_episode(episode())[0]
    torch.testing.assert_close(a[:, [0, 2, 3, 4]], b[:, [0, 2, 3, 4]])


def test_invalid_modes_and_width():
    with pytest.raises(ValueError, match='unknown value_encoding'):
        TARConfig(value_encoding='typo')
    with pytest.raises(ValueError, match='width=128'):
        TARConfig(value_encoding='constant_weight')
