"""64D correction retains old interfaces and exact baseline random directions."""
import pytest
import torch

from tabu_lab.models.completion_v8 import (
    V8CompletionModel,
    V8Config,
    from_v8,
    load_weights,
    prepare_completion,
    save_weights,
)

from . import test_feature_codebook as shared
from .test_completion import runtime as runtime
from .test_completion import spec

MODE = "learned_feature64_v1"


def config(**kwargs):
    return V8Config(spec(**kwargs), codebook_mode=MODE)


def test_every_donor_weight_is_copied_only_bank_is_new(tmp_path):
    donor = V8CompletionModel(V8Config(spec())).double()
    model = from_v8(donor)
    assert model.config.codebook_mode == MODE and model.config.code_dim == 64
    assert model.feature_codebook.weight.shape == (10000, 64)
    assert model.feature_codebook.weight.numel() == 640000
    torch.testing.assert_close(
        model.feature_codebook.weight.norm(dim=-1), torch.ones(10000).double()
    )
    assert model.migration_receipt['new_keys'] == ['feature_codebook.weight']
    assert set(model.migration_receipt['copied_keys']) == set(donor.state_dict())
    for key, value in donor.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    assert model.query_seed.shape == (64,) and model.lift.weight.shape == (128, 64)
    assert model.encoder.blocks[0].attention.v.weight.shape == (32, 32)
    inputs, _ = shared.mixed(dtype=torch.float64)
    old_ep = prepare_completion(inputs, donor.config)
    ep = prepare_completion(inputs, model.config)
    # If feature lookup returns the old bases, the donor computation is retained.
    assert len(set(ep.codec.indices)) == len(ep.codec.indices)
    with torch.no_grad():
        for index, column in zip(ep.codec.indices, old_ep.codec.columns, strict=True):
            model.feature_codebook.weight[index].copy_(column.base)
    torch.testing.assert_close(donor(old_ep).completed_codes, model(ep).completed_codes,
                               rtol=0, atol=0)
    path = tmp_path / 'feature64.pt'
    save_weights(model, path)
    restored = load_weights(path)
    assert restored.config.codebook_mode == MODE
    assert torch.equal(model(ep).completed_codes,
                       restored(prepare_completion(inputs, restored.config)).completed_codes)


@pytest.mark.parametrize('family', ['C64/8', 'G64'])
def test_directions_statistics_match_unchanged_v8_codec(family):
    inputs, _ = shared.mixed(dtype=torch.float64)
    old = prepare_completion(inputs, V8Config(spec(codec=family))).codec
    new = prepare_completion(inputs, config(codec=family)).codec.template
    assert new.dim == 64 and new.numeric_preprocessing == old.numeric_preprocessing
    for a, b in zip(old.columns, new.columns, strict=True):
        if a.direction is not None:
            assert torch.equal(a.direction, b.direction) and not b.direction.requires_grad
        if a.codes is not None:
            assert torch.equal(a.codes - a.base, b.codes) and not b.codes.requires_grad
        for key in ['mean', 'scale', 'numeric_encoding']:
            if hasattr(a, key):
                assert getattr(a, key) == getattr(b, key)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_two_steps_and_partial_codebook_lifecycle(monkeypatch, device):
    monkeypatch.setattr(shared, 'config', config)
    shared.test_two_optimizer_steps_refresh_bank_without_stale_graph(device)
    shared.test_partial_recoding_keeps_reference_indices_and_random_directions()
    shared.test_hidden_truth_isolation_and_same_forward_reference()


def test_feature64_gradient_matches_finite_difference(monkeypatch):
    monkeypatch.setattr(shared, 'config', config)
    shared.test_feature_gradient_matches_finite_difference_with_frozen_reference()
