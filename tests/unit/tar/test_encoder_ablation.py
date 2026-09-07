from dataclasses import replace

import pytest
import torch

from tabu_lab.models.tar import TabUTARModel, TARConfig, TARTrainer, TARTrainingConfig
from tabu_lab.models.tar.checkpoint import load_checkpoint, save_checkpoint
from tabu_lab.models.tar.verification import mixed_fixture
from tabu_lab.tar_encoder_monitor import EncoderMonitor


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def cfg():
    return TARConfig(width=128, heads=4, blocks=1, ff_width=32, semantic_slots=2,
                     inducing_slots=2, value_encoding='unified_constant_weight')


@pytest.mark.parametrize('initialization', ['design', 'identity'])
def test_frozen_projection_is_unchanged_and_checkpoint_preserves_freeze(initialization, tmp_path):
    config = replace(cfg(), encoder_initialization=initialization)
    trainable = TabUTARModel(config)
    frozen = TabUTARModel(replace(config, encoder_trainable=False))
    for name, weight in trainable.state_dict().items():
        assert torch.equal(weight, frozen.state_dict()[name])
    before = frozen.W_enc.detach().clone()
    if initialization == 'identity':
        assert torch.equal(before, torch.eye(128))
    else:
        torch.testing.assert_close(torch.linalg.svdvals(before), torch.full((128,), .125))
    item = mixed_fixture()
    monitor = EncoderMonitor(frozen, item[0])
    trainer = TARTrainer(frozen, TARTrainingConfig(effective_episode_batch=1,
                                                 optimizer_steps=2, warmup_steps=0))
    trainer.train_step([item])
    assert torch.equal(before, frozen.W_enc)
    assert frozen.W_enc.grad is None and monitor.gradient_norm is None
    assert monitor.measure()['drift_frobenius'] == 0
    assert frozen.frequencies.requires_grad
    save_checkpoint(frozen, tmp_path/'checkpoint', trainer=trainer)
    restored, restored_trainer = load_checkpoint(tmp_path/'checkpoint', restore_trainer=True)
    restored_trainer.train_step([item])
    assert not restored.W_enc.requires_grad
    assert torch.equal(restored.W_enc, before)
    monitor.close()


def test_monitor_preserves_outputs_rng_and_measures_preclip_gradient():
    model = TabUTARModel(cfg())
    ep, _ = mixed_fixture()
    initial = model.compile_episode(ep)[0].detach().clone()
    rng = torch.get_rng_state()
    monitor = EncoderMonitor(model, ep)
    measured = monitor.measure()
    assert torch.equal(rng, torch.get_rng_state())
    assert measured['relative_drift'] == measured['visible_carrier_relative_drift'] == 0
    assert torch.equal(initial, model.compile_episode(ep)[0])
    (model.W_enc * 1000).sum().backward()
    expected = float(model.W_enc.grad.double().norm())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert monitor.gradient_norm == expected
    assert monitor.gradient_norm > float(model.W_enc.grad.norm()) * 100
    monitor.close()


def test_only_projection_changes_for_identity_initialization():
    design = TabUTARModel(cfg())
    identity = TabUTARModel(replace(cfg(), encoder_initialization='identity'))
    for name, value in design.state_dict().items():
        if name != 'continuous':
            assert torch.equal(value, identity.state_dict()[name])
    with pytest.raises(ValueError, match='ablations require'):
        TARConfig(encoder_trainable=False)
