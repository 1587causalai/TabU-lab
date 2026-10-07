"""Default asinh coordinates: discrete parity, extreme values, and full training."""
from dataclasses import replace
import pytest
import torch
from tabu_lab.models.restoration_v7 import (
    V7Config, V7Model, make_optimizer, prepare_episode, prepare_joint_episode,
    score_joint, score_rounds, reference_values, checkpoint_state, save_checkpoint, load_checkpoint,
)
from tabu_lab.models.restoration_v7.codec import build_value_codec, select_softlog
from test_v7_numeric_zero import inputs, isolated_mps_algorithm_mode
from test_v7_joint import task, config

MODE = 'standard_asinh_v1'

@pytest.mark.parametrize('family', ['C64/8', 'G64'])
def test_roundtrip_extremes_no_query_switch_or_hidden_truth(family):
    codec = build_value_codec(inputs([-.005,.005,0.]), codec=family, numeric_preprocessing=MODE)
    c = codec.columns[0]
    assert c.mean == 0 and c.scale == .01 and c.numeric_encoding == 'asinh'
    values = torch.tensor([0.,-1e-5,1e-5,-1.,1.,-6425.6,6425.6,-1e308,1e308],dtype=torch.float64)
    encoded = c.encode(values)
    assert torch.isfinite(encoded).all()
    torch.testing.assert_close(c.decode(encoded), values, rtol=2e-12, atol=1e-16)
    assert torch.equal(c.encode(torch.zeros(1,dtype=torch.float64))[0],c.base)
    assert select_softlog(codec, inputs([-.005,.005,1e308], hidden=1e308)) is codec
    with pytest.raises(FloatingPointError, match='decoded values'):
        c.decode((c.base + 1000*c.direction)[None])

@pytest.mark.parametrize('sign', [-1.,1.])
def test_extreme_mean_cancellation(sign):
    c = build_value_codec(inputs([sign*1e308,sign*1.000000001e308]), numeric_preprocessing=MODE).columns[0]
    values = torch.tensor([sign*1e308, -sign*1e308],dtype=torch.float64)
    torch.testing.assert_close(c.decode(c.encode(values)),values,rtol=2e-12,atol=0.)


def test_random_bases_categories_null_and_hidden_truth():
    item = task(single=True)
    old = build_value_codec(item.inputs, numeric_preprocessing='standard_softlog_v1')
    new = build_value_codec(item.inputs, numeric_preprocessing=MODE)
    for a,b in zip(old.columns,new.columns,strict=True):
        assert torch.equal(a.base,b.base)
        if a.direction is not None: assert torch.equal(a.direction,b.direction)
        if a.codes is not None:
            assert torch.equal(a.codes,b.codes) and torch.equal(a.categories,b.categories)
    poisoned = build_value_codec(task(single=True,changed_truth=True).inputs,numeric_preprocessing=MODE)
    assert torch.equal(new.encode_observed(item.inputs),poisoned.encode_observed(item.inputs))
    ep = prepare_episode(inputs([5.]*5),donor_seed=3,numeric_preprocessing=MODE)
    assert not ep.inputs.visible[:,0].any() and ep.observed[:,0].count_nonzero()==0


@pytest.mark.parametrize('device,dtype', [('cpu',torch.float64),('cpu',torch.float32),('mps',torch.float32)])
@pytest.mark.parametrize('single', [True,False])
def test_k4_train_and_joint(device,dtype,single):
    if device=='mps' and not torch.backends.mps.is_available(): pytest.skip('MPS unavailable')
    torch.manual_seed(21)
    cfg = config(model_version='v7.3', numeric_preprocessing=MODE, rounds=4)
    model = V7Model(cfg).to(device=device,dtype=dtype)
    item = task(single=single,device=device,dtype=dtype)
    joint = prepare_joint_episode(item.inputs,donor_seed=4,config=cfg)
    output = model(joint,decode=False)
    score = score_joint(output,joint,item.truth,cfg)
    assert torch.isfinite(score.loss)
    score.loss.backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    make_optimizer(model).step()
    if single:
        ep = prepare_episode(item.inputs,donor_seed=4,numeric_preprocessing=MODE)
        torch.testing.assert_close(ep.observed,joint.observed)
        model.zero_grad(set_to_none=True)
        loss=score_rounds(model(ep,decode=False),ep,reference_values(ep,item.truth),cfg).loss
        loss.backward()
        assert torch.isfinite(loss)
    c=joint.codec.columns[0]
    vals=torch.tensor([0.,.001,-1000.,1000.],device=device,dtype=dtype)
    torch.testing.assert_close(c.decode(c.encode(vals)),vals.to(c.base),rtol=2e-5 if dtype==torch.float32 else 1e-12,atol=1e-6)


def test_config_checkpoint_protocol_boundary(tmp_path):
    old=V7Config.v73(numeric_preprocessing='standard_softlog_v1',backbone=dict(width=64,layers=1,heads=2,ff_width=64,slots=3),coupling_hidden=(16,))
    assert old.numeric_preprocessing=='standard_softlog_v1'
    cfg=V7Config.v73(backbone=old.backbone,coupling_hidden=old.coupling_hidden)
    assert cfg.numeric_preprocessing == MODE
    assert V7Config.from_dict(cfg.as_dict()) == cfg
    model=V7Model(old).double();opt=make_optimizer(model)
    path=tmp_path/'old.pt';save_checkpoint(path,checkpoint_state(model,opt,step=0,manifest={}))
    new=V7Model(cfg).double()
    new.load_state_dict(model.state_dict(),strict=True)
    restored_old=V7Model(V7Config.from_dict(old.as_dict())).double()
    load_checkpoint(path,restored_old,make_optimizer(restored_old),manifest={})
    assert restored_old.config.numeric_preprocessing == 'standard_softlog_v1'
    new_path=tmp_path/'asinh.pt'
    save_checkpoint(new_path,checkpoint_state(new,make_optimizer(new),step=0,manifest={}))
    restored_new=V7Model(V7Config.from_dict(cfg.as_dict())).double()
    load_checkpoint(new_path,restored_new,make_optimizer(restored_new),manifest={})
    assert restored_new.config.numeric_preprocessing == MODE
    for key,value in new.state_dict().items():
        torch.testing.assert_close(restored_new.state_dict()[key],value,rtol=0,atol=0)
    with pytest.raises(ValueError,match='ModelSpec'):
        load_checkpoint(path,new,make_optimizer(new),manifest={})
    with pytest.raises(ValueError,match='requires C64/8 or G64'):
        replace(cfg,codec='constant_weight_composition_v2',code_dim=128)
