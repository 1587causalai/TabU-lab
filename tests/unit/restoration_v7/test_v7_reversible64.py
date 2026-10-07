from dataclasses import replace
import hashlib
import pytest
import torch
from tabu_lab.models.restoration_v7 import *
from tabu_lab.models.restoration_v7 import RowReversible64Config, RowReversible64Encoder, row_reversible64_from_checkpoint
from test_v7_row_dual_stream import task, _episode, _score, _donor

@pytest.fixture(autouse=True)
def runtime():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(184)
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)

DEVICES=[('cpu',torch.float64,1e-10),('mps',torch.float32,3e-4)]

@pytest.mark.parametrize('device,dtype,tol',DEVICES)
def test_arbitrary_inverse_context_null_and_grad(device,dtype,tol):
    enc=RowReversible64Encoder(RowReversible64Config(),strict_finite_content=True).to(device=device,dtype=dtype)
    x=torch.randn(3,4,64,device=device,dtype=dtype)
    active=torch.ones(3,4,device=device,dtype=torch.bool);active[0,2]=False
    assert torch.equal(enc(x,active,active),x)
    # Nonidentity perturbation verifies genuine inverse, not just identity initialization.
    with torch.no_grad():
        for a,f in zip(enc.attention,enc.ffn):
            a.out.weight.normal_(std=.03);f.ff[2].weight.normal_(std=.03)
    x.requires_grad_(); y=enc(x,active,active); back=enc.inverse(y,active,active)
    assert torch.allclose(back,x,atol=tol,rtol=tol)
    assert torch.equal(y[~active],x[~active])
    arbitrary=torch.randn_like(x)
    assert torch.allclose(enc(enc.inverse(arbitrary,active,active),active,active),arbitrary,atol=tol,rtol=tol)
    changed=x.detach().clone(); changed[1,0]+=1
    other=enc(changed,active,active)
    assert torch.equal(y.detach()[0],other[0]) and not torch.equal(y.detach()[1,1],other[1,1])
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())
    if device=='cpu':
        probe=x.detach().clone().requires_grad_(); direction=torch.randn_like(probe); objective=(enc.inverse(probe,active,active)**2).sum()
        analytic=(torch.autograd.grad(objective,probe)[0]*direction).sum()
        eps=1e-5
        fd=((enc.inverse(probe+eps*direction,active,active)**2).sum()-(enc.inverse(probe-eps*direction,active,active)**2).sum())/(2*eps)
        assert torch.allclose(analytic,fd,atol=1e-5,rtol=1e-6)

@pytest.mark.parametrize('device,dtype,tol',DEVICES)
@pytest.mark.parametrize('single',[True,False])
@pytest.mark.parametrize('query_source',[True,False])
def test_migration_parity_backward_update_reload(tmp_path,device,dtype,tol,single,query_source):
    donor,path=_donor(tmp_path,query_source=query_source)
    model=row_reversible64_from_checkpoint(path,device=device,dtype=dtype,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    donor.to(device=device,dtype=dtype)
    item=task(single=single,device=device,dtype=dtype); ep=_episode(item,model.config,single)
    old=donor(ep,decode=False); new=model(ep,decode=False)
    assert all(torch.equal(x,y) for x,y in zip(old.states,new.states))
    _score(old,ep,item,donor.config,single).loss.backward();_score(new,ep,item,model.config,single).loss.backward()
    for key,p in donor.named_parameters():
        q=dict(model.named_parameters())[key]
        if p.grad is None: assert q.grad is None
        else: assert torch.allclose(p.grad,q.grad,atol=tol,rtol=tol),key
    outputs=[p for k,p in model.named_parameters() if '.row_encoder.' in k and ('.out.weight' in k or '.ff.2.weight' in k)]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in outputs)
    assert any(p.grad.abs().max()>0 for p in outputs)
    assert model.transfer_receipt['not_copied']==[]
    assert len(model.transfer_receipt['new_tensors'])==36
    opt=make_optimizer(model,OptimizerSpec(lr=3e-5));before={k:v.clone() for k,v in model.state_dict().items()}
    record=train_step(model,opt,[item],grad_clip_norm=1.)
    assert torch.isfinite(torch.tensor(record.loss))
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if '.row_encoder.' in k)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    observed=ep.observed.clone(); out=model(ep,decode=False)
    assert torch.equal(observed,ep.observed)
    changed=task(single=single,device=device,dtype=dtype,changed_truth=True)
    assert all(torch.equal(a,b) for a,b in zip(out.states,model(_episode(changed,model.config,single),decode=False).states))
    saved=tmp_path/'target.pt';manifest={'qualification':True}
    save_checkpoint(saved,checkpoint_state(model,opt,step=1,manifest=manifest))
    restored=V7Model(model.config).to(device=device,dtype=dtype)
    assert load_checkpoint(saved,restored,make_optimizer(restored),manifest=manifest)==1
    assert all(torch.equal(a,b) for a,b in zip(out.states,restored(ep,decode=False).states))
    with pytest.raises(ValueError,match='ModelSpec'):
        load_checkpoint(saved,donor,manifest=manifest)

@pytest.mark.parametrize('share',[True,False])
@pytest.mark.parametrize('units',[0,1])
def test_config_migration_metadata(tmp_path,share,units):
    donor,path=_donor(tmp_path,share_rounds=share,unit_layers=units)
    model=row_reversible64_from_checkpoint(path)
    assert V7Config.from_dict(model.config.as_dict())==model.config
    assert replace(model.config,value_encoder='phi_lift',row_reversible64=None)==donor.config
    assert set(donor.state_dict())=={x['key'] for x in model.transfer_receipt['copied_tensors']}
    assert len(model.transfer_receipt['new_tensors'])==36*(1 if share else 2)
    with pytest.raises(ValueError,match='SHA-256'):
        row_reversible64_from_checkpoint(path,expected_sha256='0'*64)


def test_defaults_and_invalid_config():
    assert 'row_reversible64' not in V7Config.v73().as_dict()
    assert 'row_reversible64' not in V7Config.v73(value_encoder='row_dual_stream',dual_stream=DualStreamConfig()).as_dict()
    for args in ({'value_encoder':'row_reversible64'}, {'row_reversible64':RowReversible64Config()}, {'value_encoder':'row_reversible64','row_reversible64':{'heads':3}}):
        with pytest.raises(ValueError): V7Config.v73(**args)
