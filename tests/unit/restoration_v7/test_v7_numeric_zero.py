"""Versioned numeric coordinates and structural zero preservation."""
import pytest
import torch
from tabu_lab.models.restoration.contracts import ColumnSchema, RestorationInput
from tabu_lab.models.restoration_v7 import V7Config, V7Model, prepare_episode
from tabu_lab.models.restoration_v7.codec import build_value_codec, select_softlog
from tabu_lab.primitives.coupling import CouplingValueMap


@pytest.fixture(autouse=True)
def isolated_mps_algorithm_mode(request):
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    if parameters.get("device") == "mps":
        # Match the supported MPS runtime; preserve the caller's global mode.
        torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def inputs(x, y=None, hidden=0.):
    x = torch.tensor(x, dtype=torch.float64)
    if y is None:
        y = list(range(len(x)))
    y = torch.tensor(y, dtype=torch.float64)
    y[-1] = hidden
    vis = torch.ones(len(x), 2, dtype=torch.bool)
    vis[-1, -1] = False
    qry = torch.zeros_like(vis)
    qry[-1, -1] = True
    return RestorationInput((ColumnSchema('x','numeric'),ColumnSchema('y','numeric')), (x,y),vis,qry,4)


@pytest.mark.parametrize('family,dim', [('C64/8',64),('G64',64),('constant_weight_composition_v2',128)])
def test_std_floor_threshold_full_column_roundtrip(family, dim):
    support = inputs([-.005, .005, 0.])
    codec = build_value_codec(support, codec=family, dim=dim, epsilon=1e-20,
                              numeric_preprocessing=V7Config.v73(numeric_preprocessing='standard_softlog_v1').numeric_preprocessing)
    c = codec.columns[0]
    assert c.scale == .01
    assert c.mean == 0.
    at = select_softlog(codec, inputs([-.005, .005, 10000.]))
    above_inputs = inputs([-.005, .005, 10000.01])
    above = select_softlog(codec, above_inputs)
    assert not at.columns[0].softlog
    assert above.columns[0].softlog
    assert not codec.columns[0].softlog
    c = above.columns[0]
    values = torch.tensor([-.005, 0., .005, -10000.01, 10000.01],dtype=torch.float64)
    z = values / .01
    expected = c.base + (z.sign()*z.abs().log1p())[:,None]*c.direction
    torch.testing.assert_close(c.encode(values),expected)
    torch.testing.assert_close(c.decode(c.encode(values)), values, atol=1e-9, rtol=1e-12)
    # Both low-amplitude supports and extreme query use the one switched codec.
    assert not torch.equal(codec.encode_observed(above_inputs)[0,0],above.encode_observed(above_inputs)[0,0])
    torch.testing.assert_close(above.encode_observed(above_inputs)[:,0],c.encode(above_inputs.values[0]))
    poisoned = select_softlog(codec, inputs([-.005,.005,10000.01], hidden=1e100))
    assert poisoned.columns[-1].mean == codec.columns[-1].mean
    assert poisoned.columns[-1].softlog == codec.columns[-1].softlog


def test_constant_sparse_null_and_target_roles():
    preprocessing = V7Config.v73().numeric_preprocessing
    constant = prepare_episode(inputs([2.]*10), donor_seed=0, numeric_preprocessing=preprocessing)
    assert not constant.inputs.visible[:,0].any()
    assert constant.observed[:,0].count_nonzero() == 0
    sparse = prepare_episode(inputs([0.]*9+[1.]), donor_seed=0, numeric_preprocessing=preprocessing)
    assert sparse.inputs.visible[:,0].all()  # IQR=0 is not a constant
    assert sparse.codec.columns[0].scale == pytest.approx(.3)
    infer = prepare_episode(inputs([0.,1.,2.],y=[4.,4.,0.]),donor_seed=0,admission='inference',
                            numeric_preprocessing=preprocessing)
    assert infer.inputs.visible[:2,1].all()  # target is protected
    assert infer.codec.columns[1].scale == .01
    missing = inputs([1.,2.,3.])
    vis=missing.visible.clone(); vis[:,0]=False
    episode=prepare_episode(RestorationInput(missing.schema,missing.values,vis,missing.query,4),
                            donor_seed=0,numeric_preprocessing=preprocessing)
    assert not episode.inputs.visible[:,0].any()
    assert episode.inputs.visible.shape == (3,2)


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_nonidentity_zero_inverse_and_optimized_gradients(dtype):
    torch.manual_seed(194)
    phi=CouplingValueMap(8,n_blocks=4,hidden=(12,12),bias=False).to(dtype)
    assert all(m.bias is None for m in phi.modules() if isinstance(m,torch.nn.Linear))
    with torch.no_grad():
        for block in phi.blocks:
            block.net[-1].weight.normal_(0,.08)
    x=torch.randn(12,8,dtype=dtype,requires_grad=True)
    assert not torch.allclose(phi(x),x)
    opt=torch.optim.AdamW(phi.parameters(),lr=.001)
    for _ in range(3):
        opt.zero_grad(); loss=(phi(x)-.5*x).square().mean(); loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in phi.parameters())
        assert torch.isfinite(x.grad).all()
        opt.step()
    zero=torch.zeros(3,8,dtype=dtype)
    assert torch.equal(phi(zero),zero)
    assert torch.equal(phi.inverse(zero),zero)
    tol=2e-6 if dtype==torch.float32 else 1e-12
    torch.testing.assert_close(phi.inverse(phi(x)),x,rtol=tol,atol=tol)
    torch.testing.assert_close(phi(phi.inverse(x)),x,rtol=tol,atol=tol)


def test_checkpoint_boundary_and_legacy_bias_behavior():
    config=V7Config.v73(backbone=dict(width=64,layers=1,heads=2,ff_width=64,slots=4),coupling_hidden=(12,))
    assert not config.coupling_bias and config.epsilon==1e-6
    old=config.as_dict(); del old['model_version']; del old['coupling_bias']; del old['numeric_preprocessing']
    legacy=V7Config.from_dict(old)
    assert legacy.coupling_bias and legacy.numeric_preprocessing=='legacy'
    old_model=V7Model(legacy).double()
    with torch.no_grad():
        old_model.rounds[0].phi.blocks[0].net[-1].bias.fill_(.1)
    assert old_model.rounds[0].phi(torch.zeros(1,64,dtype=torch.float64)).count_nonzero()>0
    restored=V7Model(V7Config.from_dict(old)).double()
    restored.load_state_dict(old_model.state_dict(),strict=True)
    new=V7Model(config).double()
    assert new.rounds[0].lift.bias is None
    with pytest.raises(RuntimeError,match='Unexpected key'):
        new.load_state_dict(old_model.state_dict(),strict=True)
    restored_new=V7Model(V7Config.from_dict(config.as_dict())).double()
    restored_new.load_state_dict(new.state_dict(),strict=True)
    ep=prepare_episode(inputs([0.,.001,.002]),donor_seed=0,
                        numeric_preprocessing=config.numeric_preprocessing)
    with pytest.raises(ValueError,match='preprocessing'):
        old_model(ep)
    legacy_ep=prepare_episode(inputs([0.,.001,.002]),donor_seed=0,numeric_preprocessing='legacy')
    assert legacy_ep.codec.columns[0].scale < .01
    assert torch.isfinite(old_model(legacy_ep).states[-1]).all()


@pytest.mark.parametrize('device,dtype',[('cpu',torch.float64),('cpu',torch.float32),('mps',torch.float32)])
def test_full_model_training_device_and_zero(device,dtype):
    if device=='mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    from tabu_lab.models.restoration.contracts import make_episode
    from tabu_lab.models.restoration_v7 import V7Task, make_optimizer, train_step, evaluate_task
    from tabu_lab.models.restoration_v7.joint import prepare_joint_episode
    torch.manual_seed(8)
    schema=(ColumnSchema('x','numeric'),ColumnSchema('y','numeric'),ColumnSchema('constant','numeric'))
    values=tuple(torch.tensor(v,device=device,dtype=dtype) for v in ([0.,.001,.002,.003],[-1.,0.,1.,2.],[5.,5.,5.,5.]))
    query=torch.zeros(4,3,device=device,dtype=torch.bool);query[-1,1]=True
    inp,_,truth=make_episode(schema,values,torch.ones_like(query),query,code_seed=4)
    cfg=V7Config.v73(backbone=dict(width=64,layers=1,heads=2,ff_width=64,slots=4),coupling_hidden=(16,),rounds=2)
    model=V7Model(cfg).to(dtype=dtype).to(device)
    with torch.no_grad():
        for block in model.rounds[0].phi.blocks:
            block.net[-1].weight.normal_(0,.01)
    task=V7Task(inp,truth,3)
    record=train_step(model,make_optimizer(model),[task])
    assert record.loss >= 0 and torch.isfinite(torch.tensor(record.loss))
    report=evaluate_task(model,task)
    assert report.status=='ok'
    zero=torch.zeros(2,64,device=device,dtype=dtype)
    assert torch.equal(model.rounds[0].phi(zero),zero)
    assert torch.equal(model.rounds[0].phi.inverse(zero),zero)
    sample=torch.randn(3,64,device=device,dtype=dtype)
    tol=1e-5 if dtype==torch.float32 else 1e-12
    torch.testing.assert_close(model.rounds[0].phi.inverse(model.rounds[0].phi(sample)), sample, rtol=tol, atol=tol)
    ep=prepare_episode(inp,donor_seed=3,numeric_preprocessing=cfg.numeric_preprocessing)
    joint=prepare_joint_episode(inp,donor_seed=3,config=cfg)
    torch.testing.assert_close(ep.observed,joint.observed)
    assert ep.codec.columns[0].scale == .01
    assert not ep.inputs.visible[:,2].any()
    assert next(model.parameters()).device.type==device
    assert next(model.parameters()).dtype==dtype


def test_softlog_intermediate_overflow_is_not_raw_value_overflow():
    codec=build_value_codec(inputs([-.005,.005,0.]),
                            numeric_preprocessing=V7Config.v73(numeric_preprocessing='standard_softlog_v1').numeric_preprocessing)
    ep=inputs([-.005,.005,1e308])
    col=select_softlog(codec,ep).columns[0]
    values=torch.tensor([-1e308,0.,1e308],dtype=torch.float64)
    codes=col.encode(values)
    assert torch.isfinite(codes).all()
    torch.testing.assert_close(col.decode(codes),values,rtol=2e-12,atol=0.)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_softlog_decode_cancels_unrepresentable_displacement_with_large_mean(sign):
    support = inputs([sign * 1e308, sign * 1.000000001e308])
    codec = build_value_codec(support, numeric_preprocessing=V7Config.v73(numeric_preprocessing='standard_softlog_v1').numeric_preprocessing)
    episode = inputs([sign * 1e308, sign * 1.000000001e308, -sign * 1e308])
    column = select_softlog(codec, episode).columns[0]
    assert column.softlog
    values = episode.values[0]
    encoded = column.encode(values)
    assert torch.isfinite(encoded).all()
    torch.testing.assert_close(column.decode(encoded), values, rtol=2e-12, atol=0.)
    # A genuinely unrepresentable original-unit answer still fails explicitly.
    impossible = column.base + sign * 1000. * column.direction
    with pytest.raises(FloatingPointError, match='decoded values'):
        column.decode(impossible[None])


def test_legacy_strict_resume_keeps_optimizer_and_rejects_new_structure(tmp_path):
    from tabu_lab.models.restoration_v7 import V7Task, make_optimizer, train_step, checkpoint_state, save_checkpoint, load_checkpoint
    from tabu_lab.models.restoration.contracts import make_episode
    inp=inputs([0.,.001,.002,.003])
    values=(inp.values[0],torch.tensor([-1.,0.,1.,2.],dtype=torch.float64))
    inp,_,truth=make_episode(inp.schema,values,torch.ones_like(inp.visible),inp.query,code_seed=4)
    task=V7Task(inp,truth,2)
    config=V7Config(coupling_bias=True,numeric_preprocessing='legacy',coupling_hidden=(12,),
                    backbone=dict(width=64,layers=1,heads=2,ff_width=64,slots=4))
    model=V7Model(config).double(); opt=make_optimizer(model)
    train_step(model,opt,[task])
    manifest={'test':'old-config-strict-resume'}
    state=checkpoint_state(model,opt,step=1,manifest=manifest)
    state['config'].pop('coupling_bias'); state['config'].pop('numeric_preprocessing')
    path=tmp_path/'old.pt'; save_checkpoint(path,state)
    resumed=V7Model(V7Config.from_dict(state['config'])).double(); opt2=make_optimizer(resumed)
    assert load_checkpoint(path,resumed,opt2,manifest=manifest)==1
    assert train_step(model,opt,[task]).loss == train_step(resumed,opt2,[task]).loss
    for a,b in zip(model.parameters(),resumed.parameters(),strict=True):
        assert torch.equal(a,b)
    new=V7Model(V7Config.v73(backbone=config.backbone,coupling_hidden=config.coupling_hidden)).double()
    with pytest.raises(ValueError,match='ModelSpec'):
        load_checkpoint(path,new,make_optimizer(new),manifest=manifest)
