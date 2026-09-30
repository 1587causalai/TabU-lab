"""Verify exact continuation; no optimizer step or new training is performed."""
import run as R
import torch,json,gc
from model import make_model,make_optimizer

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

R.runtime();plan=R.load_v55_plan(R.B/'single-manifest.json')
sel=json.loads((R.B/'selection.json').read_text());refs=json.loads((R.B/'resume-parents.json').read_text());bank=json.loads((R.B/'fit-bank.json').read_text());out={}
old=R.B.parent/'v6-nonlinear-encoder-single-table-mini-20260929'
assert R.sha256(R.B/'source-hashes.json')==R.sha256(old/'source-hashes.json')
for arm,ref in refs.items():
    assert R.sha256(ref['checkpoint'])==ref['checkpoint_sha256']
    d=torch.load(ref['checkpoint'],map_location='cpu',weights_only=False)
    m=make_model(plan.config,d,arm,R.E['module_seed']);opt=make_optimizer(m,plan.optimizer,d)
    assert equal(m.state_dict(),d['model']) and equal(opt.state_dict(),d['optimizer'])
    R.restore_rng(d['rng']);rng=R.rng_state()
    assert torch.equal(rng['torch'],d['rng']['torch']) if 'torch' in rng else True
    assert torch.equal(rng['mps'],d['rng']['mps'])
    assert d['state']['cursor']==ref['updates'] and d['state']['table_episode_offsets'][sel['table']]==sel['episode_start']
    m.eval();metrics,rows=R.prediction(m,R.get_data(sel['entry']),sel['table'],bank[sel['table']][0])
    prior=json.loads((old/'evaluations'/arm/'900/fit-0-predictions.json').read_text())
    assert [x['row_id'] for x in rows]==[x['row_id'] for x in prior]
    error=max(abs(x['prediction']-y['prediction']) for x,y in zip(rows,prior));assert error<1e-5
    out[arm]=dict(parent_sha256=ref['checkpoint_sha256'],parameters_exact=True,optimizer_exact=True,rng_restored=True,
        cursor=d['state']['cursor'],next_episode=sel['episode_start']+d['state']['cursor'],prediction_max_abs_error=error,
        formal_updates=0,residual_u_norm=float(m.encoder.residual_u.weight.norm()) if arm=='residual_gelu' else None)
    if arm=='residual_gelu':assert out[arm]['residual_u_norm']>0
    del m,opt,d;gc.collect();torch.mps.empty_cache()
R.save('preflight.json',dict(outcome='passed',arms=out,formal_updates=0))
R.status(outcome='ready',additional_seconds_per_arm=600,formal_updates=0)
print(json.dumps(out))
