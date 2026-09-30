import json
from types import SimpleNamespace
import torch
from finite_backward import backward_checked

torch.set_num_threads(1)
device='mps'
torch.mps.set_per_process_memory_fraction(.5)
def sync(): torch.mps.synchronize()
def capture(): return (torch.get_rng_state(),torch.mps.get_rng_state())
def restore(s): torch.set_rng_state(s[0]);torch.mps.set_rng_state(s[1])
model=torch.nn.Linear(2,1).to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
saved={k:v.clone() for k,v in model.state_dict().items()}
def run(injections):
    calls=[];failures=[];remaining=[injections]
    def hook(g):
        if remaining[0]:
            remaining[0]-=1
            return g*float('nan')
        return g
    h=model.weight.register_hook(hook)
    def forward():
        x=torch.rand(3,2,device=device);calls.append(x.detach().cpu())
        return SimpleNamespace(loss=model(x).square().mean())
    initial=capture()
    try:
        result,grads,retries,failed_seconds=backward_checked(forward,model,opt,capture_rng=capture,restore_rng=restore,sync=sync,on_failure=failures.append)
        assert injections<=1
        actual=[p.grad.clone() for p in model.parameters()]
        ending=capture();h.remove();restore(initial);opt.zero_grad(set_to_none=True)
        reference=forward();reference.loss.backward();sync()
        assert all(torch.equal(a,p.grad) for a,p in zip(actual,model.parameters()))
        assert torch.equal(ending[0],capture()[0]) and torch.equal(ending[1],capture()[1])
    except FloatingPointError:
        assert injections==2
        retries=None;h.remove()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in saved.items()) and not opt.state
    assert all(torch.equal(calls[0],x) for x in calls)
    return dict(injections=injections,failures=len(failures),retries=retries,unchanged_parameters=True,optimizer_updates=0,identical_rng_inputs=True)
print(json.dumps(dict(outcome='passed',device=device,cases=[run(0),run(1),run(2)])))
