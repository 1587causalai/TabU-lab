from pathlib import Path
import json, torch
from tabu_lab.models.restoration_v53 import readout
from tabu_lab.curriculum_v53.runner import configure_runtime
configure_runtime('mps')
base=Path(__file__).resolve().parent
original=dict(readout.__dict__)
exec(compile((base/'readout-original.py').read_text(),'original','exec'),original)
torch.manual_seed(180)
data=[torch.randn(32,8,device='mps',requires_grad=True),torch.randn(20,8,device='mps',requires_grad=True)]
answers=torch.randn(20,128,device='mps');rows=torch.arange(20,device='mps')
outputs=[];grads=[]
for fn, ev in [(original['shared_slope'],original['evaluate_column']),(readout.shared_slope,readout.evaluate_column)]:
 u,x=[a.detach().clone().requires_grad_(True) for a in data]
 out=fn(u,rows,x,answers,ridge=.001,bandwidth=1.,center_chunk_size=7)
 cells=torch.cat((x,x[:12]),0)
 pred=ev(u,cells,rows,answers,torch.arange(20,32,device='mps'),out,bandwidth=1.,chunk_size=7).encoding
 outputs.append(pred.detach());grads.append(torch.autograd.grad(pred.square().mean(),(u,x)))
assert torch.allclose(outputs[0],outputs[1],rtol=1e-5,atol=1e-6)
assert all(torch.allclose(a,b,rtol=1e-4,atol=1e-5) for a,b in zip(*grads))
r={'outcome':'passed','forward_max_diff':float((outputs[0]-outputs[1]).abs().max()),'gradient_max_diffs':[float((a-b).abs().max()) for a,b in zip(*grads)]}
(base/'readout-verification.json').write_text(json.dumps(r));print(json.dumps(r))
