"""One isolated branch, Puma only, 900 successful training seconds."""
import argparse,copy,hashlib,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from datetime import datetime,timezone
import torch
from common import B,load_data,evaluate_model
from dual import DualModel,prepare,score
from training_support import checkpoint,restore_rng,rng_state,episode_seeds
from tabu_lab.curriculum_v53.artifacts import atomic_json,append_event,finite_state,sha256
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.protocol import load_v55_plan,schedule_entry
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v53 import V53LossConfig
from tabu_lab.models.restoration_v55 import RestorationRequest
from tabu_lab.restoration_optimizers import adamw

def main(preflight=False):
    q=json.loads((B/'queue.json').read_text());p=q['parent'];device=q['device'];branch=q['branch']
    runtime=configure_runtime(device)
    if device=='mps': torch.mps.set_per_process_memory_fraction(.5)
    else: torch.cuda.set_per_process_memory_fraction(.5)
    plan=load_v55_plan(B/'single-manifest.json');old_plan=load_v55_plan(p['manifest'])
    assert len(plan.tables)==1 and plan.tables[0].name=='pumadyn32nh'
    assert sha256(p['checkpoint'])==p['checkpoint_sha256']
    donor=torch.load(p['checkpoint'],map_location='cpu',weights_only=False)
    assert donor['purpose']=='training' and donor['model_config']==plan.config.as_dict()
    assert donor['identity']['parent_manifest_sha256']==sha256(p['manifest'])
    offset=donor['state']['table_episode_offsets']['pumadyn32nh'];used=0
    for i in range(donor['state']['cursor']):
        t,e=schedule_entry(old_plan,0,i)
        if t.name=='pumadyn32nh': used=max(used,e+1)
    offset+=used
    model=DualModel(plan.config).to(device,dtype=execution_dtype(device));model.branch=branch
    model.load_state_dict(donor['model'],strict=True)
    optimizer=adamw(model,plan.optimizer);optimizer.load_state_dict(donor['optimizer'])
    restore_rng(donor['rng'])
    state=dict(update=donor['state']['update'],cursor=0,successful_update_seconds=0.,table_episode_offsets={'pumadyn32nh':offset},table_updates={'pumadyn32nh':0},parent_cursor=donor['state']['cursor'])
    identity=dict(schema='tabu.puma.single.v1',supervision='target_only',objective='puma_single_'+branch,
                  inference_branch=branch,parent_sha256=p['checkpoint_sha256'],parent_manifest_sha256=sha256(B/'single-manifest.json'),
                  model_config=plan.config.as_dict(),runtime=runtime,strict_resume=False,
                  initialization='model optimizer RNG retained; single-table schedule reset and Puma episode offset settled',
                  code={f.name:sha256(f) for f in sorted(B.glob('*.py'))})
    identity['sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    del donor,old_plan
    data,bank=load_data();table=plan.tables[0];stage=plan.spec['stages'][0];seeds=episode_seeds(plan);lc=V53LossConfig(**stage['loss'])
    status=dict(outcome='preflight' if preflight else 'evaluating_before',host=q['host'],branch=branch,parent_sha256=p['checkpoint_sha256'],additional_seconds_target=900,successful_update_seconds=0,started_utc=datetime.now(timezone.utc).isoformat(),evaluations={},identity=identity,parameter_count=sum(t.numel() for t in model.parameters()),optimizer_restored=True,rng_restored=True,schedule_cursor_reset=True,first_episode_index=offset,loss_scope='Query target column only; own-cell encoding squared error',window_rows=512,memory_fraction=.5)
    root=B/('preflight' if preflight else 'run');root.mkdir(exist_ok=False)
    status_path=root/'campaign.json'
    def save():
        atomic_json(status_path,status)
        if not preflight: atomic_json(B/'status.json',status)
    def step():
        tick=time.monotonic();ep=offset+state['cursor']
        inputs,request,truth,info=build_episode(table,stage['recipe']['real'],ep,seeds,device,epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
        assert inputs.query.shape==(512,33)
        assert int(inputs.query[:,:-1].sum())==0
        assert not bool(inputs.visible[inputs.query].any())
        model.train();optimizer.zero_grad(set_to_none=True)
        prepared=prepare(model,inputs,request,truth,lc)
        result=score(model,prepared,lc,branch)
        assert len(prepared.visible.request.targets)==int(inputs.query.sum())
        result.loss.backward();grads=[p for p in model.parameters() if p.grad is not None]
        assert grads and finite_state([p.grad for p in grads])
        norm=torch.nn.utils.clip_grad_norm_(grads,plan.optimizer.grad_clip,error_if_nonfinite=True)
        optimizer.step()
        assert finite_state(model.state_dict()) and finite_state(optimizer.state_dict())
        if device=='mps': torch.mps.synchronize()
        else: torch.cuda.synchronize()
        elapsed=time.monotonic()-tick
        state['successful_update_seconds']+=elapsed;state['update']+=1;state['cursor']+=1;state['table_updates']['pumadyn32nh']+=1
        rec=dict(update=state['update'],episode_index=ep,table=table.name,branch=branch,loss=float(result.loss.detach()),gradient_norm=float(norm),seconds=elapsed,successful_update_seconds=state['successful_update_seconds'],train_rows=512,query_rows=int(inputs.query.sum()),support_rows=512-int(inputs.query.sum()),scored_cells=len(prepared.visible.request.targets),forward_passes=1,optimizer_steps=1,selected_loss_weight=1.)
        append_event(root/'updates.jsonl',rec)
        return rec
    save()
    try:
        if preflight:
            rec=step()
            # Decode a real masked episode after a disposable update.
            inputs,request,truth,_=build_episode(table,stage['recipe']['real'],offset+1,seeds,device,epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
            model.eval()
            query_request=RestorationRequest(inputs.query.nonzero())
            with torch.no_grad(): answer=model(inputs,query_request)
            assert len(answer.columns)==1 and answer.columns[0].result.status=='ok'
            assert sha256(p['checkpoint'])==p['checkpoint_sha256']
            status.update(outcome='passed',discarded_updates=1,formal_updates=0,step=rec,parent_unchanged=True,runtime=runtime)
            save();print(json.dumps(status));return
        for seconds,label in [(0,'before'),(300,'minute05'),(600,'minute10'),(900,'minute15')]:
            status.update(outcome='training',target_seconds=seconds);save()
            while state['successful_update_seconds']<seconds:
                step()
                if state['cursor']%50==0:
                    status.update(successful_update_seconds=state['successful_update_seconds'],new_updates=state['cursor']);save()
            if seconds:
                path,digest=checkpoint(root,model,optimizer,state,identity)
            else: path,digest=p['checkpoint'],p['checkpoint_sha256']
            status.update(outcome='evaluating',phase=label,checkpoint=str(path),checkpoint_sha256=digest,successful_update_seconds=state['successful_update_seconds'],new_updates=state['cursor']);save()
            saved_rng=rng_state()
            ev=evaluate_model(model,device,label,digest,data,bank)
            restore_rng(saved_rng)
            status['evaluations'][label]=dict(metrics=ev['metrics'],checkpoint_sha256=digest,checkpoint=str(path),successful_update_seconds=state['successful_update_seconds'],new_updates=state['cursor'],bank_sha256=ev['bank_sha256'])
            save();print(json.dumps(dict(node=label,metrics=ev['metrics'],successful_update_seconds=state['successful_update_seconds'])),flush=True)
        assert 900<=state['successful_update_seconds']<930
        status.update(outcome='completed',completed_utc=datetime.now(timezone.utc).isoformat());save()
    except Exception as ex:
        status.update(outcome='failed',error_type=type(ex).__name__,error=str(ex));save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');args=p.parse_args();main(args.preflight)
