"""One fixed learning rate per table; monitor-only selection before any test."""
import argparse,copy,hashlib,json,math,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
B=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
OLD=B.parent.parent.parent/'v6-puma-single-hostloss-15m-20260929'
sys.path[:0]=[str(OLD),str(OLD/'source/src')]
from common import metric_rows
def load_data():
    return json.loads((B/'fit-data.json').read_text()),json.loads((B/'fit-bank.json').read_text())
DATA,BANK=load_data()
NAME=B.name
WIDTH=len(DATA['values'][0])
NFIT=len(DATA['splits']['train'])
NVALID=len(DATA['splits']['validation'])
NTEST=len(DATA['splits']['test'])
WINDOW=min(512,NFIT)
NQUERY=max(1,round(WINDOW/3))
from frozen_evaluate import make_input,state_hash
from dual import DualModel,prepare,score
from training_support import checkpoint,restore_rng,rng_state,episode_seeds
from tabu_lab.curriculum_v53.artifacts import atomic_json,append_event,finite_state,sha256
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.protocol import load_v55_plan,schedule_entry
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v53 import V53LossConfig
from tabu_lab.restoration_optimizers import adamw

from finite_backward import backward_checked

LRS=[('lr1e-4',1e-4)]
def utc():return datetime.now(timezone.utc).isoformat()

def main(preflight=False):
    q=json.loads((B/'queue.json').read_text());parent=q['parent'];device=q['device'];branch=q['branch']
    runtime=configure_runtime(device)
    if device=='mps':torch.mps.set_per_process_memory_fraction(.5)
    else:torch.cuda.set_per_process_memory_fraction(.5)
    data,bank=load_data();sp=json.loads((B/'split.json').read_text())
    plan=load_v55_plan(B/'single-manifest.json');oldplan=load_v55_plan(parent['manifest'])
    assert sha256(parent['checkpoint'])==parent['checkpoint_sha256']
    donor=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)
    assert donor['model_config']==plan.config.as_dict()
    assert donor['identity']['parent_manifest_sha256']==sha256(parent['manifest'])
    offset=donor['state']['table_episode_offsets'][NAME];used=0
    for i in range(donor['state']['cursor']):
        table,ep=schedule_entry(oldplan,0,i)
        if table.name==NAME:used=max(used,ep+1)
    offset+=used
    table=plan.tables[0];assert set(table.row_ids)==set(sp['fit_row_ids'])
    stage=plan.spec['stages'][0];seeds=episode_seeds(plan);lc=V53LossConfig(**stage['loss'])
    prior_path=B/'recovery-budget.json'
    prior_seconds=json.loads(prior_path.read_text())['prior_attempt_successful_seconds'] if prior_path.exists() else 0.
    root=B/('repair-preflight' if preflight else 'run');root.mkdir(exist_ok=False)
    status=dict(outcome='preflight' if preflight else 'running',host=q['host'],branch=branch,
        parent_sha256=parent['checkpoint_sha256'],parent_checkpoint=parent['checkpoint'],runtime=runtime,
        split_sha256=sha256(B/'split.json'),bank_sha256=sha256(B/'fit-bank.json'),
        update_rows=NFIT,monitor_rows=NVALID,original_train_rows=NFIT+NVALID,test_rows=NTEST,window_rows=WINDOW,
        memory_fraction=.5,per_trial_successful_seconds=900,max_total_successful_seconds=900,
        monitor_exposure='parent previously trained these rows; no new updates on monitor rows',
        selection='minimum monitor MSE; earliest node on ties; all 15-minute curves retained',
        original_train_fit_scope='only new-update rows, masked target query; no validation labels in support',
        prior_attempt_successful_seconds=prior_seconds,repair='bounded-identical-backward-replay-v1',optimizer_restored=True,strict_resume=False,first_episode_index=offset,started_utc=utc(),trials={},test_started=False)
    def save():
        atomic_json(root/'campaign.json',status)
        if not preflight:atomic_json(B/'status.json',status)
    def sync():
        if device=='mps':torch.mps.synchronize()
        else:torch.cuda.synchronize()
    def eval_traces(model,out,traces,digest,scope):
        out.mkdir(parents=True,exist_ok=False);saved=rng_state();before=state_hash(model);rows=[]
        model.eval()
        try:
            for trace in traces:
                inputs,req=make_input(data,trace,NAME,device,execution_dtype(device))
                n=len(trace['context_row_ids'])
                assert not bool(inputs.visible[n:,-1].any()) and bool((inputs.values[-1][n:]==0).all())
                with torch.no_grad():answer=model(inputs,req)
                assert len(answer.columns)==1 and answer.columns[0].result.status=='ok'
                col=answer.columns[0];address=req.targets.detach().cpu().tolist()
                for value,pos in zip(col.decoded.detach().cpu().tolist(),col.target_indices.detach().cpu().tolist(),strict=True):
                    r,c=address[pos];assert c==WIDTH-1 and r>=n and math.isfinite(value)
                    rid=trace['query_row_ids'][r-n]
                    rows.append(dict(row_id=rid,target=float(data['values'][rid][-1]),prediction=float(value)))
                del inputs,req,answer,col
                if device=='mps':torch.mps.empty_cache()
                else:torch.cuda.empty_cache()
            assert len(rows)==len({r['row_id'] for r in rows})==sum(len(t['query_row_ids']) for t in traces)
            assert state_hash(model)==before
            rows.sort(key=lambda r:r['row_id']);atomic_json(out/'predictions.json',rows)
            report=dict(outcome='completed',scope=scope,checkpoint_sha256=digest,branch=branch,
                split_sha256=status['split_sha256'],bank_sha256=status['bank_sha256'],metrics=metric_rows(rows,data),
                predictions=len(rows),optimizer_updates=0,model_state_unchanged=True)
            atomic_json(out/'terminal.json',report)
            return report
        finally:restore_rng(saved)
    monitor_trace=dict(context_row_ids=sp['fit_row_ids'],query_row_ids=sp['validation_row_ids'],
                       codebook_seed=bank['test_trace']['codebook_seed'])
    save()
    try:
        for tag,lr in LRS:
            trialroot=root/tag;trialroot.mkdir()
            model=DualModel(plan.config).to(device,dtype=execution_dtype(device));model.branch=branch
            model.load_state_dict(donor['model'],strict=True)
            optimizer=adamw(model,plan.optimizer);optimizer.load_state_dict(copy.deepcopy(donor['optimizer']))
            for pg in optimizer.param_groups:pg['lr']=lr
            restore_rng(donor['rng'])
            assert state_hash(model)==state_hash_from_parent
            state=dict(update=donor['state']['update'],cursor=0,successful_update_seconds=prior_seconds,
                table_episode_offsets={NAME:offset},table_updates={NAME:0},parent_cursor=donor['state']['cursor'])
            identity=dict(schema='tabu.openml12.single.earlystop.v1',supervision='target_only',objective=NAME+'_'+branch,
                inference_branch=branch,parent_sha256=parent['checkpoint_sha256'],
                parent_manifest_sha256=sha256(B/'single-manifest.json'),model_config=plan.config.as_dict(),
                runtime=runtime,learning_rate=lr,split_sha256=status['split_sha256'],strict_resume=False,
                code={f.name:sha256(f) for f in Path(__file__).resolve().parent.glob('*.py')})
            identity['sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            tr=dict(outcome='running',learning_rate=lr,nodes={},successful_update_seconds=prior_seconds,prior_attempt_successful_seconds=prior_seconds,updates=0,
                parent_sha256=parent['checkpoint_sha256'],initial_state_hash=state_hash(model),
                first_episode_index=offset,initial_rng_restored=True)
            status['trials'][tag]=tr;status['active_trial']=tag;save()
            def step():
                tick=time.monotonic();episode=offset+state['cursor']
                inputs,request,truth,info=build_episode(table,stage['recipe']['real'],episode,seeds,device,
                    epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
                assert inputs.query.shape==(WINDOW,WIDTH) and int(inputs.query[:,:-1].sum())==0
                assert int(inputs.query.sum())==NQUERY and not bool(inputs.visible[inputs.query].any())
                model.train()
                def forward():
                    prepared=prepare(model,inputs,request,truth,lc)
                    assert len(prepared.visible.request.targets)==NQUERY
                    return score(model,prepared,lc,branch)
                def record_failure(event):
                    assert finite_state(model.state_dict()) and finite_state(optimizer.state_dict())
                    path,digest=checkpoint(trialroot,model,optimizer,state,identity)
                    append_event(trialroot/'numerical-events.jsonl',dict(event,utc=utc(),episode=episode,
                        cursor=state['cursor'],checkpoint=str(path),checkpoint_sha256=digest,optimizer_updated=False))
                result,grads,retry_count,retry_seconds=backward_checked(forward,model,optimizer,
                    capture_rng=rng_state,restore_rng=restore_rng,sync=sync,on_failure=record_failure)
                gn=torch.nn.utils.clip_grad_norm_(grads,plan.optimizer.grad_clip,error_if_nonfinite=True)
                assert all(pg['lr']==lr for pg in optimizer.param_groups)
                optimizer.step();assert finite_state(model.state_dict()) and finite_state(optimizer.state_dict());sync()
                elapsed=time.monotonic()-tick-retry_seconds;state['successful_update_seconds']+=elapsed
                state['update']+=1;state['cursor']+=1;state['table_updates'][NAME]+=1
                row=dict(trial=tag,branch=branch,learning_rate=lr,table=table.name,update=state['update'],
                    episode_index=episode,loss=float(result.loss.detach()),gradient_norm=float(gn),seconds=elapsed,
                    successful_update_seconds=state['successful_update_seconds'],train_rows=WINDOW,support_rows=WINDOW-NQUERY,
                    query_rows=NQUERY,scored_cells=NQUERY,forward_passes=1,optimizer_steps=1,
                    backward_replays=retry_count,failed_attempt_seconds=retry_seconds,attempted_forward_passes=1+retry_count)
                append_event(trialroot/'updates.jsonl',row)
                tr.update(successful_update_seconds=state['successful_update_seconds'],retained_update_seconds=state['successful_update_seconds']-prior_seconds,updates=state['cursor'])
                return row
            if preflight:
                for _ in range(6):tr['discarded_step']=step()
                if tag==LRS[0][0]:
                    tr['monitor_smoke']=eval_traces(model,trialroot/'monitor-smoke',[monitor_trace],'discarded','monitor')
                tr['outcome']='passed';save()
            else:
                for seconds in [0,180,360,540,720,900]:
                    status.update(phase='training',active_target_seconds=seconds);save()
                    while state['successful_update_seconds']<seconds:
                        step()
                        if state['cursor']%25==0:save()
                    if seconds:path,digest=checkpoint(trialroot,model,optimizer,state,identity)
                    else:path,digest=parent['checkpoint'],parent['checkpoint_sha256']
                    status.update(phase='monitor_evaluation');save()
                    ev=eval_traces(model,trialroot/'monitor'/f'sec{seconds:04d}',[monitor_trace],digest,'monitor')
                    tr['nodes'][str(seconds)]=dict(checkpoint=str(path),checkpoint_sha256=digest,
                        seconds=state['successful_update_seconds'],updates=state['cursor'],metrics=ev['metrics'])
                    save();print(json.dumps(dict(event='monitor',trial=tag,node=seconds,metrics=ev['metrics'])),flush=True)
                chosen=min(tr['nodes'],key=lambda k:(tr['nodes'][k]['metrics']['mse'],int(k)))
                tr.update(outcome='completed',selected_node=chosen,selected=tr['nodes'][chosen])
                assert 900<=tr['successful_update_seconds']<930
                atomic_json(trialroot/'selection.json',tr['selected']);save()
            del optimizer,model
            if device=='mps':torch.mps.empty_cache()
            else:torch.cuda.empty_cache()
        if preflight:
            assert sha256(parent['checkpoint'])==parent['checkpoint_sha256']
            status.update(outcome='passed',discarded_updates=6,formal_updates=0,parent_unchanged=True);save();return
        winner=min(status['trials'],key=lambda k:status['trials'][k]['selected']['metrics']['mse'])
        status.update(selection_completed_utc=utc(),selected_trial=winner)
        atomic_json(B/'selection.json',dict(selected_trial=winner,completed_utc=status['selection_completed_utc'],
            trials={k:t['selected'] for k,t in status['trials'].items()}));save()
        # Freeze all monitor decisions before scoring any original test predictions.
        status.update(phase='final_evaluations',test_started=True,test_started_utc=utc());save()
        for tag,tr in status['trials'].items():
            selected=tr['selected'];path=selected['checkpoint']
            assert sha256(path)==selected['checkpoint_sha256']
            payload=torch.load(path,map_location='cpu',weights_only=False)
            model=DualModel(plan.config).to(device,dtype=execution_dtype(device));model.branch=branch
            model.load_state_dict(payload['model'],strict=True);del payload
            tr['final_evaluations']={}
            for group,traces in [('train',bank['train_traces']),('test',[bank['test_trace']])]:
                tr['final_evaluations'][group]=eval_traces(model,root/tag/'selected-eval'/group,traces,
                    selected['checkpoint_sha256'],'original_'+group)
                save()
            del model
            if device=='mps':torch.mps.empty_cache()
            else:torch.cuda.empty_cache()
        status.update(outcome='completed',completed_utc=utc(),total_successful_seconds=sum(t['successful_update_seconds'] for t in status['trials'].values()));save()
        atomic_json(B/'terminal.json',status)
    except Exception as ex:
        status.update(outcome='failed',error_type=type(ex).__name__,error=str(ex));save()
        if not preflight:atomic_json(B/'terminal.json',status)
        raise

if __name__=='__main__':
    # This hash uses the same implementation as live models and does not mutate the parent.
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    q=json.loads((B/'queue.json').read_text())
    d=torch.load(q['parent']['checkpoint'],map_location='cpu',weights_only=False)
    cfg=load_v55_plan(B/'single-manifest.json').config
    m=DualModel(cfg);m.load_state_dict(d['model'],strict=True)
    # FP64 CPU load must preserve parent dtype for CUDA; MPS parent uses FP32.
    m=m.to(dtype=execution_dtype(q['device']));m.load_state_dict(d['model'],strict=True)
    state_hash_from_parent=state_hash(m)
    del m,d
    main(args.preflight)
