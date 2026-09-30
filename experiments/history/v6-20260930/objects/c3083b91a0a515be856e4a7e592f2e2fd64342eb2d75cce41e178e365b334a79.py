"""Single-table encoder ablation, two independently initialized 900-second arms."""
import os,sys,json,time,hashlib,random,statistics,copy,gc,argparse
from pathlib import Path
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'source/src'));sys.path.insert(0,str(B))
import torch
from model import make_model,make_optimizer
from tabu_lab.models.restoration_v53.encoding import AffineValueEncoder
from tabu_lab.models.restoration.contracts import RestorationRequest
from frozen_evaluate import make_input,metrics,state_hash
from training_support import restore_rng,rng_state,episode_seeds,checkpoint
from tabu_lab.curriculum_v53.artifacts import atomic_json,append_event,sha256,finite_state
from tabu_lab.curriculum_v53.protocol import load_v55_plan,schedule_entry
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration_v53 import V53LossConfig
from tabu_lab.models.restoration_v6 import score_training_episode
E=json.loads((B/'experiment.json').read_text())
P=json.loads((B/'parents.json').read_text())['gongqian-mini']
BANK=B.parent/'openml12-frozen-icl-20260927'

def save(name,obj):
    path=B/name;path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,obj)
def status(**kw):
    p=B/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
    
    if kw.get('outcome')=='training':
        for k in ['formal_updates','checkpoint_sha256','node']:s.pop(k,None)
    if kw.get('outcome') and kw['outcome']!='failed':
        for k in ['error','error_type','mode']:s.pop(k,None)
    s.update(kw,updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat());save('status.json',s)
def runtime():
    r=configure_runtime('mps');torch.mps.set_per_process_memory_fraction(.75);return r

def load_parent():
    assert sha256(P['checkpoint'])==P['checkpoint_sha256']
    donor=torch.load(P['checkpoint'],map_location='cpu',weights_only=False)
    assert donor['schema']=='tabu.v6.weights-only-checkpoint.v1' and donor['purpose']=='training'
    assert donor['identity']['parent_manifest_sha256']==sha256(P['manifest'])
    return donor

def get_data(entry):
    p=BANK/entry['path'];assert sha256(p)==entry['sha256']
    data=json.loads(p.read_text());assert not set(data['splits']['train'])&set(data['splits']['test'])
    return data

def slog(rows,train):
    med=statistics.median(train);mad=statistics.median(abs(v-med) for v in train)
    if mad<=1e-12:return None
    import math
    den=statistics.fmean(math.log1p(((r['target']-med)/mad)**2) for r in rows)
    return None if den<=1e-12 else 1-statistics.fmean(math.log1p(((r['target']-r['prediction'])/mad)**2) for r in rows)/den

def prediction(model,data,name,trace):
    x,req=make_input(data,trace,name,'mps',torch.float32)
    with torch.no_grad():o=model(x,req)
    assert len(o.columns)==1 and o.columns[0].result.status=='ok'
    c=o.columns[0];assert c.column==len(data['values'][0])-1
    vals=c.decoded.detach().cpu().tolist();addresses=req.targets.cpu().tolist()
    rows=[]
    for val,pos in zip(vals,c.target_indices.cpu().tolist(),strict=True):
        row,col=addresses[pos];idx=row-len(trace['context_row_ids']);assert idx>=0 and col==c.column
        rid=trace['query_row_ids'][idx];rows.append(dict(row_id=rid,target=float(data['values'][rid][-1]),prediction=val))
    assert len(rows)==len(trace['query_row_ids']) and len({r['row_id'] for r in rows})==len(rows)
    m=metrics(rows);m['slog']=slog(rows,[float(data['values'][i][-1]) for i in data['splits']['train']])
    return m,rows

def fit_eval(model,data,name,traces,folder=None):
    result=[]
    for i,tr in enumerate(traces):
        m,rows=prediction(model,data,name,tr);result.append(m)
        if folder:save(f'{folder}/fit-{i}-predictions.json',rows)
    return dict(windows=result,mean_r2=statistics.fmean(r['r2'] for r in result),
        median_r2=statistics.median(r['r2'] for r in result),mean_slog=statistics.fmean(r['slog'] for r in result if r['slog'] is not None),
        mean_rmse=statistics.fmean(r['rmse'] for r in result),predictions=sum(r['n'] for r in result))

def evaluate(model,entry,traces,arm,node,checkpoint_sha):
    folder=f'evaluations/{arm}/{node}';data=get_data(entry);tick=time.monotonic();before=state_hash(model)
    saved=rng_state();model.eval()
    fit=fit_eval(model,data,entry['name'],traces,folder)
    full=dict(context_row_ids=data['splits']['train'],query_row_ids=data['splits']['test'],codebook_seed=entry['traces'][0]['codebook_seed'])
    test,rows=prediction(model,data,entry['name'],full);save(f'{folder}/test-predictions.json',rows)
    assert state_hash(model)==before;restore_rng(saved)
    report=dict(outcome='completed',arm=arm,node=node,table=entry['name'],checkpoint_sha256=checkpoint_sha,
        fixed_bank_sha256=sha256(B/'fit-bank.json'),test_bank_sha256=sha256(BANK/'bank.json'),fit=fit,test=test,
        model_state_sha256=before,optimizer_updates=0,seconds=time.monotonic()-tick)
    save(f'{folder}/terminal.json',report);torch.mps.empty_cache();return report

def prepare():
    assert not (B/'preflight.json').exists()
    status(outcome='selecting_table',formal_updates=0)
    rt=runtime();donor=load_parent();old=load_v55_plan(P['manifest']);assert donor['model_config']==old.config.as_dict()
    model=make_model(old.config,donor,'linear',E['module_seed']);model.eval()
    bank=json.loads((BANK/'bank.json').read_text());fitbank={};scores={}
    for entry in bank['tables']:
        data=get_data(entry);traces=[]
        for i in range(E['fit_windows']):
            seed=int.from_bytes(hashlib.sha256(f"{E['fit_seed']}/{entry['name']}/{i}".encode()).digest()[:8],'little')
            ids=random.Random(seed).sample(data['splits']['train'],min(512,len(data['splits']['train'])))
            nquery=len(ids)//3
            traces.append(dict(context_row_ids=ids[:-nquery],query_row_ids=ids[-nquery:],codebook_seed=seed))
        fitbank[entry['name']]=traces;scores[entry['name']]=fit_eval(model,data,entry['name'],traces)
        save('selection-progress.json',scores)
    selected=min(scores,key=lambda n:scores[n]['median_r2']);entry=next(t for t in bank['tables'] if t['name']==selected)
    save('fit-bank.json',fitbank)
    selected_table=next(t for t in old.tables if t.name==selected)
    data=get_data(entry);train_data=json.loads(selected_table.path.read_text())
    assert train_data['values']==data['values'] and train_data['splits']['train']==data['splits']['train'] and train_data['splits']['test']==data['splits']['test']
    spec=copy.deepcopy(old.spec);spec['tables']=[t for t in spec['tables'] if t['id']==selected]
    spec['tables'][0]['path']=os.path.relpath(selected_table.path,B);spec['tables'][0]['cohort']='single';spec['tables'][0]['window_rows']=512
    spec['experiment_id']='v6-residual-encoder-single-mini-20260929';stage=spec['stages'][0]
    stage['sampling']=[dict(cohort='single',episodes=1)];stage['question']='Single-table encoder ablation';stage['loss']=dict(state_weights=[0.,1.,0.,0.],discrete_weight=1.)
    save('single-manifest.json',spec)
    cycles,rem=divmod(donor['state']['cursor'],730)
    assert sum(v['episodes'] for v in old.spec['stages'][0]['sampling'])==730
    advance=cycles+sum(schedule_entry(old,0,i)[0].name==selected for i in range(cycles*730,cycles*730+rem))
    offset=donor['state']['table_episode_offsets'][selected]+advance
    save('selection.json',dict(table=selected,entry=entry,ranking=sorted(scores,key=lambda n:scores[n]['median_r2']),scores=scores,
        criterion=E['selection'],episode_start=offset,parent_cursor=donor['state']['cursor'],parent_sha256=P['checkpoint_sha256'],
        training_and_test_values_and_splits_match=True))
    plan=load_v55_plan(B/'single-manifest.json');table=plan.tables[0]
    residual=make_model(plan.config,donor,'residual_gelu',E['module_seed']);residual.eval()
    x,req,truth,info=build_episode(table,stage['recipe'][table.kind],offset,episode_seeds(plan),'mps',epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
    with torch.no_grad():
        qr=RestorationRequest(x.query.nonzero())
        p0=model.prepare(x,qr);p1=residual.prepare(x,qr)
        h0=model.encoder.forward_prepared(x,p0.features);h1=residual.encoder.forward_prepared(x,p1.features)
        assert torch.equal(h0,h1),'zero-init encoder mismatch'
        a=model.forward_prepared(p0);b=residual.forward_prepared(p1)
        assert torch.allclose(a.columns[0].result.encoding,b.columns[0].result.encoding,rtol=1e-5,atol=1e-6)
        maxerr=float((a.columns[0].result.encoding-b.columns[0].result.encoding).abs().max())
    baseline=evaluate(model,entry,fitbank[selected],'shared','before',P['checkpoint_sha256'])
    optimizer=make_optimizer(residual,plan.optimizer,donor);loss_cfg=V53LossConfig(**stage['loss']);gn=[]
    for i in range(2):
        optimizer.zero_grad(set_to_none=True);sc=score_training_episode(residual,x,truth,request=req,loss_config=loss_cfg);sc.loss.backward()
        u=float(residual.encoder.residual_u.weight.grad.norm());v=float(residual.encoder.residual_v.weight.grad.norm());gn.append(dict(u=u,v=v,loss=float(sc.loss.detach())))
        assert finite_state([p.grad for p in residual.parameters() if p.grad is not None]) and u>0
        if i==1:assert v>0
        torch.nn.utils.clip_grad_norm_(residual.parameters(),plan.optimizer.grad_clip,error_if_nonfinite=True);optimizer.step()
    # Verify new-parameter checkpoint/optimizer serialization in memory only.
    import io
    buf=io.BytesIO();torch.save(dict(model=residual.state_dict(),optimizer=optimizer.state_dict()),buf);buf.seek(0)
    restored=torch.load(buf,map_location='cpu',weights_only=False)
    clone=make_model(plan.config,donor,'residual_gelu',E['module_seed']);opt2=make_optimizer(clone,plan.optimizer,donor)
    clone.load_state_dict(restored['model'],strict=True);opt2.load_state_dict(restored['optimizer']);assert state_hash(clone)==state_hash(residual)
    # Nonzero residual remains confined to visible addresses.
    with torch.no_grad():
        layout=residual.prepare(x,req).features
        hh=residual.encoder.forward_prepared(x,layout)
        unchanged=AffineValueEncoder.forward_prepared(residual.encoder,x,layout)
        mask=torch.zeros(hh.shape[:2],device='mps',dtype=torch.bool);mask[:x.visible.shape[0],:x.visible.shape[1]]=x.visible
        assert torch.equal(hh[~mask],unchanged[~mask])
    assert sha256(P['checkpoint'])==P['checkpoint_sha256']
    save('preflight.json',dict(outcome='passed',runtime=rt,selected_table=selected,zero_init_exact_carriers=True,
        output_max_abs_error=maxerr,discarded_updates=2,residual_gradient_norms=gn,checkpoint_roundtrip=True,seeds_and_null_preserved=True,
        original_parent_unchanged=True,formal_updates=0,memory_cap=.75,source_snapshot_sha256=sha256(B/'source-hashes.json')))
    status(outcome='ready',table=selected,formal_updates=0)
    print(json.dumps(dict(outcome='ready',selected=selected,fit=scores[selected],test=baseline['test'])),flush=True)

def report():
    selection=json.loads((B/'selection.json').read_text());rows=[]
    old=B.parent/'v6-nonlinear-encoder-single-table-mini-20260929'
    for root,offset in [(old,0),(B,15)]:
        for path in sorted((root/'evaluations').glob('*/*/terminal.json')):
            q=json.loads(path.read_text());q['cumulative_minutes']=0 if q['node']=='before' else offset+int(q['node'])/60;rows.append(q)
    rows.sort(key=lambda q:(q['cumulative_minutes'],q['arm']))
    save('summary.json',dict(table=selection['table'],evaluations=rows))
    lines=['# Puma encoder 延长对照','',
        '线性与残差非线性分别从自身15分钟终点继续，各追加600秒成功训练；全参数、优化器、RNG及采样cursor延续。',
        '拟合为8个固定训练窗口1360个Query，测试为完整训练支持+1639个测试Query。新增20/25分钟评估，默认点不变。','',
        '|版本|累计分钟|拟合R²|拟合S_log|测试R²|测试S_log|','|---|---:|---:|---:|---:|---:|']
    for q in rows:
        f=q['fit'];t=q['test'];lines.append(f"|{q['arm']}|{q['cumulative_minutes']:g}|{f['mean_r2']:.6f}|{f['mean_slog']:.6f}|{t['r2']:.6f}|{t['slog']:.6f}|")
    (B/'RESULT.md').write_text('\n'.join(lines)+'\n')

def train_arm(arm,donor,plan,selection):
    model=make_model(plan.config,donor,arm,E['module_seed']);opt=make_optimizer(model,plan.optimizer,donor);restore_rng(donor['rng'])
    table=plan.tables[0];stage=plan.spec['stages'][0];seed=episode_seeds(plan);offset=selection['episode_start'];loss_cfg=V53LossConfig(**stage['loss'])
    root=B/arm;root.mkdir(exist_ok=False)
    identity=dict(parent_sha256=P['checkpoint_sha256'],supervision='target_only',objective='single_table_encoder_ablation',arm=arm,
        parent_manifest_sha256=sha256(B/'single-manifest.json'),module=E['residual'] if arm=='residual_gelu' else 'W e',
        strict_resume=False,optimizer_existing_state_preserved=True,new_parameters_fresh=False,code_sha256={p.name:sha256(p) for p in B.glob('*.py')},source_sha256=sha256(B/'source-hashes.json'))
    identity['sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    start_cursor=donor['state']['cursor']
    assert donor['state']['table_episode_offsets'][table.name]==offset
    state=dict(update=donor['state']['update'],parent_update=donor['state']['update'],cursor=start_cursor,successful_update_seconds=0.,table_episode_offsets={table.name:offset})
    totals=[];traces=json.loads((B/'fit-bank.json').read_text())[table.name]
    for threshold in [300,600]:
        status(outcome='training',arm=arm,target_seconds=threshold,completed_arm_seconds=state['successful_update_seconds'])
        while state['successful_update_seconds']<threshold:
            tick=time.monotonic();episode=offset+state['cursor']
            x,req,truth,info=build_episode(table,stage['recipe'][table.kind],episode,seed,'mps',epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
            assert x.visible.shape[0]==min(512,table.train_rows)
            model.train();opt.zero_grad(set_to_none=True);score=score_training_episode(model,x,truth,request=req,loss_config=loss_cfg)
            score.loss.backward();params=[p for p in model.parameters() if p.grad is not None]
            assert finite_state([p.grad for p in params]);norm=torch.nn.utils.clip_grad_norm_(params,plan.optimizer.grad_clip,error_if_nonfinite=True)
            opt.step();torch.mps.synchronize();assert finite_state(model.state_dict()) and finite_state(opt.state_dict())
            sec=time.monotonic()-tick;state['successful_update_seconds']+=sec;state['cursor']+=1;state['update']+=1
            row=dict(arm=arm,table=table.name,update=state['update'],arm_update=state['cursor'],segment_update=state['cursor']-start_cursor,episode=episode,
                seconds=sec,successful_update_seconds=state['successful_update_seconds'],loss=float(score.loss.detach()),gradient_norm=float(norm),
                rows=len(x.visible),query_rows=score.query_rows,scored_cells=score.scored_cells,forward_passes=1,optimizer_steps=1)
            append_event(root/'updates.jsonl',row)
            if state['cursor']==start_cursor+1:save(f'{arm}/first-update.json',row)
            del x,req,truth,score
        cp,digest=checkpoint(root,model,opt,state,identity)
        receipt=dict(outcome='training_node_completed',arm=arm,seconds=state['successful_update_seconds'],updates=state['cursor']-start_cursor,total_arm_updates=state['cursor'],initial_cursor=start_cursor,checkpoint=str(cp),checkpoint_sha256=digest,parent_sha256=P['checkpoint_sha256'])
        save(f'{arm}/node-{threshold}.json',receipt)
        status(outcome='evaluating',arm=arm,node=threshold,checkpoint_sha256=digest)
        evaluate(model,selection['entry'],traces,arm,str(threshold),digest);report();totals.append(receipt)
    save(f'{arm}/terminal.json',dict(outcome='completed',**{k:v for k,v in totals[-1].items() if k!='outcome'}))
    del model,opt;gc.collect();torch.mps.empty_cache()
    return totals[-1]

def run():
    global P
    assert not (B/'linear').exists() and not (B/'residual_gelu').exists()
    assert json.loads((B/'preflight.json').read_text())['outcome']=='passed'
    runtime();plan=load_v55_plan(B/'single-manifest.json');selection=json.loads((B/'selection.json').read_text())
    refs=json.loads((B/'resume-parents.json').read_text());results={}
    for arm in E['arms']:
        P=refs[arm]
        assert sha256(P['checkpoint'])==P['checkpoint_sha256']
        donor=torch.load(P['checkpoint'],map_location='cpu',weights_only=False)
        assert donor['identity']['arm']==arm and donor['model_config']==plan.config.as_dict()
        assert donor['identity']['parent_manifest_sha256']==sha256(B/'single-manifest.json')
        results[arm]=train_arm(arm,donor,plan,selection)
        assert sha256(P['checkpoint'])==P['checkpoint_sha256']
    assert all(600<=r['seconds']<630 for r in results.values())
    report();status(outcome='completed',results=results,defaults_unchanged=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
    try:prepare() if args.mode=='prepare' else run()
    except Exception as ex:status(outcome='failed',mode=args.mode,error_type=type(ex).__name__,error=str(ex));raise
