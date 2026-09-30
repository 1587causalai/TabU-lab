"""One authorized mixed stage, successful-update budget, terminal evaluation."""
import collections, datetime as dt, fcntl, json, math, os, signal, subprocess, time
from pathlib import Path
from common import ROOT,PARENT,PARENT_SHA,PARENT_EVAL,MANIFEST,URL,atomic,sha,rows,probe_digest,paired

BASE=[str(Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-m','tabu_lab.cli','curriculum-v55']
GATE=ROOT/'evidence/execution'
FIRST=ROOT/'runs/attempt-001-initial'
MAIN=ROOT/'runs/attempt-002-main'
FINAL=ROOT/'evaluations/final'
INITIAL=FIRST/'evaluation-000-000000000-initial.json'
BUDGET=json.loads((ROOT/'budget.json').read_text())
TARGET=BUDGET['successful_training_seconds_target']
MAXIMUM=BUDGET['successful_training_seconds_maximum']
assert 7195<=TARGET<=7198 and MAXIMUM==7200

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def checkpoint(run):
    p=run/'checkpoint-progress.pt';s=json.loads(p.with_suffix('.json').read_text());t=json.loads((run/'terminal.json').read_text());digest=sha(p)
    assert s['sha256']==t['checkpoint_sha256']==digest
    assert s['update']==t['update']==t['durable_update'] and not t.get('error')
    return {'path':str(p),'sha256':digest,'update':s['update'],'terminal_outcome':t['outcome']}

def main():
    GATE.mkdir(parents=True,exist_ok=True)
    lock=(GATE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (GATE/'controller.json').exists()
    admission=json.loads((ROOT/'evidence/qualification/preflight.json').read_text())
    assert admission['outcome']=='passed' and admission['parent_checkpoint_sha256']==PARENT_SHA and sha(PARENT)==PARENT_SHA
    inventory=json.loads((ROOT/'evidence/inventory.json').read_text());ids={x['id'] for x in inventory['tables']}
    prior=json.loads(PARENT_EVAL.read_text());old_names=sorted(prior['probes'])
    parent={'path':str(PARENT),'sha256':PARENT_SHA,'update':6656,'evaluation':str(PARENT_EVAL)}
    result={'schema':'tabu.joint618.controller.v1','started_utc':now(),'status':'starting','parent':parent,'identity_sha256':admission['identity']['sha256'],'budget':BUDGET,'budget_sha256':sha(ROOT/'budget.json'),'manifest_sha256':sha(MANIFEST),'table_count':618,'mask_count':1238,'cohorts':['ordinal100','nominal100','tanh100','old120','new120','real78'],'attempts':[],'monitor_url':URL,'latest_saved_checkpoint':None,'latest_completed_evaluation':{'path':str(PARENT_EVAL),'checkpoint_sha256':PARENT_SHA,'checkpoint_update':6656,'scope':'parent five probes; no joint618 new120 baseline yet'},'successful_training_seconds':0.0}
    previous_save=0
    def save(status,force=True):
        nonlocal previous_save
        result.update(status=status,updated_utc=now(),active_stage='joint618' if status not in ('completed','failed') else None)
        if not force and time.monotonic()-previous_save<3:return
        previous_save=time.monotonic();atomic(GATE/'controller.json',result);atomic(ROOT/'CURRENT.json',result)
        cp=result['latest_saved_checkpoint'];ev=result['latest_completed_evaluation']
        text=f"# DGX2 joint618：{status}\n\n更新时间：{result['updated_utc']}。目录名保留60min，本轮实际授权与控制器上限为7200秒（120分钟）。\n\n- 当前阶段：{result['active_stage']}；成功参数更新：{result.get('update',0)} 步，{result['successful_training_seconds']:.6f} / 7200 秒。评估与保存另计。\n- 最新保存点：{json.dumps(cp,ensure_ascii=False)}\n- 最新完整评估：{json.dumps(ev,ensure_ascii=False)}\n- 六组618表等权，每个完整周期每表一次。无主任务/回放比例。\n- 监控：{URL}\n- 详细回执：`{GATE/'controller.json'}`。\n"
        if result.get('error'):text+='\n错误：'+result['error']+'\n'
        temp=ROOT/'CURRENT.md.tmp';temp.write_text(text);os.replace(temp,ROOT/'CURRENT.md')
    all_rows=[]
    def exposure():
        result['successful_training_seconds']=sum(r['seconds'] for r in all_rows)
        result['update']=len(all_rows)
        result['cohort_exposure']=dict(collections.Counter(r['cohort'] for r in all_rows))
        counter=collections.Counter(r['table'] for r in all_rows)
        result['table_exposure']={k:counter[k] for k in sorted(ids)}
        result['complete_joint_cycles']=min(result['table_exposure'].values())
    def run_attempt(run,initial,parent):
        cmd=BASE+['run','--manifest',str(MANIFEST),'--output-root',str(run),'--device','cuda:0','--max-updates-this-invocation','1' if initial else '50000', '--initialize-from' if initial else '--resume-checkpoint',str(parent)]
        attempt={'path':str(run),'mode':'weights_only_initialization' if initial else 'strict_resume','command':cmd,'started_utc':now()};result['attempts'].append(attempt)
        p=None;offset=0;pending=b'';stop_sent=False;seen_saved=None
        log=(GATE/(run.name+'.stdout.log')).open('x')
        try:
            p=subprocess.Popen(cmd,cwd=ROOT/'source/src',stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            attempt['pid']=p.pid;save('initial_evaluating' if initial else 'training')
            while p.poll() is None:
                journal=run/'updates.jsonl'
                if journal.exists():
                    with journal.open('rb') as f:f.seek(offset);new=f.read();offset=f.tell()
                    lines=(pending+new).split(b'\n');pending=lines.pop()
                    for line in lines:
                        row=json.loads(line);assert row['update']==len(all_rows)+1
                        assert row['table'] in ids and all(math.isfinite(row[k]) for k in ('loss','objective_loss','gradient_norm','seconds'))
                        all_rows.append(row)
                    exposure()
                sidecar=run/'checkpoint-progress.json'
                if sidecar.exists():
                    side=json.loads(sidecar.read_text())
                    if side['sha256']!=seen_saved:
                        seen_saved=side['sha256'];result['latest_saved_checkpoint']={'path':str(run/'checkpoint-progress.pt'),'sha256':side['sha256'],'update':side['update']}
                    if initial and side['update']==0 and 'initial_checkpoint' not in result:
                        generation=run/'checkpoints'/f"{side['sha256']}.pt"
                        assert sha(generation)==side['sha256']
                        result['initial_checkpoint']={'path':str(generation),'sha256':side['sha256'],'update':0,'source_checkpoint_sha256':PARENT_SHA}
                save('initial_evaluating' if initial and not all_rows else 'training',force=False)
                if result['successful_training_seconds']>=TARGET and not stop_sent:
                    p.send_signal(signal.SIGTERM);stop_sent=True;attempt['stop_reason']='authorized_successful_training_seconds'
                time.sleep(.25)
            attempt.update(returncode=p.returncode,finished_utc=now())
            # Read the final journal tail after the child has flushed its terminal.
            existing=len(all_rows);all_rows[:]=rows(FIRST/'updates.jsonl')+([] if run==FIRST else rows(MAIN/'updates.jsonl'));exposure()
            assert len(all_rows)>=existing
            assert p.returncode==(0 if initial else 3)
            cp=checkpoint(run);assert cp['terminal_outcome']==('stopped' if initial else 'interrupted')
            if not initial:assert stop_sent
            attempt['checkpoint']=cp;result['latest_saved_checkpoint']=cp
            return cp
        finally:
            if p is not None and p.poll() is None:
                p.send_signal(signal.SIGTERM)
                try:p.wait(timeout=120)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()
    try:
        save('starting')
        first=run_attempt(FIRST,True,PARENT)
        initial=json.loads(INITIAL.read_text())
        assert len(initial['probes'])==6 and all(v['complete'] for v in initial['probes'].values())
        assert probe_digest(initial,old_names)==probe_digest(prior,old_names)
        result['initial_parity']={'passed':True,'old_five_probe_content_sha256':probe_digest(initial,old_names),'source_checkpoint_sha256':PARENT_SHA,'initial_update':0,'initial_evaluation':str(INITIAL)}
        assert result['initial_checkpoint']['update']==0 and sha(result['initial_checkpoint']['path'])==result['initial_checkpoint']['sha256']
        result['initial_parity']['initial_checkpoint']=result['initial_checkpoint']
        result['latest_completed_evaluation']={'path':str(INITIAL),'checkpoint_update':0,'source_checkpoint_sha256':PARENT_SHA,'checkpoint_sha256':result['initial_checkpoint']['sha256'],'scope':'six complete probes at u0; immutable pre-update generation captured during initial evaluation'}
        save('initial_qualified')
        final=run_attempt(MAIN,False,first['path'])
        assert TARGET<=result['successful_training_seconds']<=MAXIMUM
        assert result['update']==final['update'] and len(all_rows)==len({r['update'] for r in all_rows})
        assert max(result['table_exposure'].values())-min(result['table_exposure'].values())<=1
        save('final_evaluating')
        cmd=BASE+['evaluate','--manifest',str(MANIFEST),'--output-root',str(FINAL),'--device','cuda:0','--checkpoint',final['path']]
        result['final_evaluation_command']=cmd;start=time.monotonic()
        with (GATE/'final-evaluation.stdout.log').open('x') as f:
            p=subprocess.Popen(cmd,cwd=ROOT/'source/src',stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);result['evaluation_pid']=p.pid;save('final_evaluating');code=p.wait()
        assert code==0
        end=json.loads((FINAL/'terminal.json').read_text());assert end['outcome']=='completed' and end['checkpoint_sha256']==final['sha256'] and end['checkpoint_update']==final['update']
        assert len(end['probes'])==6 and all(v['complete'] for v in end['probes'].values())
        result['latest_completed_evaluation']={'path':str(FINAL/'terminal.json'),'checkpoint_sha256':final['sha256'],'checkpoint_update':final['update'],'scope':'all six fixed probes complete'}
        result['final_evaluation_seconds']=time.monotonic()-start
        processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
        residual=[line for line in processes if str(ROOT) in line and 'tabu_lab.cli' in line]
        assert not residual,residual
        result['process_audit']={'utc':now(),'remaining_training_or_evaluation_processes':residual,'final_evaluation_returncode':code}
        result['paired_metrics']=paired(initial,end);result['budget_completed']=True;result['finished_utc']=now();result['child_processes_exited']=True
        save('completed');atomic(GATE/'completion.json',result)
        report=['# DGX2 joint618 两小时结果','',f"状态 completed；成功更新 {result['successful_training_seconds']:.6f} 秒，{result['update']} 步；每表曝光 {min(result['table_exposure'].values())}–{max(result['table_exposure'].values())} 次。评估与保存另计。",'','| Probe / type / metric | 起点 mean / median | 终点 mean / median | 改善 / 持平 / 下降 |','|---|---:|---:|---:|']
        for key,v in result['paired_metrics'].items():report.append(f"| {key} | {v['start']['mean']:.6f} / {v['start']['median']:.6f} | {v['final']['mean']:.6f} / {v['final']['median']:.6f} | {' / '.join(map(str,v['better_tie_worse']))} |")
        report+=['','固定 Query 为已知训练表拟合；new120 只比较本轮起终点，不与不同 bank 历史分数直接比较。',f"最终 checkpoint：{final['path']}，SHA {final['sha256']}，u{final['update']}。",'',f'监控：{URL}']
        (ROOT/'RESULT.md').write_text('\n'.join(report)+'\n')
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error),finished_utc=now());save('failed');raise

if __name__=='__main__':main()
