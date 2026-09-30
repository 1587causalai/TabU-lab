"""Run one bounded joint 618-table stage on DustinStudio, with immutable receipts."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source/src'
BASE=[str(Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-m','tabu_lab.cli','curriculum-v55']
STAGES=((1,'joint618'),)
PARENT=Path('/Users/dustinstudio/experiments/tabu-v55-three-stage-25replay-dustin-20260926-r2/runs/03-tanh100/checkpoint-progress.pt')
PARENT_SHA='be9fcd6406b6bd8a11ef059b9c7f93f086d11a7956548deb4001fb1e1419ac41'
TARGET_SECONDS=7196.0  # leave room for the in-flight update, strictly below 7200
PROBES=('ordinal100_fit','nominal100_fit','tanh100_fit','train_fit','real78_fit','new120_fit')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def start(cmd,log):
    with log.open('x') as f:
        return subprocess.Popen(cmd,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)

def plan(stage):
    p=ROOT/'receipts/plan/plan.json'
    value=json.loads(p.read_text())
    if value['outcome']!='planned_not_run':raise ValueError('plan missing')
    return value

def baseline(attempt,stage,identity):
    target=ROOT/'evaluations'/stage/'step0/terminal.json'
    if target.exists():return True
    source=attempt/'evaluation-000-000000000-initial.json'
    sidecar=attempt/'checkpoint-progress.json'
    if not source.exists() or not sidecar.exists():return False
    event=json.loads(source.read_text());cp=json.loads(sidecar.read_text())
    if event['update']!=0 or cp['update']!=0 or cp['identity']['sha256']!=identity:return False
    if set(event['probes'])!=set(PROBES) or not all(x['complete'] for x in event['probes'].values()):
        raise ValueError('incomplete initial fixed Query bank')
    atomic(target,{'outcome':'completed','checkpoint_update':0,'checkpoint_sha256':cp['sha256'],
                   'identity':cp['identity'],'probes':event['probes'],
                   'source_kind':'runner_initial_projection_no_duplicate_evaluation'})
    return True

def verify_parent_baseline(stage):
    receipt=ROOT/'receipts/parent-u0-match.json'
    if receipt.exists():return
    now=json.loads((ROOT/'evaluations'/stage/'step0/terminal.json').read_text())
    parent=json.loads(Path('/Users/dustinstudio/experiments/tabu-v55-three-stage-25replay-dustin-20260926-r2/evaluations/03-tanh100/endpoint/terminal.json').read_text())
    names=PROBES[:-1]
    checks={name:all(now['probes'][name][part]==parent['probes'][name][part]
                     for part in ('macro','by_table','masks')) for name in names}
    if not all(checks.values()):raise ValueError('parent endpoint and joint u0 fixed Query mismatch: '+str(checks))
    atomic(receipt,{'status':'matched','parent_checkpoint_sha256':PARENT_SHA,
                    'joint_step0_checkpoint_sha256':now['checkpoint_sha256'],
                    'probes':checks,'new120_probe':'new baseline at joint u0 only'})


def write_current(stage,updates,seconds,cohorts,status):
    attempt=ROOT/'runs'/stage
    side=attempt/'checkpoint-progress.json'
    saved=json.loads(side.read_text()) if side.exists() else None
    step0=ROOT/'evaluations'/stage/'step0/terminal.json'
    endpoint=ROOT/'evaluations'/stage/'endpoint/terminal.json'
    evaluated='endpoint' if endpoint.exists() else ('step0' if step0.exists() else 'none')
    line=['# Dustin joint618 120-minute run','',f'Status: {status}',
          f'Stage: {stage}',f'Latest observed update: {updates}',
          f'Successful training seconds: {seconds:.3f} / <7200',
          f'Latest durable checkpoint: update {saved.get("update") if saved else "none"}, SHA {saved.get("sha256") if saved else "none"}',
          f'Latest completed full fixed-Query evaluation: {evaluated}',
          f'Cohort updates: {json.dumps(cohorts,sort_keys=True)}',
          f'Parent: {PARENT} (SHA {PARENT_SHA})',
          f'W&B: https://wandb.ai/zj3712/restoration-v55-single-dgp/runs/v55-dustin-joint618-2a7d192b',
          '']
    target=ROOT/'CURRENT.md'
    tmp=ROOT/'CURRENT.md.tmp'
    tmp.write_text('\n'.join(line));os.replace(tmp,target)


def consume(journal,offset,updates,seconds,cohorts):
    if not journal.exists():return offset,updates,seconds,cohorts
    with journal.open('rb') as f:
        f.seek(offset)
        while True:
            line=f.readline()
            if not line or not line.endswith(b'\n'):break
            row=json.loads(line)
            if row['update']!=updates+1 or row['cohort'] not in cohorts:
                raise ValueError('journal update/cohort discontinuity')
            if row['objective_kind']!='query_log1p_scaled' or row['objective_tau']!=.1:
                raise ValueError('objective drift')
            updates+=1;seconds+=row['seconds'];cohorts[row['cohort']]+=1;offset=f.tell()
    return offset,updates,seconds,cohorts

def checkpoint(attempt,identity,updates):
    term=json.loads((attempt/'terminal.json').read_text())
    side=json.loads((attempt/'checkpoint-progress.json').read_text())
    digest=sha(attempt/'checkpoint-progress.pt')
    if not (term['outcome']=='interrupted' and not term.get('error') and
            term['update']==term['durable_update']==side['update']==updates and
            term['identity']['sha256']==side['identity']['sha256']==identity and
            digest==term['checkpoint_sha256']==side['sha256'] and
            sha(attempt/'checkpoints'/(digest+'.pt'))==digest):
        raise ValueError('terminal checkpoint mismatch')
    return term

def fixed_evaluate(stage,manifest,attempt,identity,term):
    target=ROOT/'evaluations'/stage/'endpoint'
    if target.exists():raise FileExistsError(str(target))
    cmd=BASE+['evaluate','--manifest',str(manifest),'--output-root',str(target),
              '--device','mps','--checkpoint',str(attempt/'checkpoint-progress.pt')]
    for name in PROBES:cmd+=['--probe',name]
    p=start(cmd,ROOT/'receipts'/(stage+'-evaluation.stdout'))
    atomic(ROOT/'receipts'/(stage+'-evaluation-launch.json'),{
        'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':p.pid,'command':cmd,
        'checkpoint_update':term['update'],'checkpoint_sha256':term['checkpoint_sha256']})
    if p.wait()!=0:raise RuntimeError(stage+' endpoint evaluation failed')
    result=json.loads((target/'terminal.json').read_text())
    if not (result['outcome']=='completed' and result['checkpoint_update']==term['update']
            and result['checkpoint_sha256']==term['checkpoint_sha256']
            and result['identity']['sha256']==identity and set(result['probes'])==set(PROBES)
            and all(x['complete'] for x in result['probes'].values())):
        raise ValueError(stage+' evaluation incomplete')

def one(index,main,parent,parent_sha):
    stage=f'{index:02d}-{main}'
    manifest=ROOT/'manifests/joint618-60min-v55.json'
    identity=plan(stage)['identity']['sha256']
    attempt=ROOT/'runs'/stage
    final=ROOT/'receipts'/(stage+'-terminal.json')
    if attempt.exists() or final.exists():raise FileExistsError(stage+' already launched')
    if sha(parent)!=parent_sha:raise ValueError('parent checkpoint drift')
    cmd=BASE+['run','--manifest',str(manifest),'--output-root',str(attempt),
              '--device','mps','--initialize-from',str(parent),
              '--max-updates-this-invocation','24000']
    p=start(cmd,ROOT/'receipts'/(stage+'-training.stdout'))
    atomic(ROOT/'receipts'/(stage+'-launch.json'),{
        'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':p.pid,'command':cmd,
        'identity_sha256':identity,'parent_checkpoint':str(parent),'parent_sha256':parent_sha,
        'initialization':'weights_only_cross_identity','optimizer_rng_cursor':'fresh',
        'actual_training_seconds_stop_target':TARGET_SECONDS,'hard_limit_seconds':7200.0})
    offset=0;updates=0;seconds=0.0;cohorts={c:0 for c in ('ordinal100','nominal100','tanh100','old120','new120','real78')}
    stop=False;last_report=0.0
    try:
        while p.poll() is None:
            if baseline(attempt,stage,identity):verify_parent_baseline(stage)
            offset,updates,seconds,cohorts=consume(attempt/'updates.jsonl',offset,updates,seconds,cohorts)
            if seconds>=TARGET_SECONDS and not stop:
                os.kill(p.pid,signal.SIGTERM);stop=True
                atomic(ROOT/'receipts'/(stage+'-stop-signal.json'),{
                    'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':p.pid,
                    'update':updates,'actual_training_seconds':seconds})
            if time.monotonic()-last_report>=15:
                atomic(ROOT/'receipts/controller-progress.json',{
                    'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'stage':stage,
                    'update':updates,'actual_training_seconds':seconds,'cohort_updates':cohorts,
                    'stop_signal_sent':stop,'pid':p.pid})
                write_current(stage,updates,seconds,cohorts,'training' if updates else 'initial_evaluating')
                last_report=time.monotonic()
            time.sleep(.5)
        p.wait()
        offset,updates,seconds,cohorts=consume(attempt/'updates.jsonl',offset,updates,seconds,cohorts)
        if not stop or not TARGET_SECONDS<=seconds<7200.0:
            raise RuntimeError('training update seconds outside authorized stage window')
        term=checkpoint(attempt,identity,updates)
        if not baseline(attempt,stage,identity):raise ValueError('step0 fixed Query missing')
        verify_parent_baseline(stage)
        atomic(final,{'status':'training_window_completed_evaluation_pending',
            'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'stage':stage,'update':updates,
            'checkpoint_sha256':term['checkpoint_sha256'],'actual_training_seconds':seconds,
            'cohort_updates':cohorts})
        fixed_evaluate(stage,manifest,attempt,identity,term)
        atomic(final,{'status':'completed','utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'stage':stage,'update':updates,'checkpoint_sha256':term['checkpoint_sha256'],
            'actual_training_seconds':seconds,'cohort_updates':cohorts,
            'step0_evaluation':str(ROOT/'evaluations'/stage/'step0/terminal.json'),
            'endpoint_evaluation':str(ROOT/'evaluations'/stage/'endpoint/terminal.json')})
        write_current(stage,updates,seconds,cohorts,'completed')
        return attempt/'checkpoint-progress.pt',term['checkpoint_sha256'],seconds
    except BaseException as error:
        if p.poll() is None:
            os.kill(p.pid,signal.SIGTERM);p.wait(timeout=180)
        write_current(stage,updates,seconds,cohorts,'failed')
        atomic(final,{'status':'failed','utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'stage':stage,'update':updates,'actual_training_seconds':seconds,
            'error_type':type(error).__name__,'error':str(error)[:1000]})
        raise

def main():
    if sha(PARENT)!=PARENT_SHA:raise ValueError('completed r2 parent drift')
    first=json.loads((ROOT/'receipts/preflight/terminal.json').read_text())
    if first['outcome']!='passed' or first['runtime']['device']!='mps' or first['runtime']['mps_cpu_fallback']:
        raise ValueError('joint MPS preflight not passed')
    parent=PARENT;digest=PARENT_SHA;total=0.0
    for index,main_cohort in STAGES:
        parent,digest,seconds=one(index,main_cohort,parent,digest)
        total+=seconds
    if total>=7200.0:raise ValueError('joint actual update time exceeded cap')
    atomic(ROOT/'receipts/controller-terminal.json',{
        'status':'completed','utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'actual_training_seconds_total':total,'latest_checkpoint':str(parent),
        'latest_checkpoint_sha256':digest,'stages':[f'{i:02d}-{c}' for i,c in STAGES]})

if __name__=='__main__':main()
