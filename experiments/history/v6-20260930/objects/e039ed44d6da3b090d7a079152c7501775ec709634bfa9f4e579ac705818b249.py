"""Bounded continuation, preserving per-host lineage and hourly evidence."""
import pathlib,json,os,subprocess,time
from datetime import datetime,timezone
B=pathlib.Path(__file__).resolve().parent
def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
def execute(script,args,log):
    env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
    cmd=[str(pathlib.Path.home()/'.local/bin/wehub-python'),'--profile','train-20260920','-u','-c',
         f"import runpy;runpy.run_path({str(B/script)!r},run_name='__main__')",*map(str,args)]
    with (B/log).open('w') as f:
        subprocess.run(cmd,cwd=B/'source/src',env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=21600)
def fit(q,donor,phase):
    execute('eval_fit.py',['--host',q['host'],'--device',q['device'],
        '--checkpoint',donor['checkpoint'],'--expected-sha',donor['checkpoint_sha256'],
        '--output',B/'evaluations'/phase/'fit718'],phase+'-fit.log')
    ev=json.loads((B/'evaluations'/phase/'fit718/terminal.json').read_text())
    assert ev['outcome']=='completed' and ev['optimizer_updates']==0
    assert ev['total_masks']==1436 and ev['total_predictions_per_branch']==97648
    assert ev['checkpoint_sha256']==donor['checkpoint_sha256']
    return ev
def main():
    q=json.loads((B/'queue.json').read_text());h=q['host'];old=pathlib.Path(q['parent_root'])
    s=dict(outcome='waiting_parent',host=h,supervision=q['supervision'],parent_root=str(old),
        total_successful_seconds_target=3600,completed_training_seconds=0,
        started_utc=datetime.now(timezone.utc).isoformat())
    save(B/'status.json',s)
    try:
        while True:
            prior=json.loads((old/'status.json').read_text())
            if prior['outcome']=='failed':raise RuntimeError('Parent30m failed; extension not started')
            if prior['outcome']=='completed':break
            time.sleep(20)
        donor=json.loads((old/'half2/run/campaign.json').read_text())
        before=json.loads((old/'evaluations/half2/openml12/terminal.json').read_text())
        baseline=json.loads((old/'evaluations/half2/fit718/terminal.json').read_text())
        prior_seconds=prior['completed_training_seconds'];extension_budget=3600-prior_seconds
        assert 1800<=prior_seconds<1920 and 0<extension_budget<=1800
        assert donor['outcome']=='training_completed' and before['outcome']==baseline['outcome']=='completed'
        assert before['total_predictions']==7894 and baseline['total_masks']==1436 and baseline['total_predictions_per_branch']==97648
        assert before['optimizer_updates']==baseline['optimizer_updates']==0
        assert donor['checkpoint_sha256']==before['checkpoint_sha256']==baseline['checkpoint_sha256']==prior['checkpoint_sha256']
        assert donor['supervision']=='target_only' and donor['identity']['mixing']['kind']=='random_two_nontarget_columns'
        for name in ['before-eval','before-fit-eval']:
            save(B/('mix-start-'+name.replace('before-','')+'.json'),json.loads((old/(name+'.json')).read_text()))
        save(B/'before-eval.json',before);save(B/'before-fit-eval.json',baseline)
        save(B/'resolved-parent.json',dict(checkpoint=donor['checkpoint'],checkpoint_sha256=donor['checkpoint_sha256'],manifest=donor['balanced_manifest'],prior_training_seconds=prior_seconds))
        s.update(outcome='starting',start_sha256=donor['checkpoint_sha256'],prior_training_seconds=prior_seconds,additional_seconds_target=extension_budget)
        save(B/'status.json',s)
        for i in range(1,3):
            phase=f'half{i}';total_target=2700 if i==1 else 3600
            budget=total_target-prior_seconds-s['completed_training_seconds'];assert budget>0
            s.update(outcome='training',phase=phase,phase_budget=budget);save(B/'status.json',s)
            execute('train.py',['--host',h,'--device',q['device'],'--parent',donor['checkpoint'],
                '--parent-sha',donor['checkpoint_sha256'],'--manifest',donor['balanced_manifest'],
                '--supervision',q['supervision'],'--segment',phase,'--seconds',budget],phase+'-train.log')
            c=json.loads((B/phase/'run/campaign.json').read_text())
            assert c['outcome']=='training_completed' and c['parent_sha256']==donor['checkpoint_sha256']
            s['completed_training_seconds']+=c['successful_update_seconds']
            s['total_completed_training_seconds']=prior_seconds+s['completed_training_seconds']
            donor=c;s.update(outcome='evaluating',checkpoint_sha256=c['checkpoint_sha256']);save(B/'status.json',s)
            execute('evaluate.py',['--manifest',c['balanced_manifest'],'--checkpoint',c['checkpoint'],
                '--expected-sha',c['checkpoint_sha256'],'--device',q['device'],
                '--bank-root',B.parent/'openml12-frozen-icl-20260927','--output',B/'evaluations'/phase/'openml12',
                '--memory-fraction',.75 if h=='gongqian-mini' else .5],phase+'-openml.log')
            ev=json.loads((B/'evaluations'/phase/'openml12/terminal.json').read_text())
            assert ev['outcome']=='completed' and ev['total_predictions']==7894 and ev['optimizer_updates']==0
            assert ev['checkpoint_sha256']==c['checkpoint_sha256'] and ev['bank_sha256']==before['bank_sha256']
            f=fit(q,c,phase);assert f['bank_sha256']==baseline['bank_sha256']
            s.setdefault('evaluations',{})[phase]=dict(openml12=ev['macro'],fit718=f['by_family'],checkpoint_sha256=c['checkpoint_sha256'])
            save(B/'status.json',s)
        assert 3600<=s['total_completed_training_seconds']<3720
        s.update(outcome='completed',completed_utc=datetime.now(timezone.utc).isoformat());save(B/'status.json',s)
    except Exception as ex:
        s.update(outcome='failed',error_type=type(ex).__name__,error=str(ex));save(B/'status.json',s);raise
if __name__=='__main__':main()
