"""Generate macro and full per-table reports; pending stages stay explicit."""
import json,os
from pathlib import Path
B=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text()) if p.exists() else None
def save(p,s):
    t=p.with_suffix('.tmp');t.write_text(s);os.replace(t,p)
def fmt(x):return '—' if x is None else f'{x:.5f}'
summary={};lines=['# 随机混合两列：全730表总计60分钟','',
    'OpenML12为固定测试；old618/sparse100为固定训练行Query拟合，分开报告。original为原父点无混合；before为同一父点开启混合的零更新评估，15/30/45/60分钟均为混合开启，before仍指整个实验最初的混合零更新起点。','']
detail=['# 逐表结果','']
for h in ['dustinstudio','dgx2','gongqian-mini']:
    d=B/'monitor'/h;c=read(d/'controller.json') or {}
    old=B.parent/'v6-randommix2-all730-30m-20260929'/'monitor'/h
    before=read(d/'mix-start-eval.json') or read(old/'before-eval.json')
    before_fit=read(d/'mix-start-fit-eval.json') or read(old/'before-fit-eval.json');entry=dict(controller=c,phases={});summary[h]=entry
    lines += [f'## {h}','',f"状态：{c.get('outcome','未启动')}。",'',
              '| 节点 | OpenML R² | S_log | ΔR² vs mix-before | ΔS_log vs mix-before | ΔR² vs original | ΔS_log vs original |','|---|---:|---:|---:|---:|---:|---:|']
    panels={}
    original=read(d/'original-parent-eval.json') or read(old/'original-parent-eval.json');original_fit=read(d/'original-parent-fit-eval.json') or read(old/'original-parent-fit-eval.json')
    if original:panels['original']={'openml12':original,'fit718':original_fit}
    if before:panels['before']={'openml12':before,'fit718':before_fit}
    for phase,src,segment in [('15min',old,'half1'),('30min',old,'half2'),('45min',d,'half1'),('60min',d,'half2')]:
        o=read(src/(segment+'-openml12-eval.json'));f=read(src/(segment+'-fit718-eval.json'));train=read(src/(segment+'-train.json'))
        if not (o and f and o.get('outcome')==f.get('outcome')=='completed'):continue
        assert o['total_predictions']==7894 and f['total_predictions_per_branch']==97648 and f['total_masks']==1436
        assert o['optimizer_updates']==f['optimizer_updates']==0
        assert train['outcome']=='training_completed' and train['checkpoint_sha256']==o['checkpoint_sha256']==f['checkpoint_sha256']
        assert before and before_fit and o['bank_sha256']==before['bank_sha256'] and f['bank_sha256']==before_fit['bank_sha256']
        panels[phase]={'openml12':o,'fit718':f}
        entry['phases'][phase]=dict(openml12=o['macro'],fit718=f['by_family'],successful_seconds=train['successful_update_seconds'],
            updates=sum(train['table_updates'].values()),coverage=train['table_coverage'],checkpoint_sha256=train['checkpoint_sha256'])
    for phase,evs in panels.items():
        m=evs['openml12']['macro'];base=(before or original)['macro']
        lines.append(f"| {phase} | {fmt(m['r2'])} | {fmt(m['slog'])} | {fmt(m['r2']-base['r2'])} | {fmt(m['slog']-base['slog'])} | {fmt(m['r2']-original['macro']['r2'])} | {fmt(m['slog']-original['macro']['slog'])} |")
    for family in ['old618','sparse100']:
        lines+=['',f'### {family} 固定训练行Query拟合','',
                '| 节点 | R² | S_log | nominal accuracy | ordinal accuracy | rank MAE | Query loss |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for phase,evs in panels.items():
            if not evs['fit718']:continue
            m=evs['fit718']['by_family'][family]
            lines.append('| '+phase+' | '+' | '.join(fmt(m.get('v6/'+k)) for k in ['r2','slog','nominal_accuracy','ordinal_accuracy','rank_mae','loss'])+' |')
    detail += [f'## {h}','']
    for phase,evs in panels.items():
        detail += [f'### {phase}','', '| 数据组/表 | R² | S_log | accuracy | rank MAE | Query loss |','|---|---:|---:|---:|---:|---:|']
        for t,m in evs['openml12']['tables'].items():
            detail.append('| OpenML12/'+t+' | '+' | '.join(fmt(m.get(k)) for k in ['r2','slog','accuracy','rank_mae','loss'])+' |')
        if evs['fit718']:
            for t,row in evs['fit718']['tables'].items():
                m=row['v6'];detail.append('| '+row['family']+'/'+t+' | '+' | '.join(fmt(m.get(k)) for k in ['r2','slog','accuracy','rank_mae','loss'])+' |')
        detail.append('')
    lines.append('')
complete=all(s['controller'].get('outcome')=='completed' and '60min' in s['phases'] for s in summary.values())
save(B/'RESULT.md','\n'.join(lines)+'\n');save(B/'DETAIL.md','\n'.join(detail)+'\n')
save(B/'summary.json',json.dumps(dict(all_complete=complete,hosts=summary),indent=2)+'\n')
print(json.dumps(dict(all_complete=complete,phases={h:list(s['phases']) for h,s in summary.items()})))
