"""Generate macro and full per-table reports; pending stages stay explicit."""
import json,os
from pathlib import Path
B=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text()) if p.exists() else None
def save(p,s):
    t=p.with_suffix('.tmp');t.write_text(s);os.replace(t,p)
def fmt(x):return '—' if x is None else f'{x:.5f}'
summary={};lines=['# Dustin全730表恢复target_only，1小时结果','',
    'OpenML12为固定测试；old618/sparse100为固定训练行Query拟合，分开报告。所有节点统一V6广播target_only推理。','']
detail=['# 逐表结果','']
for h in ['dustinstudio']:
    d=B/'monitor'/h;c=read(d/'controller.json') or {}
    before=read(d/'before-eval.json') or read(B/'baselines'/h/'before-eval.json')
    before_fit=read(d/'before-fit-eval.json');entry=dict(controller=c,phases={});summary[h]=entry
    lines += [f'## {h}','',f"状态：{c.get('outcome','未启动')}。",'',
              '| 节点 | OpenML R² | S_log | ΔR² | ΔS_log |','|---|---:|---:|---:|---:|']
    panels={}
    if before:panels['before']={'openml12':before,'fit718':before_fit}
    for i in range(1,3):
        phase=f'half{i}';o=read(d/(phase+'-openml12-eval.json'));f=read(d/(phase+'-fit718-eval.json'));train=read(d/(phase+'-train.json'))
        if not (o and f and o.get('outcome')==f.get('outcome')=='completed'):continue
        assert o['total_predictions']==7894 and f['total_predictions_per_branch']==97648 and f['total_masks']==1436
        assert o['optimizer_updates']==f['optimizer_updates']==0
        assert train['outcome']=='training_completed' and train['checkpoint_sha256']==o['checkpoint_sha256']==f['checkpoint_sha256']
        assert before and before_fit and o['bank_sha256']==before['bank_sha256'] and f['bank_sha256']==before_fit['bank_sha256']
        panels[phase]={'openml12':o,'fit718':f}
        entry['phases'][phase]=dict(openml12=o['macro'],fit718=f['by_family'],successful_seconds=train['successful_update_seconds'],
            updates=sum(train['table_updates'].values()),coverage=train['table_coverage'],checkpoint_sha256=train['checkpoint_sha256'])
    for phase,evs in panels.items():
        m=evs['openml12']['macro'];base=before['macro']
        lines.append(f"| {phase} | {fmt(m['r2'])} | {fmt(m['slog'])} | {fmt(m['r2']-base['r2'])} | {fmt(m['slog']-base['slog'])} |")
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
complete=all(s['controller'].get('outcome')=='completed' and 'half2' in s['phases'] for s in summary.values())
save(B/'RESULT.md','\n'.join(lines)+'\n');save(B/'DETAIL.md','\n'.join(detail)+'\n')
save(B/'summary.json',json.dumps(dict(all_complete=complete,hosts=summary),indent=2)+'\n')
print(json.dumps(dict(all_complete=complete,phases={h:list(s['phases']) for h,s in summary.items()})))
