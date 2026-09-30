"""Regenerate concise and per-table results from completed local receipts."""
import json,os
from pathlib import Path
B=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text()) if p.exists() else None
def write(p,data):
    tmp=p.with_suffix('.tmp');tmp.write_text(data);os.replace(tmp,p)
def fmt(x):return '未定义' if x is None else f'{x:.6f}'
lines=['# 广播后全 cell restoration：30 分钟结果','',
       '全 train support + 全 test Query，7894预测；R²/S_log 越大越好。',
       '按各机自己的父点比较；本轮未额外评估旧618/合成100留存。','']
summary={}
for h in ['dustinstudio','dgx2','gongqian-mini']:
    root=B/'monitor'/h
    before=read(root/'before-eval.json') or read(B/'baselines'/h/'before-eval.json')
    controller=read(root/'controller.json') or {}
    s=dict(controller=controller,evaluations={});summary[h]=s
    lines += [f'## {h}','',f"控制器状态：{controller.get('outcome','未同步')}。",'',
              '| 节点 | 宏R² | ΔR²/父点 | S_log | ΔS_log/父点 |',
              '|---|---:|---:|---:|---:|']
    vals={'before':before}
    for phase in ['half1','half2']:
        ev=read(root/f'{phase}-eval.json')
        if not ev or ev.get('outcome')!='completed':continue
        campaign=read(root/f'{phase}-train.json')
        assert ev['total_predictions']==7894 and ev['optimizer_updates']==0
        assert ev['bank_sha256']==before['bank_sha256']
        assert ev['checkpoint_sha256']==campaign['checkpoint_sha256']
        assert campaign['outcome']=='training_completed'
        assert campaign['identity']['objective']=='broadcast_allcell_restoration'
        vals[phase]=ev
        deltas={k:ev['macro'][k]-before['macro'][k] for k in ('r2','slog')}
        counts={k:dict(improved=sum(ev['tables'][t][k]>before['tables'][t][k] for t in ev['tables']),
                       declined=sum(ev['tables'][t][k]<before['tables'][t][k] for t in ev['tables'])) for k in ('r2','slog')}
        s['evaluations'][phase]=dict(macro=ev['macro'],delta=deltas,counts=counts,
            successful_seconds=campaign['successful_update_seconds'],updates=sum(campaign['table_updates'].values()))
    for phase,ev in vals.items():
        m=ev['macro'];dm={k:m[k]-before['macro'][k] for k in ('r2','slog')}
        lines.append(f"| {phase} | {fmt(m['r2'])} | {fmt(dm['r2'])} | {fmt(m['slog'])} | {fmt(dm['slog'])} |")
    lines+=['','| 表 | 父点 R²/S_log | 15分钟 R²/S_log | 30分钟 R²/S_log |','|---|---:|---:|---:|']
    for t in before['tables']:
        cells=[]
        for phase in ['before','half1','half2']:
            m=vals[phase]['tables'][t] if phase in vals else None
            cells.append(f"{fmt(m['r2'])} / {fmt(m['slog'])}" if m else '待评估')
        lines.append('| '+t+' | '+' | '.join(cells)+' |')
    lines.append('')
write(B/'RESULT.md','\n'.join(lines)+'\n')
complete=all(s['controller'].get('outcome')=='completed' and 'half2' in s['evaluations'] for s in summary.values())
write(B/'summary.json',json.dumps(dict(all_complete=complete,hosts=summary),indent=2)+'\n')
print(json.dumps(dict(all_complete=complete,phases={h:list(s['evaluations']) for h,s in summary.items()})))
