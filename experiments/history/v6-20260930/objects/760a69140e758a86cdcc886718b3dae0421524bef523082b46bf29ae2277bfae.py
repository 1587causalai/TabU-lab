import json
from pathlib import Path
B=Path(__file__).resolve().parent
summary=dict(hosts={})
lines=['# Puma 学习率与监控选点','',
    '每档同一父点、15分钟成功训练；5242行更新，1311行监控选点。监控行曾被父点训练过。本轮没有选点后的6553行全量重训，须与基线流程区别；不按测试选节点或学习率。','']
for h in ['dustinstudio','dgx2']:
    p=B/'monitor'/h/'status.json'
    if not p.exists():continue
    s=json.loads(p.read_text());summary['hosts'][h]=s
    lines += [f'## {h} / {s["branch"]}', '', f'状态：{s["outcome"]}；监控选择主候选：{s.get("selected_trial","待完成")}。', '',
        '| 学习率档 | 成功秒数 | 更新次数 | 选中节点秒数 | 监控R² | 原6553行恢复R² | 测试R² |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for tag,t in s['trials'].items():
        selected=t.get('selected',{});ev=t.get('final_evaluations',{})
        def num(x):return '待完成' if x is None else f'{x:.5f}'
        lines.append(f'| {tag} | {t["successful_update_seconds"]:.2f} | {t["updates"]} | {t.get("selected_node","待完成")} | {num(selected.get("metrics",{}).get("r2"))} | {num(ev.get("train",{}).get("metrics",{}).get("r2"))} | {num(ev.get("test",{}).get("metrics",{}).get("r2"))} |')
    lines += ['', '| 学习率档/评分集 | R² | S_log | MSE | 预测数 |', '|---|---:|---:|---:|---:|']
    for tag,t in s['trials'].items():
        for group,ev in t.get('final_evaluations',{}).items():
            m=ev['metrics'];lines.append(f'| {tag}/{group} | {m["r2"]:.5f} | {m["slog"]:.5f} | {m["mse"]:.9f} | {ev["predictions"]} |')
    lines += ['', '| 学习率档/节点 | 监控R² | 监控S_log | 监控MSE |', '|---|---:|---:|---:|']
    for tag,t in s['trials'].items():
        for node,v in t['nodes'].items():
            m=v['metrics'];lines.append(f'| {tag}/{node}s | {m["r2"]:.5f} | {m["slog"]:.5f} | {m["mse"]:.9f} |')
    lines.append('')
summary['all_complete']=len(summary['hosts'])==2 and all(s['outcome']=='completed' for s in summary['hosts'].values())
lines += ['原基线早停XGBoost depth6测试R²0.64171。TabU原完整6553行单表训练：Dustin5分钟0.64754、15分钟0.62964；DGX2 15分钟0.62271。本轮新更新只有5242行，故不是严格同训练数据量对照。各主机历史/分支/精度不同，不作纯损失因果归因，单切分微小差距不等于稳定优越。','']
(B/'RESULT.md').write_text('\n'.join(lines))
(B/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(all_complete=summary['all_complete'],hosts={h:s['outcome'] for h,s in summary['hosts'].items()})))
