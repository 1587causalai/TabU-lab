"""Read-only local-mirror reporting; never launches any remote job."""
import json
from pathlib import Path
B=Path(__file__).resolve().parent
lines=['# Puma 单表拟合：Dustin V6 / DGX2 V5.5，15分钟','',
'TabU从各自默认检查点出发，仅Puma、无回放。原默认点和代码保留。训练拟合覆盖6553行、分三批隐藏Query标签；测试为6553 train support + 1639 test Query。','',
'两机本次模型结构相同，但训练起点、精度和单位时间更新数不同；不能将两机差距因果归于损失。MLP/XGBoost无预训练，拟合相同全部训练行，预设候选、不按测试择优；预算/历史不同，不是严格等算力模型竞赛。','']
summary={'hosts':{}}
for host in ['dustinstudio','dgx2']:
    root=B/'monitor'/host
    if not (root/'status.json').exists():continue
    s=json.loads((root/'status.json').read_text());summary['hosts'][host]=s
    lines += [f'## {host} / {s["branch"]}', '',f'状态：{s["outcome"]}；实际成功训练 {s.get("successful_update_seconds",0):.2f} 秒；更新 {s.get("new_updates",0)} 次。','',
    '| 节点 | 训练R² | 训练S_log | 训练MSE | 测试R² | 测试S_log | 测试MSE |','|---|---:|---:|---:|---:|---:|---:|']
    for node in ['before','minute05','minute10','minute15']:
        ev=s.get('evaluations',{}).get(node)
        if ev:
            tr,te=ev['metrics']['train'],ev['metrics']['test']
            lines.append(f'| {node} | {tr["r2"]:.5f} | {tr["slog"]:.5f} | {tr["mse"]:.8f} | {te["r2"]:.5f} | {te["slog"]:.5f} | {te["mse"]:.8f} |')
    lines.append('')
bp=B/'monitor/dgx2/baselines/status.json'
if bp.exists():
    s=json.loads(bp.read_text());summary['baselines']=s
    lines+=['## MLP / XGBoost', '',f'状态：{s["outcome"]}。全部为CPU上的预设配置；模型选择不查看测试结果。','',
    '| 模型 | 拟合秒数 | 训练R² | 训练S_log | 训练MSE | 测试R² | 测试S_log | 测试MSE |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in s['models'].items():
        tr,te=m['metrics']['train'],m['metrics']['test']
        lines.append(f'| {name} | {m["fit_seconds"]:.1f} | {tr["r2"]:.5f} | {tr["slog"]:.5f} | {tr["mse"]:.8f} | {te["r2"]:.5f} | {te["slog"]:.5f} | {te["mse"]:.8f} |')
summary['all_complete']=len(summary['hosts'])==2 and all(s['outcome']=='completed' for s in summary['hosts'].values()) and summary.get('baselines',{}).get('outcome')=='completed'
(B/'RESULT.md').write_text('\n'.join(lines)+'\n')
(B/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'all_complete':summary['all_complete'],'hosts':{h:s['outcome'] for h,s in summary['hosts'].items()},'baselines':summary.get('baselines',{}).get('outcome')}))
