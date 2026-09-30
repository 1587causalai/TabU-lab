"""Summarize completed remote fine-tunes against the fixed historical baselines."""
from pathlib import Path
import json
import math

ROOT = Path(__file__).parent
state = json.loads((ROOT/'status.json').read_text())
before = json.loads((ROOT.parent/'openml12-frozen-icl-20260927/results.json').read_text())
hosts = ['dgx2', 'dustinstudio', 'gongqian-mini']
names = ['airfoil_self_noise','concrete_compressive_strength','qsar_fish_toxicity',
         'bodyfat','energy_efficiency','cars','red_wine','white_wine','space_ga',
         'cpu_activity','kin8nm','pumadyn32nh']
historical = before['historical_reference']['datasets']
assert all(state['hosts'][h]['outcome'] == 'completed' for h in hosts)
assert all(set(state['hosts'][h]['datasets']) == set(names) for h in hosts)
assert len({state['hosts'][h]['bank_sha256'] for h in hosts}) == 1
summary = dict(scope='Exploratory independent per-table fine-tuning, fixed historical test bank',
               refreshed_utc=state['refreshed_utc'], hosts={}, tables={})
for name in names:
    summary['tables'][name] = dict(
        mlp=historical[name]['mlp'], xgboost=historical[name]['xgboost'],
        historical_tar=historical[name]['tar'], hosts={})
    for host in hosts:
        new = state['hosts'][host]['datasets'][name]
        old = before['hosts'][host]['datasets'][name]['metrics']
        summary['tables'][name]['hosts'][host] = dict(
            before=old, after=new['metrics'], delta_r2=new['metrics']['r2']-old['r2'],
            updates=new['checkpoint_update'], successful_seconds=new['successful_update_seconds'],
            checkpoint=new['checkpoint'], checkpoint_sha256=new['checkpoint_sha256'])
for host in hosts:
    new = state['hosts'][host]
    rows = [summary['tables'][n]['hosts'][host] for n in names]
    avg = math.fsum(r['after']['r2'] for r in rows)/12
    assert abs(avg-new['macro_r2']) < 1e-12
    summary['hosts'][host] = dict(
        before_macro_r2=before['hosts'][host]['macro_r2'], after_macro_r2=avg,
        delta_macro_r2=avg-before['hosts'][host]['macro_r2'],
        after_median_r2=new['median_r2'], improved_tables=sum(r['delta_r2']>0 for r in rows),
        wins_mlp=sum(r['after']['r2']>historical[n]['mlp']['r2'] for n,r in zip(names,rows)),
        wins_xgboost=sum(r['after']['r2']>historical[n]['xgboost']['r2'] for n,r in zip(names,rows)),
        successful_seconds=math.fsum(r['successful_seconds'] for r in rows),
        updates=sum(r['updates'] for r in rows), parent_sha256=new['parent_sha256'],
        remote_root=new['remote_root'])
summary['baseline_macro_r2'] = {k: math.fsum(historical[n][k]['r2'] for n in names)/12
                                for k in ('mlp','xgboost','tar')}
(ROOT/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines = ['# OpenML12：每表2分钟独立微调结果', '',
    '三台各完成12个独立单表模型。每表从本机同一个 joint618 父 checkpoint 重新初始化，仅用当前表的训练划分适配；测试复用历史固定 context/Query 行地址，共7,894行。', '',
    '| 模型 | 微调前宏平均 R² | 微调后 | 提升 | 改善表数 | 胜 MLP | 胜 XGBoost |',
    '|---|---:|---:|---:|---:|---:|---:|']
for h,v in summary['hosts'].items():
    lines.append(f"| {h} | {v['before_macro_r2']:.4f} | {v['after_macro_r2']:.4f} | {v['delta_macro_r2']:+.4f} | {v['improved_tables']}/12 | {v['wins_mlp']}/12 | {v['wins_xgboost']}/12 |")
lines += ['', '历史同划分基线：MLP 宏平均 R² '+f"{summary['baseline_macro_r2']['mlp']:.4f}"+
          '；XGBoost '+f"{summary['baseline_macro_r2']['xgboost']:.4f}"+'。基线结果直接复用，未重新训练。', '',
          '## 每表测试 R²', '', '| 表 | DGX2 H8 | Dustin H8 | Mini H4 | MLP | XGBoost |',
          '|---|---:|---:|---:|---:|---:|']
for name, row in summary['tables'].items():
    scores=[row['hosts'][h]['after']['r2'] for h in hosts]+[row['mlp']['r2'],row['xgboost']['r2']]
    lines.append('| '+name+' | '+' | '.join(f'{s:.4f}' for s in scores)+' |')
lines += ['', '## 执行与比较口径', '',
          '- 固定每表约120秒成功参数更新时间，终点选取不使用测试分数；保存、加载和评测另计。',
          '- 各机保留自己的模型、loss、AdamW LR=1e-4 与 runtime；每表 weights-only 初始化，admission 后严格恢复。',
          '- 与历史方案相同：旧3表全训练池 episode，后9表最多204行窗口，训练 Query 约1/3；V5.5 的训练随机流和优化器保留当前实现。',
          '- 当前是同一表、同一训练/测试划分、一表一模型的探索性性能比较。TabU 有预训练历史，双方计算预算与调参次数未等化。',
          '- MLP/XGBoost 使用全部训练行拟合；历史 MLP 为(128,128)、训练内早停选轮数；XGBoost depth=6、最多600树、训练内早停选树数。TabU 测试沿用历史有限 context。',
          '- 按用户要求未做测试行历史训练曝光审计。', '',
          '## 有效训练量', '', '| 主机 | 成功更新秒数 | 总更新数 |', '|---|---:|---:|']
for h,v in summary['hosts'].items():
    lines.append(f"| {h} | {v['successful_seconds']:.3f} | {v['updates']} |")
lines += ['', '各表更新数、实际训练秒数、RMSE/MAE/R²、前后差值、checkpoint SHA 与路径见 [comparison.json](comparison.json)。',
          '逐行预测和原始训练日志保留在各机实验根的 `runs/<table>/`；完整远端终态读回见 [status.json](status.json)。',
          '执行配方见 [PLAN.md](PLAN.md)，控制代码见 [run_campaign.py](run_campaign.py)。']
(ROOT/'RESULT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary['hosts'],ensure_ascii=False,indent=2))
print('Report:', ROOT/'RESULT.md')
