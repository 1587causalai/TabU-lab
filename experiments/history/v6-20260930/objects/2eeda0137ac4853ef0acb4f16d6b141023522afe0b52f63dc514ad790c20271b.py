"""Compare verified early-stopping outputs with the preserved fit probes."""
import json
from pathlib import Path

B = Path(__file__).resolve().parent
OLD = B.parent/'v6-puma-single-hostloss-15m-20260929'
new = json.loads((B/'terminal.json').read_text())
old = json.loads((OLD/'monitor/dgx2/baselines/terminal.json').read_text())
assert new['outcome'] == old['outcome'] == 'completed'
assert new['bank_sha256'] == old['bank_sha256']
assert new['selection_completed_utc'] < new['test_started_utc']
assert new['train_n'] == old['train_n'] == 6553 and new['test_n'] == old['test_n'] == 1639
assert set(new['models']) == set(old['models'])
lines = ['# Puma 基线：验证集早停结果', '',
    '固定原6553/1639行划分；从训练集内划5242拟合行和1311验证行选择轮数，再用全部6553行按固定轮数重训。原四档结构、学习率、正则等不变；CPU FP32、2线程、seed=20260929。全部选择/重训完成后统一评分原测试集。', '',
    '| 模型 | 原训练R² | 早停重训R² | 原测试R² | 早停重训测试R² | 测试变化 | 选定轮数 |',
    '|---|---:|---:|---:|---:|---:|---:|']
comparisons = {}
for name in ['mlp-128x3', 'mlp-256x3', 'xgboost-depth6', 'xgboost-depth12']:
    o, n, sel = old['models'][name], new['models'][name], new['selected'][name]
    assert n['selected_rounds'] == n['refit_rounds'] == sel['best_rounds']
    assert n['metrics']['train']['n'] == 6553 and n['metrics']['test']['n'] == 1639
    a, b = o['metrics'], n['metrics']
    delta = b['test']['r2']-a['test']['r2']
    comparisons[name] = dict(old=a, earlystop_refit=b, selected=sel,
                             delta_test_r2=delta, delta_test_slog=b['test']['slog']-a['test']['slog'])
    lines.append(f'| {name} | {a["train"]["r2"]:.6f} | {b["train"]["r2"]:.6f} | {a["test"]["r2"]:.6f} | {b["test"]["r2"]:.6f} | {delta:+.6f} | {sel["best_rounds"]} |')
lines += ['', '## 完整指标', '',
    '| 模型 | 训练S_log | 测试S_log | 训练MSE | 测试MSE | 训练预测std | 测试预测std |',
    '|---|---:|---:|---:|---:|---:|---:|']
for name, comp in comparisons.items():
    a, b = comp['earlystop_refit']['train'], comp['earlystop_refit']['test']
    lines.append(f'| {name} | {a["slog"]:.6f} | {b["slog"]:.6f} | {a["mse"]:.9f} | {b["mse"]:.9f} | {a["prediction_std"]:.6f} | {b["prediction_std"]:.6f} |')
lines += ['', '## 选择与预算', '',
    '| 模型 | 验证R² | 搜索轮数 | 最佳轮数 | 耐心耗尽 | 选择秒数 | 重训秒数 |',
    '|---|---:|---:|---:|---|---:|---:|']
for name, sel in new['selected'].items():
    lines.append(f'| {name} | {sel["validation_metrics"]["r2"]:.6f} | {sel["searched_rounds"]} | {sel["best_rounds"]} | {sel["stopped_by_patience"]} | {sel["selection_seconds"]:.2f} | {new["models"][name]["refit_seconds"]:.2f} |')
lines += ['', '## 比较边界', '',
    '本轮只增加验证集选轮数与固定轮数全量重训；不是完整超参数搜索，不保证消除所有过拟合。一轮固定验证切分与随机种子不能证明稳定上限。全部四档结果保留，未按测试分数选择最终模型。测试集此前已被报告，因此不称全新盲测。', '',
    'TabU此前15分钟终点：Dustin V6训练R²0.66332、测试0.62964；DGX2 V5.5训练0.63463、测试0.62271。Dustin5分钟测试0.64754为已观察到的中间节点，不是验证集选择的模型。TabU有预训练史与support上下文，基线为冷启动；当前不是等算力、同起点的架构因果对照。', '',
    f'bank SHA：`{new["bank_sha256"]}`；选择完成：{new["selection_completed_utc"]}；测试开始：{new["test_started_utc"]}。', '']
(B/'RESULT.md').write_text('\n'.join(lines))
(B/'summary.json').write_text(json.dumps(dict(outcome='completed', comparisons=comparisons,
    selection_completed_utc=new['selection_completed_utc'], test_started_utc=new['test_started_utc']), indent=2)+'\n')
print(json.dumps({k: dict(train_r2=v['earlystop_refit']['train']['r2'],
    test_r2=v['earlystop_refit']['test']['r2'], delta_test_r2=v['delta_test_r2'],
    selected_rounds=v['selected']['best_rounds']) for k,v in comparisons.items()}))
