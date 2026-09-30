import json
from pathlib import Path
P=Path(__file__).resolve().parent
hosts=['dustinstudio','dgx2','gongqian-mini']; rows=[]
for h in hosts:
 d=P/'monitor'/h
 b=json.loads((d/'openml12-baseline.json').read_text());e=json.loads((d/'openml12-transfer.json').read_text());parent=json.loads((d/'before-eval.json').read_text());t=json.loads((d/'half2-train.json').read_text())
 assert e['outcome']==b['outcome']=='completed' and e['total_predictions']==b['total_predictions']==7894 and len(e['tables'])==12
 assert b['bank_sha256']==e['bank_sha256'] and b['checkpoint_sha256']==parent['checkpoint_sha256'] and e['checkpoint_sha256']==t['checkpoint_sha256']
 assert e['optimizer_updates']==0
 rows.append((h,b,e))
lines=['# 合成100表专项后的OpenML12迁移评估','', '三机最终检查点均完成评估，每机12表7894预测，无额外参数更新。评估完全沿用原冻结bank：全训练集带标签support＋全测试集Query，一表一次完整前向。起点为此次合成专项的各自直接父检查点，即512窗口一小时终点；已核对父SHA、最终SHA、bank SHA。','', '本轮100张合成表专项后，以下变化包含合成训练与40%旧618回放的整体影响。OpenML12曾在历史阶段训练过，所以这里是历史真实任务上的性能变化，不是未见表泛化实验。不同主机损失、头数、历史和更新次数不同，不作损失/硬件因果归因。','', '| 主机 / 损失 | 训练前R² | 训练后R² | ΔR² | 训练前S_log | 训练后S_log | ΔS_log | R²改善/下降 |','|---|---:|---:|---:|---:|---:|---:|---|']
for h,b,e in rows:
 improved=sum(e['tables'][k]['r2']>b['tables'][k]['r2'] for k in e['tables']);declined=sum(e['tables'][k]['r2']<b['tables'][k]['r2'] for k in e['tables']); vals=[b['macro']['r2'],e['macro']['r2'],e['macro']['r2']-b['macro']['r2'],b['macro']['slog'],e['macro']['slog'],e['macro']['slog']-b['macro']['slog']]
 lines.append('| '+h+' / '+e['identity']['supervision']+' | '+' | '.join(f'{v:.6f}' for v in vals)+f' | {improved}/{declined} |');print(h,vals,improved,declined)
for h,b,e in rows:
 lines += ['',f'## {h}','',f"父检查点：`{b['checkpoint_sha256']}`；最终检查点：`{e['checkpoint_sha256']}`。",'','| 表 | 训练前R² | 训练后R² | ΔR² | 训练前S_log | 训练后S_log | ΔS_log |','|---|---:|---:|---:|---:|---:|---:|']
 for k in e['tables']:
  a=b['tables'][k];z=e['tables'][k];v=[a['r2'],z['r2'],z['r2']-a['r2'],a['slog'],z['slog'],z['slog']-a['slog']];lines.append('| '+k+' | '+' | '.join(f'{x:.6f}' for x in v)+' |')
  if k in ['kin8nm','pumadyn32nh','space_ga','cpu_activity']:print(h,k,[round(x,4) for x in v])
lines += ['', '## 结论与范围','', '三个模型均在此次合成专项后出现OpenML12宏指标下降；不能声称此轮专项已改善真实困难表。合成100表测试改善与真实表退化同时发生。尚不能由这一次对比区分梯度冲突、任务分布偏移、回放覆盖或其他机制，也不能据此否定广播设计本身。旧618留存未在本轮测量，40%回放不构成留存证据。', '', '原始回执：monitor/<host>/openml12-baseline.json 和 openml12-transfer.json；逐样本预测保存在远端 evaluations/openml12-transfer/。评估器沿用上一轮，只显式设置现有内存保护：Dustin/DGX2 50%，Mini 75%。']
(P/'OPENML12-TRANSFER.md').write_text('\n'.join(lines)+'\n')
f=P/'README.md';s=f.read_text();s+='\n## OpenML12迁移评估完成\n\n三机最终检查点已完成每机7894条完整预测，均较合成专项前出现宏R²/S_log下降。完整逐表前后对照及口径见 [OPENML12-TRANSFER.md](OPENML12-TRANSFER.md)。该评估没有额外训练，不能视为未见表泛化；旧618留存仍未补测。\n';f.write_text(s)
