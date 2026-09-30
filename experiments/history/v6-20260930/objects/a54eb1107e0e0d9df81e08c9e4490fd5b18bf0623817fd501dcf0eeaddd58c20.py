import json,pathlib,datetime,sys
P=pathlib.Path(__file__).resolve().parent
phase=sys.argv[1] if len(sys.argv)>1 else 'half1'
label={'half1':'15分钟','half2':'30分钟终点'}[phase]
hosts=['dustinstudio','dgx2','gongqian-mini'];all_data={}
metrics=['loss','r2','slog','accuracy','rank_mae']
def fmt(x):return '—' if x is None else f'{x:.6f}'
def change(x,y):return None if x is None or y is None else x-y
lines=[f'# 718表双损失：{label}完整结果','','同一套参数，两种前向各0.5；这是同一检查点的两个推理分支，不是两个模型。每机718表、1436固定训练行Query掩码，每分支97648预测；只解释本轮固定Query拟合，不作测试泛化或历史不同银行的对比。R²、S_log、accuracy越大越好；loss、rank MAE越小越好。三机起点/头数/历史不同，不跨机归因。','', '## 分组汇总','', '| 主机 | 数据 | 分支 | 指标 | 起点 | 本节点 | Δ |','|---|---|---|---|---:|---:|---:|']
summary={}
for h in hosts:
 m=P/'monitor'/h
 before=json.loads((m/'before-eval.json').read_text());current=json.loads((m/(phase+'-eval.json')).read_text());train=json.loads((m/(phase+'-train.json')).read_text())
 assert before['outcome']==current['outcome']=='completed' and train['outcome']=='training_completed'
 assert before['bank_sha256']==current['bank_sha256'] and len(current['tables'])==718 and current['total_masks']==1436 and current['total_predictions_per_branch']==97648
 assert train['checkpoint_sha256']==current['checkpoint_sha256']
 if phase=='half1':assert train['parent_sha256']==before['checkpoint_sha256']
 all_data[h]=(before,current,train)
 summary[h]={'macro':current['macro'],'by_family':{},'training_seconds':train['successful_update_seconds'],'updates':train['checkpoint_update']-train['parent_update'],'coverage':train['table_coverage']}
 for family in ['old618','sparse100']:
  summary[h]['by_family'][family]={}
  for branch in ['v6','v55']:
   for metric in ['loss','r2','slog','nominal_accuracy','ordinal_accuracy','rank_mae']:
    k=branch+'/'+metric;a=before['by_family'][family][k];b=current['by_family'][family][k]
    if a is None and b is None:continue
    lines.append(f'| {h} | {family} | {branch} | {metric} | {fmt(a)} | {fmt(b)} | {fmt(change(b,a))} |')
   counts={}
   for metric in metrics:
    up=down=tied=undefined=0
    for name,row in current['tables'].items():
     if row['family']!=family:continue
     a=before['tables'][name][branch].get(metric);b=row[branch].get(metric)
     if metric not in row[branch]:continue
     if a is None or b is None:undefined+=1;continue
     d=b-a
     if metric in ['loss','rank_mae']:d=-d
     if d>0:up+=1
     elif d<0:down+=1
     else:tied+=1
    counts[metric]=dict(improved=up,declined=down,tied=tied,undefined=undefined)
   summary[h]['by_family'][family][branch]=counts
lines+=['','## 各表改善/下降数量','','按每项指标的优劣方向统计；accuracy只统计分类表，R²/S_log只统计数值表。未定义项不填0。','', '| 主机 | 数据 | 分支 | 指标 | 改善 | 下降 | 持平 | 未定义 |','|---|---|---|---|---:|---:|---:|---:|']
for h,data in summary.items():
 for family,branches in data['by_family'].items():
  for branch,counts in branches.items():
   for metric,c in counts.items():
    if sum(c.values()):lines.append(f'| {h} | {family} | {branch} | {metric} | {c["improved"]} | {c["declined"]} | {c["tied"]} | {c["undefined"]} |')
lines+=['','## 训练与回执','','| 主机 | 本段成功秒数 | 本段更新 | 表覆盖 |','|---|---:|---:|---:|']
for h,d in summary.items():lines.append(f'| {h} | {d["training_seconds"]:.6f} | {d["updates"]} | {d["coverage"]}/718 |')
lines+=['','每步2次前向、1次优化器更新、两loss算术平均、204/68窗口均已核对half1完整更新日志。half2父SHA与half1终点一致。检查点与评估SHA配对，固定bank为 `'+all_data[hosts[0]][1]['bank_sha256']+'`。','','## 逐表完整对比','']
for h,(before,current,train) in all_data.items():
 lines += [f'### {h}', '', '检查点：`'+current['checkpoint_sha256']+'`。','']
 for branch in ['v6','v55']:
  lines += [f'#### {branch}', '', '| 表 | 数据/类型 | loss 起点→当前 | Δloss | R² 起点→当前 | ΔR² | S_log 起点→当前 | ΔS_log | accuracy 起点→当前 | Δaccuracy | rank MAE 起点→当前 | Δrank MAE |','|---|---|---|---:|---|---:|---|---:|---|---:|---|---:|']
  for name,row in current['tables'].items():
   cells=[name,row['family']+'/'+row['kind']]
   for metric in metrics:
    a=before['tables'][name][branch].get(metric);b=row[branch].get(metric);cells.extend([fmt(a)+' → '+fmt(b),fmt(change(b,a))])
   lines.append('| '+' | '.join(cells)+' |')
lines+=['','## 当前结论','','半程：V5.5无广播分支在旧618表回归、分类上大幅恢复；V6分支旧618的R²/S_log和分类准确率小幅下降，V6稀疏100的R²/S_log提高。V5.5稀疏100仍接近或低于均值基线，DGX2该组半程还略退化。联合优化存在取舍，尚无证据两分支所有能力都同步提升，更不能证明优于匹配预算的单目标对照。后半段按既定预算继续。']

if phase=='half2':
 lines[-1]='三机30分钟训练及完整终点评估均完成。V5.5无广播分支旧618能力明显恢复；V6旧618回归仍略低于起点，稀疏100拟合高于起点。Mini的V6稀疏100在后半段从R² 0.242002回落到0.203036左右，非单调提升。各机V5.5稀疏100平均R²仍为负。本轮只衡量固定训练行Query拟合，缺匹配单目标对照，不能将旧路径提升单独归因于V6损失。'
(P/'RESULT.md').write_text('\n'.join(lines)+'\n');(P/(phase+'-summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
s=json.loads((P/'sync-state.json').read_text());s['phase']='completed' if phase=='half2' else 'half2_training';s['last_checked_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for h in hosts:
 key=h+':'+phase
 if key not in s['reported_evaluations']:s['reported_evaluations'].append(key)
 s.setdefault('hosts',{}).setdefault(h,{})[phase+'_complete']=summary[h]
(P/'sync-state.json').write_text(json.dumps(s,indent=2)+'\n')
r=P/'README.md';text=r.read_text().replace('尚无完整半程评估，继续既有控制器。','完整半程评估现已齐全，后半段继续既有控制器。')
if '## 15分钟结果' not in text:text+='\n## 15分钟结果\n\n三机各完成约900秒成功训练，Dustin/DGX2/Mini分别2486/1517/1484次更新，全部覆盖718表。完整固定Query评估各分支97648预测。V5.5旧618分支大幅恢复，V6旧618回归/分类小降、稀疏100拟合上升，尚不是两分支全面同时提升。汇总、改善数量和全部718表对比见[RESULT.md](RESULT.md)。已同步半程，后半段仍在运行，自动化保留。\n'
r.write_text(text)
for h,d in summary.items():
 b,e,t=all_data[h];print(h,'loss',[(k,b['macro'][k],e['macro'][k]) for k in ['v6/loss','v55/loss']]);print(json.dumps(d['by_family']))
