import pathlib,json,statistics
B=pathlib.Path(__file__).resolve().parent;H=['dustinstudio','dgx2','gongqian-mini'];N=['parent','middle','final'];DS=['openml12','sparse100'];data={};counts={}
f=lambda x:f'{x:.4f}'
lines=['# 双损失检查点补评估：配对测试结果','','父点=进入双损失前检查点，中间=累计60分钟，终点=累计90分钟。每机均为一套参数、统一V6广播target_only读出。每点OpenML12为7894个测试预测，sparse100为54400个同world测试预测；没有训练或参数更新。','','## 配对汇总','','| 主机 | 节点 | OpenML12 R² | OpenML12 S_log | Sparse100 R² | Sparse100 S_log |','|---|---|---:|---:|---:|---:|']
for h in H:
 p=B/'monitor'/h;s=json.loads((p/'status.json').read_text());assert s['outcome']=='completed'
 data[h]={}
 for n in N:
  data[h][n]={}
  for d in DS:
   e=json.loads((p/'evaluations'/n/d/'terminal.json').read_text());assert e['outcome']=='completed' and e['optimizer_updates']==0 and e['total_predictions']==(7894 if d=='openml12' else 54400)
   assert e.get('evaluation_supervision',e.get('supervision'))=='target_only'
   data[h][n][d]=e
  lines.append('| '+' | '.join([h,n,*[f(data[h][n][d]['macro'][k]) for d in DS for k in ['r2','slog']]])+' |')
 for d in DS:assert len({data[h][n][d]['bank_sha256'] for n in N})==1
lines+=['','## 相对父点的变化','','| 主机 | 候选 | 数据组 | ΔR² | ΔS_log | R²改善/下降表数 | S_log改善/下降表数 |','|---|---|---|---:|---:|---|---|']
for h in H:
 counts[h]={}
 for n in N[1:]:
  for d in DS:
   a=data[h]['parent'][d];z=data[h][n][d];cs={}
   for k in ['r2','slog']:
    diffs=[z['tables'][t][k]-a['tables'][t][k] for t in a['tables']]
    cs[k]=dict(improved=sum(x>0 for x in diffs),declined=sum(x<0 for x in diffs),tied=sum(x==0 for x in diffs),median_delta=statistics.median(diffs),macro_delta=z['macro'][k]-a['macro'][k])
   counts[h][n+'/'+d]=cs
   lines.append('| '+' | '.join([h,n,d,*[f(cs[k]['macro_delta']) for k in ['r2','slog']],*[str(cs[k]['improved'])+'/'+str(cs[k]['declined']) for k in ['r2','slog']]])+' |')
lines+=['','## 逐表结果','','不同主机的检查点历史不同，仅各机内部比较。不能将不同节点的逐表最高分拼接为一个模型。']
for h in H:
 for d in DS:
  lines+=['',f'### {h} / {d}','','| 表 | 父点R² | 中间R² | 终点R² | 父点S_log | 中间S_log | 终点S_log |','|---|---:|---:|---:|---:|---:|---:|']
  for t in data[h]['parent'][d]['tables']:
   lines.append('| '+' | '.join([t,*[f(data[h][n][d]['tables'][t][k]) for k in ['r2','slog'] for n in N]])+' |')
lines+=['','## 解释边界','','这些测试集曾用于历史探索与本次候选选择，结果用于工程承接决策，不是完全未使用的最终外部验收。合成测试只更换行，不更换world。当前额外测试仅广播路径，无广播的现有证据仍是718固定训练行Query。旧618独立测试留存尚未新增。','','数据分布、损失均在双损失阶段开始时发生变化；本次比较能回答候选检查点性能取舍，不能把变化单独归因双损失。判断下一轮起点不需要等待单目标因果对照；若研究方法本身优劣，再另做匹配对照。']
(B/'RESULT.md').write_text('\n'.join(lines)+'\n');(B/'comparison.json').write_text(json.dumps(counts,indent=2)+'\n')
for h in H:print(h,json.dumps({n:{d:data[h][n][d]['macro'] for d in DS} for n in N}))
