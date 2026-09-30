import json,pathlib,datetime
P=pathlib.Path(__file__).resolve().parent; O=P.parent/'v6-v55-dual718-30m-20260928'
H=['dustinstudio','dgx2','gongqian-mini']; D={}; S={}
f=lambda x:'—' if x is None else f'{x:.6f}'
def delta(a,b):return None if a is None or b is None else a-b
lines=['# 718表双损失新增60分钟：完整终点','','三机均完成新增3600秒成功训练（末步略超）和完整终点评估。本系列累计约90分钟。每分支718表、1436固定训练行Query掩码、97648预测，五个节点银行一致。仅说明已知表训练行拟合，不是测试或未见world泛化；没有匹配单目标对照，不能归因双损失优于单损失。','','## 结论','','无广播稀疏100未突破：Dustin最后半小时明显回落，DGX2回落，Mini改善；三机平均R²仍负，均无R²>0.1表。旧618无广播终点平均R²均高于本轮起点，但DGX2后半段回落。广播旧618本轮Dustin提升、DGX2与Mini略降。不能把部分指标增益说成持续稳定提升。','','## 成功训练及覆盖','','| 主机 | 新增成功秒数 | 新增更新 | 两段各自覆盖 |','|---|---:|---:|---|']
for h in H:
 m=P/'monitor'/h;o=O/'monitor'/h
 es=[json.loads((o/(n+'.json')).read_text()) for n in ['before-eval','half1-eval','half2-eval']]+[json.loads((m/(n+'-eval.json')).read_text()) for n in ['half1','half2']]
 ts=[json.loads((m/(n+'-train.json')).read_text()) for n in ['half1','half2']];c=json.loads((m/'controller.json').read_text())
 assert c['outcome']=='completed'
 assert all(e['outcome']=='completed' and len(e['tables'])==718 and e['total_masks']==1436 and e['total_predictions_per_branch']==97648 for e in es)
 assert len({e['bank_sha256'] for e in es})==1
 for i,t in enumerate(ts):
  assert t['outcome']=='training_completed' and not t['sampling_cursor_reset'] and t['parent_sha256']==es[i+2]['checkpoint_sha256'] and t['checkpoint_sha256']==es[i+3]['checkpoint_sha256']
  assert t['forward_passes_per_update']==2 and t['optimizer_steps_per_update']==1 and t['identity']['weights']==[.5,.5]
 sec=sum(t['successful_update_seconds'] for t in ts); assert 3600<=sec<3610
 D[h]=es;S[h]={'seconds':sec,'updates':sum(t['checkpoint_update']-t['parent_update'] for t in ts),'coverage':[t['table_coverage'] for t in ts],'counts':{},'checkpoint_sha256':es[-1]['checkpoint_sha256']}
 v=S[h];lines.append(f"| {h} | {sec:.3f} | {v['updates']} | {v['coverage']} / 718 |")
lines+=['','## 分组完整轨迹','','| 主机 | 数据 | 分支 | 指标 | 最初0分 | 累计15分 | 累计30分/本轮起点 | 累计60分/中点 | 累计90分/终点 | Δ本轮 | Δ后半段 |','|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for h,es in D.items():
 for group in ['old618','sparse100']:
  for b in ['v6','v55']:
   for k in ['loss','r2','slog','nominal_accuracy','ordinal_accuracy','rank_mae']:
    vs=[e['by_family'][group][b+'/'+k] for e in es]
    if all(v is None for v in vs):continue
    lines.append('| '+' | '.join([h,group,b,k,*map(f,vs),f(delta(vs[-1],vs[2])),f(delta(vs[-1],vs[3]))])+' |')
   for k in ['loss','r2','slog','accuracy','rank_mae']:
    for ix,label in [(2,'vs_start'),(3,'vs_midpoint')]:
     counts=dict(improved=0,declined=0,tied=0,undefined=0)
     for name,r in es[-1]['tables'].items():
      if r['family']!=group or k not in r[b]:continue
      a=es[ix]['tables'][name][b].get(k);v=r[b][k]
      if a is None or v is None:counts['undefined']+=1;continue
      d=(v-a)*(-1 if k in ['loss','rank_mae'] else 1);counts['improved' if d>0 else 'declined' if d<0 else 'tied']+=1
     if sum(counts.values()):S[h]['counts'][group+'/'+b+'/'+k+'/'+label]=counts
lines+=['','## 无广播稀疏100正R²表数','','| 主机 | 正R²数（0/15/30/60/90分） | R²>0.1数 |','|---|---|---|']
for h,es in D.items():
 cnt={str(th):[sum(r['v55']['r2']>th for r in e['tables'].values() if r['family']=='sparse100') for e in es] for th in [0,.1]};S[h]['sparse_thresholds']=cnt
 lines.append('| '+' | '.join([h,' → '.join(map(str,cnt['0'])),' → '.join(map(str,cnt['0.1']))])+' |')
lines+=['','## 逐表改善/下降','','| 主机 | 组/分支/指标/比较节点 | 改善 | 下降 | 持平 | 未定义 |','|---|---|---:|---:|---:|---:|']
for h,s in S.items():
 for k,v in s['counts'].items():lines.append('| '+' | '.join([h,k,*map(str,v.values())])+' |')
lines+=['','## 完整逐表结果','','旧618数值R²共292表；S_log有6表未定义，显示—，宏平均仅286表。合成100全部为数值表。分类accuracy与有序rank MAE见对应类型。loss及rank MAE越低越好，其他指标越高越好。']
for h,es in D.items():
 lines+=['',f'### {h}','','终点检查点：`'+es[-1]['checkpoint_sha256']+'`。']
 for b in ['v6','v55']:
  lines+=['',f'#### {b}','','| 表 | 数据/类型 | 指标 | 0分 | 15分 | 30分 | 60分 | 90分 | Δ本轮 | Δ后半段 |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
  for name,r in es[-1]['tables'].items():
   for k in ['loss','r2','slog','accuracy','rank_mae']:
    if k not in r[b]:continue
    vs=[e['tables'][name][b].get(k) for e in es];lines.append('| '+' | '.join([name,r['family']+'/'+r['kind'],k,*map(f,vs),f(delta(vs[-1],vs[2])),f(delta(vs[-1],vs[3]))])+' |')
(P/'RESULT.md').write_text('\n'.join(lines)+'\n');(P/'final-summary.json').write_text(json.dumps(S,indent=2)+'\n')
r=P/'README.md';text=r.read_text();mark='## 新增60分钟终点结果'
if mark not in text:text+='\n'+mark+'\n\n三机训练及完整终点评估均完成，新增成功秒数Dustin3600.239、DGX2 3600.743、Mini3600.189。无广播稀疏100 R²起点→中点→终点：Dustin -0.017258→-0.012556→-0.036574；DGX2 -0.026141→-0.017825→-0.022727；Mini -0.014654→-0.015821→-0.011924。终点正R²表数22/30/39，均无超过0.1表，未见稳定突破。完整loss、R²、S_log、分类、rank MAE及全部逐表轨迹见RESULT.md。预算已结束，不自动追加。\n'
r.write_text(text)
s=json.loads((P/'sync-state.json').read_text());s['phase']='completed';s['terminal_reported']=True;s['last_checked_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for h in H:
 if h+':half2' not in s['reported_evaluations']:s['reported_evaluations'].append(h+':half2')
 s['hosts'][h].update(successful_seconds=S[h]['seconds'],successful_updates=S[h]['updates'],phase='completed',controller_outcome='completed',final_complete=S[h])
(P/'sync-state.json').write_text(json.dumps(s,indent=2)+'\n')
for h,s in S.items():print(h,s['seconds'],s['updates'],s['sparse_thresholds'],s['counts']['sparse100/v55/r2/vs_start'],s['counts']['sparse100/v55/r2/vs_midpoint'])
