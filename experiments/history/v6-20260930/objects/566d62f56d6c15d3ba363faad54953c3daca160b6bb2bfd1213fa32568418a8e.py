import pathlib,json,datetime
P=pathlib.Path(__file__).resolve().parent;O=P.parent/'v6-v55-dual718-30m-20260928';hosts=['dustinstudio','dgx2','gongqian-mini'];D={};summary={}
fmt=lambda x:'—' if x is None else f'{x:.6f}'
def diff(x,y):return None if x is None or y is None else x-y
for h in hosts:
 m=P/'monitor'/h;o=O/'monitor'/h
 es=[json.loads((o/(n+'.json')).read_text()) for n in ['before-eval','half1-eval','half2-eval']]+[json.loads((m/'half1-eval.json').read_text())]
 t=json.loads((m/'half1-train.json').read_text());b=json.loads((m/'half2-train.json').read_text());c=json.loads((m/'controller.json').read_text())
 assert t['outcome']=='training_completed' and t['parent_sha256']==es[2]['checkpoint_sha256'] and b['parent_sha256']==t['checkpoint_sha256']==es[3]['checkpoint_sha256']
 assert all(e['outcome']=='completed' and len(e['tables'])==718 and e['total_masks']==1436 and e['total_predictions_per_branch']==97648 for e in es)
 assert len({e['bank_sha256'] for e in es})==1 and not t['sampling_cursor_reset']
 D[h]=es;summary[h]={'seconds':t['successful_update_seconds'],'updates':t['checkpoint_update']-t['parent_update'],'coverage':t['table_coverage'],'phase':c['phase'],'controller_outcome':c['outcome'],'counts':{}}
lines=['# 718双损失续训：新增30分钟中点（累计60分钟）','','三机均完成本段约1800秒成功训练及完整718表双分支评估，已进入后半段。每分支97648预测，固定1436mask；与原0/15/30分钟银行一致。仅衡量已知表训练行Query拟合，非测试泛化，缺匹配单目标对照，不作损失因果优劣结论。','','## 分组轨迹','','| 主机 | 数据 | 分支 | 指标 | 最初0分 | 累计15分 | 累计30分/本轮起点 | 累计60分/本轮中点 | Δ本轮起点 |','|---|---|---|---|---:|---:|---:|---:|---:|']
for h,es in D.items():
 for f in ['old618','sparse100']:
  for b in ['v6','v55']:
   for k in ['loss','r2','slog','nominal_accuracy','ordinal_accuracy','rank_mae']:
    vs=[e['by_family'][f][b+'/'+k] for e in es]
    if all(v is None for v in vs):continue
    lines.append('| '+' | '.join([h,f,b,k,*map(fmt,vs),fmt(diff(vs[-1],vs[-2]))])+' |')
   for k in ['loss','r2','slog','accuracy','rank_mae']:
    counts={'improved':0,'declined':0,'tied':0,'undefined':0}
    for name,r in es[-1]['tables'].items():
     if r['family']!=f or k not in r[b]:continue
     a=es[-2]['tables'][name][b].get(k);v=r[b][k]
     if a is None or v is None:counts['undefined']+=1;continue
     d=(v-a)*(-1 if k in ['loss','rank_mae'] else 1);counts['improved' if d>0 else 'declined' if d<0 else 'tied']+=1
    if sum(counts.values()):summary[h]['counts'][f+'/'+b+'/'+k]=counts
lines+=['','## 无广播稀疏100突破情况','','| 主机 | 各节点正R²表数（0/15/30/60分） | 各节点R²>0.1表数 | 本段成功秒数 | 本段更新 | 覆盖 |','|---|---|---|---:|---:|---:|']
for h,es in D.items():
 counts={str(th):[sum(r['v55']['r2']>th for r in e['tables'].values() if r['family']=='sparse100') for e in es] for th in [0,.1]};summary[h]['sparse_thresholds']=counts
 v=summary[h];lines.append('| '+' | '.join([h,' → '.join(map(str,counts['0'])),' → '.join(map(str,counts['0.1'])),fmt(v['seconds']),str(v['updates']),str(v['coverage'])+'/718'])+' |')
lines+=['','## 逐表改善/下降（本轮中点对本轮起点）','','| 主机 | 数据/分支/指标 | 改善 | 下降 | 持平 | 未定义 |','|---|---|---:|---:|---:|---:|']
for h,s in summary.items():
 for key,c in s['counts'].items():lines.append('| '+' | '.join([h,key,*map(str,c.values())])+' |')
lines+=['','## 全部逐表结果','','旧618的S_log有6表未定义，以—表示，汇总仅使用286张有效数值表；数值R²有292旧表＋100合成表。']
for h,es in D.items():
 lines+=['',f'### {h}','','中点检查点：`'+es[-1]['checkpoint_sha256']+'`。','']
 for b in ['v6','v55']:
  lines+=['',f'#### {b}','','| 表 | 数据/类型 | 指标 | 最初0分 | 累计15分 | 累计30分 | 累计60分 | Δ本轮 |','|---|---|---|---:|---:|---:|---:|---:|']
  for name,row in es[-1]['tables'].items():
   for k in ['loss','r2','slog','accuracy','rank_mae']:
    if k not in row[b]:continue
    vs=[e['tables'][name][b].get(k) for e in es];lines.append('| '+' | '.join([name,row['family']+'/'+row['kind'],k,*map(fmt,vs),fmt(diff(vs[-1],vs[-2]))])+' |')
lines+=['','## 当前判断','','无广播稀疏100仍未突破：Dustin与DGX2平均R²小幅改善但仍为负，Mini略退。三机没有一张该类表R²超过0.1。广播分支该类表Dustin/Mini提升、DGX2回落。旧618无广播：DGX2/Mini回归继续改善；Dustin R²略降但S_log与分类改善。不能说所有能力持续提升。按授权继续剩余半小时，不改配置。']
(P/'RESULT.md').write_text('\n'.join(lines)+'\n');(P/'half1-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
s=json.loads((P/'sync-state.json').read_text());s['phase']='half2_training';s['last_checked_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for h in hosts:
 if h+':half1' not in s['reported_evaluations']:s['reported_evaluations'].append(h+':half1')
 s['hosts'][h]['half1_complete']=summary[h]
(P/'sync-state.json').write_text(json.dumps(s,indent=2)+'\n')
r=P/'README.md';text=r.read_text()
if '## 新增30分钟中点结果' not in text:text+='\n## 新增30分钟中点结果\n\n三机完整中点评估已齐，新增半小时成功训练，累计约60分钟。无广播稀疏100 R²：Dustin -0.017258→-0.012556；DGX2 -0.026141→-0.017825；Mini -0.014654→-0.015821。正R²表数26→35、20→32、32→30，均无R²>0.1表，尚未突破均值预测水平。广播该组Dustin/Mini提升、DGX2回落；完整两分支/两数据组loss、回归/分类、逐表改善及0/15/30/60分钟轨迹见RESULT.md。后半段正常继续，半程已同步，自动化保留。\n'
r.write_text(text)
for h,s in summary.items():print(h,s['seconds'],s['updates'],s['counts']['sparse100/v55/r2'])
