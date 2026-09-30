"""Local-only reports: only complete paired evaluation nodes are comparable."""
from pathlib import Path
import json,os
B=Path(__file__).resolve().parent
HOSTS=['dustinstudio','dgx2','gongqian-mini'];KINDS=['openml12-v6','openml12-v55','fit718']
METRICS=['r2','slog','nominal_accuracy','ordinal_accuracy','rank_mae','loss']
def read(p):return json.loads(p.read_text()) if p.exists() else None
def write(p,s):
 q=p.with_suffix('.tmp');q.write_text(s);os.replace(q,p)
def fmt(x):return '—' if x is None else f'{x:.5f}'
def delta(x,y):return x-y if x is not None and y is not None else None
def changes(now,base,branch,family):
 pairs=[(n,r[branch],base[n][branch]) for n,r in now.items() if r['family']==family]
 out={}
 for metric in ['r2','slog','accuracy','rank_mae','loss']:
  ds=[(n,delta(m.get(metric),b.get(metric))) for n,m,b in pairs];ds=[(n,v) for n,v in ds if v is not None]
  sign=-1 if metric in ['rank_mae','loss'] else 1
  out[metric]=dict(improved=sum(sign*v>1e-10 for _,v in ds),declined=sum(sign*v < -1e-10 for _,v in ds),unchanged=sum(abs(v)<=1e-10 for _,v in ds),worst=sorted(ds,key=lambda x:sign*x[1])[:5],best=sorted(ds,key=lambda x:sign*x[1],reverse=True)[:5])
 vals=[m.get('r2') for _,m,_ in pairs if m.get('r2') is not None]
 out['positive_r2']=sum(v>0 for v in vals);out['r2_gt_0_1']=sum(v>.1 for v in vals)
 return out
summary={};lines=['# 随机二选一损失：稀疏100表权重30%，续训一小时','',
 '本轮从此前两小时终点续接，稀疏100表占30%，其他630表占70%；各组内按表等频。',
 '每步以50%概率选择一个完整分支损失，仅一次前向、反向及AdamW更新。概率不等于有限步数恰好各半。',
 'OpenML12是固定测试；old618和sparse100是固定训练行Query拟合。两条推理路径分开比较，不能用本轮证明随机方案优于双前向或单损失。S_log缺定义保留空值，R²极端退化仍须报告。','']
for node in range(1,3):
 if (B/f'STAGE{node}.md').exists():lines += [f'{node*30}分钟解读与指标对照：[STAGE{node}.md](STAGE{node}.md)。','']
detail=['# 逐表结果：两条推理路径','']
for host in HOSTS:
 d=B/'monitor'/host;c=read(d/'controller.json') or {};phases={};entry=dict(controller=c,phases={});summary[host]=entry
 baseline={k:read(d/f'before-{k}-eval.json') for k in KINDS}
 if all(e and e.get('outcome')=='completed' for e in baseline.values()):
  phases['before']=baseline
  assert len({e['checkpoint_sha256'] for e in baseline.values()})==1
 total=0;branch_counts={'v6':0,'v55':0};last=baseline if phases else None
 for i in range(1,3):
  phase='stage'+str(i);ev={k:read(d/f'{phase}-{k}-eval.json') for k in KINDS};t=read(d/f'{phase}-train.json')
  if not (last and t and t.get('outcome')=='training_completed' and all(e and e.get('outcome')=='completed' for e in ev.values())):continue
  for kind,e in ev.items():
   assert e['checkpoint_sha256']==t['checkpoint_sha256'] and e['bank_sha256']==baseline[kind]['bank_sha256'] and e['optimizer_updates']==0
   if kind=='fit718':assert e['total_masks']==1436 and e['total_predictions_per_branch']==97648 and len(e['tables'])==718
   else:assert e['total_predictions']==7894 and e['inference_branch']==kind.split('-')[1]
  assert t['schedule_cursor_reset'] == (i==1) and t['forward_passes_per_update']==t['optimizer_steps_per_update']==1
  total+=t['successful_update_seconds'];updates=sum(t['table_updates'].values());assert updates==sum(t['branch_counts'].values())
  for branch in branch_counts:branch_counts[branch]+=t['branch_counts'][branch]
  s=dict(checkpoint_sha256=t['checkpoint_sha256'],successful_seconds=t['successful_update_seconds'],cumulative_successful_seconds=total,updates=updates,coverage=t['table_coverage'],branch_counts=t['branch_counts'],cumulative_branch_counts=dict(branch_counts),cohort_updates=t['cohort_updates'],sparse_update_fraction=t['cohort_updates'].get('sparse100',0)/updates,branch_selector_restored=t['branch_selector_restored'],openml12={},fit718={})
  for branch in ['v6','v55']:
   k='openml12-'+branch;m=ev[k]['macro'];s['openml12'][branch]=dict(macro=m,delta_before={x:delta(m.get(x),baseline[k]['macro'].get(x)) for x in ['r2','slog']},delta_previous={x:delta(m.get(x),last[k]['macro'].get(x)) for x in ['r2','slog']},per_table={n:{x:delta(r.get(x),baseline[k]['tables'][n].get(x)) for x in ['r2','slog']} for n,r in ev[k]['tables'].items()})
   s['fit718'][branch]={}
   for family in ['old618','sparse100']:
    m=ev['fit718']['by_family'][family];bm=baseline['fit718']['by_family'][family];lm=last['fit718']['by_family'][family]
    s['fit718'][branch][family]=dict(macro={x:m.get(branch+'/'+x) for x in METRICS},delta_before={x:delta(m.get(branch+'/'+x),bm.get(branch+'/'+x)) for x in METRICS},delta_previous={x:delta(m.get(branch+'/'+x),lm.get(branch+'/'+x)) for x in METRICS},changes=changes(ev['fit718']['tables'],baseline['fit718']['tables'],branch,family))
  entry['phases'][phase]=s;phases[phase]=ev;last=ev
 lines += [f'## {host}','',f"状态：{c.get('outcome','等待首份回执')}；完整训练评估节点：{', '.join(entry['phases']) or '暂无'}。",'']
 if entry['phases']:
  lines += ['| 节点 | 累计成功训练秒 | 更新数 | 本段覆盖 | V6 / V5.5 更新数 | 稀疏采样占比 |','|---|---:|---:|---:|---|---:|']
  for phase,s in entry['phases'].items():lines.append(f"| {phase} | {s['cumulative_successful_seconds']:.2f} | {s['updates']} | {s['coverage']}/730 | {s['branch_counts']['v6']} / {s['branch_counts']['v55']} | {s['sparse_update_fraction']:.2%} |")
 for branch in ['v6','v55']:
  lines += ['',f'### {branch}：OpenML12固定测试','', '| 节点 | R² | S_log | ΔR² 起点 | ΔS_log 起点 | ΔR² 前点 | ΔS_log 前点 |','|---|---:|---:|---:|---:|---:|---:|']
  prev=None
  for phase,ev in phases.items():
   m=ev['openml12-'+branch]['macro'];bm=baseline['openml12-'+branch]['macro'];prev=prev or bm
   lines.append('| '+phase+' | '+' | '.join(fmt(v) for v in [m['r2'],m['slog'],delta(m['r2'],bm['r2']),delta(m['slog'],bm['slog']),delta(m['r2'],prev['r2']),delta(m['slog'],prev['slog'])])+' |');prev=m
  for family in ['old618','sparse100']:
   lines += ['',f'### {branch}：{family}固定训练行拟合','', '| 节点 | R² | S_log | nominal acc | ordinal acc | rank MAE | Query loss | 正R²表 | R²>0.1表 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
   for phase,ev in phases.items():
    m=ev['fit718']['by_family'][family];vals=[r[branch].get('r2') for r in ev['fit718']['tables'].values() if r['family']==family];vals=[v for v in vals if v is not None]
    lines.append('| '+phase+' | '+' | '.join(fmt(m.get(branch+'/'+x)) for x in METRICS)+f" | {sum(v>0 for v in vals)} | {sum(v>.1 for v in vals)} |")
 for phase,ev in phases.items():
  detail += [f'## {host} / {phase}','', '| 路径 | 数据组/表 | R² | S_log | accuracy | rank MAE | Query loss |','|---|---|---:|---:|---:|---:|---:|']
  for branch in ['v6','v55']:
   for name,m in ev['openml12-'+branch]['tables'].items():detail.append('| '+branch+' | OpenML12/'+name+' | '+' | '.join(fmt(m.get(x)) for x in ['r2','slog','accuracy','rank_mae','loss'])+' |')
   for name,row in ev['fit718']['tables'].items():detail.append('| '+branch+' | '+row['family']+'/'+name+' | '+' | '.join(fmt(row[branch].get(x)) for x in ['r2','slog','accuracy','rank_mae','loss'])+' |')
  detail.append('')
complete=all(s['controller'].get('outcome')=='completed' and 'stage2' in s['phases'] and s['phases']['stage2']['cumulative_successful_seconds']>=3600 for s in summary.values())
write(B/'RESULT.md','\n'.join(lines)+'\n');write(B/'DETAIL.md','\n'.join(detail)+'\n');write(B/'summary.json',json.dumps(dict(all_complete=complete,hosts=summary),indent=2)+'\n')
print(json.dumps(dict(all_complete=complete,phases={h:list(s['phases']) for h,s in summary.items()})))
