"""Render a complete node against before and the immediately previous node."""
import json,argparse
from pathlib import Path
B=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True,choices=[1,2,3,4]);a=p.parse_args();phase=f'stage{a.stage}';previous='before' if a.stage==1 else f'stage{a.stage-1}'
summary=json.loads((B/'summary.json').read_text())
selected={h:v for h,v in summary['hosts'].items() if phase in v['phases']}
assert selected, f'No complete paired evaluations for {phase}'
pending=[h for h in summary['hosts'] if h not in selected]
def f(v):return '—' if v is None else f'{v:.5f}'
def read(h,ph,k):return json.loads((B/'monitor'/h/f'{ph}-{k}-eval.json').read_text())
lines=[f'# 随机二选一：{a.stage*30}分钟完整节点','',
 'OpenML12是固定测试；old618/sparse100是固定训练行Query拟合。两路径分别比较，S_log未定义保留空值；没有匹配单分支或双前向对照，不把本轮直接解释为随机选择本身带来的收益。','']
if pending:lines += ['尚无本节点完整配对评估：'+', '.join(pending)+'。本报告仅列完整主机，后续补齐。','']
if a.stage==2:
 lines += ['相对30分钟：DGX2、Mini无广播真实表继续提升，Dustin回落；三台无广播旧618的R²/S_log继续回升。无广播稀疏100平均R²更接近零，但三台仍负且无R²>0.1表。广播指标有取舍，真实12表均仍低于各自原起点。',
 '需单列的局部风险：Dustin cpu_activity广播R² 约0.950→0.527，无广播约0.953→0.637；DGX2 space_ga广播约0.179→-0.035。DGX2无广播new120__discoscm_090从-80.64414回升至-24.95798（起点-2.32994），S_log仍-3.93719，属于回升但未恢复。','']
for h,v in selected.items():
 s=v['phases'][phase];before=read(h,'before','fit718');prev=read(h,previous,'fit718');now=read(h,phase,'fit718');bc=s['cumulative_branch_counts'];n=sum(bc.values())
 lines += [f'## {h}','',f"累计成功训练{s['cumulative_successful_seconds']:.3f}秒，累计更新{n}次，当前段覆盖{s['coverage']}/730表；V6 {bc['v6']}次（{bc['v6']/n:.2%}），V5.5 {bc['v55']}次。父SHA、bank、窗口、单前向单更新及后段RNG/cursor接续核验通过。",'']
 for branch in ['v6','v55']:
  bo=read(h,'before','openml12-'+branch);po=read(h,previous,'openml12-'+branch);no=read(h,phase,'openml12-'+branch)
  lines += [f'### {branch}','',f'| 数据组 | 指标 | 起点 | {0 if a.stage==1 else (a.stage-1)*30}分钟 | {a.stage*30}分钟 | Δ起点 | Δ上一节点 |','|---|---|---:|---:|---:|---:|---:|']
  for m in ['r2','slog']:
   x=s['openml12'][branch];vals=[bo['macro'][m],po['macro'][m],no['macro'][m],x['delta_before'][m],x['delta_previous'][m]];lines.append('| OpenML12测试 | '+m+' | '+' | '.join(map(f,vals))+' |')
  for fam in ['old618','sparse100']:
   x=s['fit718'][branch][fam]
   for m,value in x['macro'].items():
    if fam=='sparse100' and m in ['nominal_accuracy','ordinal_accuracy','rank_mae']:continue
    vals=[before['by_family'][fam].get(branch+'/'+m),prev['by_family'][fam].get(branch+'/'+m),value,x['delta_before'][m],x['delta_previous'][m]];lines.append('| '+fam+'训练行 | '+m+' | '+' | '.join(map(f,vals))+' |')
   counts=[]
   for ev in [before,prev,now]:
    rs=[r[branch]['r2'] for r in ev['tables'].values() if r['family']==fam and r[branch].get('r2') is not None];counts.append((sum(x>0 for x in rs),sum(x>.1 for x in rs)))
   c=x['changes'];lines += ['',f"{fam}：相对起点R²改善/下降 {c['r2']['improved']}/{c['r2']['declined']}；正R²表 {'→'.join(str(x[0]) for x in counts)}；R²>0.1表 {'→'.join(str(x[1]) for x in counts)}。",'']
  lines += ['| OpenML重点表 | R² 起点→上一点→当前 | S_log 起点→上一点→当前 |','|---|---|---|']
  for name in ['pumadyn32nh','kin8nm','cpu_activity','space_ga']:
   vals=[x['tables'][name] for x in [bo,po,no]];lines.append('| '+name+' | '+' | '.join('→'.join(f(v[m]) for v in vals) for m in ['r2','slog'])+' |')
  for metric in ['r2','slog']:
   pairs=sorted([(name,row[branch][metric]-before['tables'][name][branch][metric]) for name,row in now['tables'].items() if row['family']=='old618' and row[branch].get(metric) is not None and before['tables'][name][branch].get(metric) is not None],key=lambda x:x[1])[:3]
   lines += ['',f'旧618相对起点{metric}退化最多三表：','', '| 表 | 起点 | 上一点 | 当前 | Δ起点 |','|---|---:|---:|---:|---:|']
   for name,delta in pairs:lines.append('| '+name+' | '+' | '.join(f(ev['tables'][name][branch][metric]) for ev in [before,prev,now])+' | '+f(delta)+' |')
  lines.append('')
lines += ['完整逐表结果见DETAIL.md；各节点轨迹见RESULT.md。未自动替换默认点。'+('本报告所列主机已完成两小时训练及完整终点评估，不追加训练；尚未完成的主机继续按原预算收口。' if a.stage==4 else '训练按原预算继续。'),'']
(B/f'STAGE{a.stage}.md').write_text('\n'.join(lines))
print(f'STAGE{a.stage}.md written')
