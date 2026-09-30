import json
from pathlib import Path
B=Path(__file__).resolve().parent
cfg=json.loads((B/'experiment.json').read_text())
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def num(x):return '待完成' if x is None else f'{x:.5f}'
summary={'hosts':{},'tables':{},'protocol':'fit-only updates/support; validation-selected; no refit; pretrained monitor exposure'}
lines=['# OpenML12 单表早停探索','', '一档配置、一次固定划分；不按测试择优。TabU为逐表独立微调，不是同一checkpoint同时拟合12表。验证行被父点历史训练过，测试也不是全新盲测；不证明架构单独因果优势。','', '| 表 | MLP测试R² | XGB测试R² | Dustin/V6训练R² | Dustin/V6测试R² | DGX/V5.5训练R² | DGX/V5.5测试R² |','|---|---:|---:|---:|---:|---:|---:|']
for host in cfg['hosts']:summary['hosts'][host]=read(B/'monitor'/host/'status.json')
for name in cfg['tables']:
 row={}
 baseline=read(B/'monitor/dgx2/tables'/name/'baselines/terminal.json')
 for model,x in baseline.get('models',{}).items():row[model]=x
 for host in cfg['hosts']:
  s=read(B/'monitor'/host/'tables'/name/'status.json');tr=s.get('trials',{}).get('lr1e-4',{})
  row[host]={'outcome':s.get('outcome','pending'),'metrics':{k:v['metrics'] for k,v in tr.get('final_evaluations',{}).items()},'nodes':tr.get('nodes',{}),'selected_node':tr.get('selected_node'),'seconds':tr.get('successful_update_seconds',0),'updates':tr.get('updates',0)}
 summary['tables'][name]=row
 def metric(model,group,key='r2'):return row.get(model,{}).get('metrics',{}).get(group,{}).get(key)
 vals=[metric('mlp-256x3','test'),metric('xgboost-depth6','test'),metric('dustinstudio','train'),metric('dustinstudio','test'),metric('dgx2','train'),metric('dgx2','test')]
 lines.append('| '+name+' | '+' | '.join(num(v) for v in vals)+' |')
lines+=['','训练R²：TabU为更新行三分区隐藏目标恢复；MLP/XGB为更新行直接拟合，机制不同。所有测试只使用更新行所提供的监督信息，不进行全量重训。','']
for name,row in summary['tables'].items():
 lines+=['## '+name,'','| 模型 | 选择节点 | 训练R² | 测试R² | 测试S_log | 测试MSE |','|---|---|---:|---:|---:|---:|']
 for model,r in row.items():
  m=r.get('metrics',{});te=m.get('test',{});tr=m.get('train',{});lines.append('| '+model+' | '+str(r.get('selected_node',r.get('selected_rounds','待完成')))+' | '+' | '.join(num(v) for v in [tr.get('r2'),te.get('r2'),te.get('slog'),te.get('mse')])+' |')
 lines.append('')
summary['all_complete']=all(summary['hosts'][h].get('outcome')=='completed' for h in cfg['hosts'])
(B/'RESULT.md').write_text('\n'.join(lines)+'\n');(B/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'all_complete':summary['all_complete']}))
