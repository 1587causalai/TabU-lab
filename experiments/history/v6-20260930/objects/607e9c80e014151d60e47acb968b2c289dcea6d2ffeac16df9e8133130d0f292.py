"""Prepare immutable references and reuse exact parent evaluations; no training."""
import pathlib,json,os,hashlib,shutil,sys
B=pathlib.Path(__file__).resolve().parent
host=sys.argv[1]
parent=json.loads((B/'parents.json').read_text())[host]
old=pathlib.Path(parent['manifest']).parent
assert not (B/'status.json').exists() and not (B/'stage1').exists()
assert json.loads((old/'status.json').read_text())['outcome']=='completed'
assert hashlib.sha256(pathlib.Path(parent['checkpoint']).read_bytes()).hexdigest()==parent['checkpoint_sha256']
if not (B/'source').exists(): (B/'source').symlink_to(old/'source',target_is_directory=True)
assert (B/'source').resolve()==(old/'source').resolve()
q=dict(host=host,device=('cuda:0' if host=='dgx2' else 'mps'),parent=parent)
(B/'queue.json').write_text(json.dumps(q,indent=2)+'\n')
spec=json.loads((old/'fit-manifest.json').read_text())
for t in spec['tables']: t['path']=os.path.relpath((old/t['path']).resolve(),B)
(B/'fit-manifest.json').write_text(json.dumps(spec,indent=2)+'\n')
shutil.copy2(old/'fixed-fit-bank.json',B/'fixed-fit-bank.json')
proof={}
for kind in ['openml12-v6','openml12-v55','fit718']:
    src=old/'evaluations/stage4'/kind/'terminal.json';e=json.loads(src.read_text())
    assert e['outcome']=='completed' and e['checkpoint_sha256']==parent['checkpoint_sha256'] and e['optimizer_updates']==0
    if kind=='fit718': assert e['total_masks']==1436 and e['total_predictions_per_branch']==97648 and e['bank_sha256']==json.loads((B/'fixed-fit-bank.json').read_text())['sha256']
    else: assert e['total_predictions']==7894 and e['bank_sha256']==hashlib.sha256((B.parent/'openml12-frozen-icl-20260927/bank.json').read_bytes()).hexdigest()
    dst=B/'evaluations/before'/kind;dst.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst/'terminal.json')
    proof[kind]=dict(source=str(src),terminal_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),checkpoint_sha256=e['checkpoint_sha256'],bank_sha256=e['bank_sha256'])
(B/'before-reuse.json').write_text(json.dumps(proof,indent=2)+'\n')
report=dict(host=host,root=str(B),outcome='prepared',parent_sha256=parent['checkpoint_sha256'],baseline_reuse=proof,code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in B.glob('*.py')})
(B/'prepare.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
