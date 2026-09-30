from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,math,sys
b=Path.home()/'experiments/v6-branchsample50-sparse30-mini-continue60m-20260929';old=b.parent/'v6-branchsample50-sparse30-60m-20260929'
sys.path.insert(0,str(b/'source/src'));sys.path.insert(0,str(b))
from tabu_lab.curriculum_v53.protocol import load_v55_plan,schedule_entry
from branch_sampling import BranchSampler
import torch
c=json.loads((b/'stage1/run/campaign.json').read_text());p=json.loads((old/'stage2/run/campaign.json').read_text())
assert c['outcome']=='training_completed' and c['parent_sha256']==p['checkpoint_sha256']
assert c['initial_cursor']==8544 and not c['schedule_cursor_reset'] and c['branch_selector_restored']
assert c['initial_table_episode_offsets']==p['initial_table_episode_offsets']
assert c['runtime']['device']=='mps' and c['runtime']['dtype']=='float32' and c['allocator_fraction_cap']==.75
assert c['successful_update_seconds_target']==1800 and 1800<=c['successful_update_seconds']<1860
rows=[json.loads(s) for s in (b/'stage1/run/updates.jsonl').read_text().splitlines()]
plan=load_v55_plan(b/'all730-manifest.json')
payload=torch.load(p['checkpoint'],map_location='cpu',weights_only=False)
selector=BranchSampler(payload['state']['branch_rng_state'])
assert payload['state']['cursor']==c['initial_cursor'];del payload
seen=set()
for phase in ['stage1','stage2']:
 for line in (old/phase/'run/updates.jsonl').read_text().splitlines():
  r=json.loads(line);seen.add((r['table'],r['episode_index']))
cycles=defaultdict(list)
for i,r in enumerate(rows):
 assert r['update']==199162+i
 assert r['forward_passes']==r['optimizer_steps']==1 and r['selected_loss_weight']==1
 assert r['supervision']=='target_only' and r['scored_cells']==r['scored_query_cells'] and r['scored_visible_cells']==0
 assert r['train_rows']==min(r['total_train_rows'],512 if r['cohort'] in ('focus2','other10') else 204)
 assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
 t,ep=schedule_entry(plan,0,8544+i)
 assert r['table']==t.name and r['episode_index']==c['initial_table_episode_offsets'][t.name]+ep
 assert r['branch']==selector.draw()
 pair=(r['table'],r['episode_index']);assert pair not in seen;seen.add(pair)
 cycles[(8544+i)//900].append(r)
full=0
for chunk in cycles.values():
 if len(chunk)!=900:continue
 assert Counter(r['cohort'] for r in chunk)==Counter(sparse100=270,history718=618,focus2=2,other10=10)
 assert len({r['table'] for r in chunk})==730
 full+=1
assert len(rows)==sum(c['table_updates'].values())==sum(c['branch_counts'].values())
assert Counter(r['branch'] for r in rows)==Counter(c['branch_counts'])
assert Counter(r['cohort'] for r in rows)==Counter(c['cohort_updates'])
assert c['table_coverage']==len({r['table'] for r in rows})==730
assert hashlib.sha256(Path(c['checkpoint']).read_bytes()).hexdigest()==c['checkpoint_sha256']
assert abs(sum(r['seconds'] for r in rows)-c['successful_update_seconds'])<1e-6
complete=[];pending=[]
for kind in ['openml12-v6','openml12-v55','fit718']:
 f=b/'evaluations/stage1'/kind/'terminal.json'
 if not f.exists():pending.append(kind);continue
 x=json.loads(f.read_text());before=json.loads((b/'evaluations/before'/kind/'terminal.json').read_text())
 assert x['outcome']=='completed' and x['optimizer_updates']==0 and x['checkpoint_sha256']==c['checkpoint_sha256'] and x['bank_sha256']==before['bank_sha256']
 if kind=='fit718':assert x['total_masks']==1436 and x['total_predictions_per_branch']==97648 and len(x['tables'])==718
 else:assert x['total_predictions']==7894
 complete.append(kind)
print(json.dumps(dict(outcome='passed',host='gongqian-mini',checked_utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=c['checkpoint_sha256'],successful_seconds=c['successful_update_seconds'],updates=len(rows),coverage=730,branch_counts=c['branch_counts'],cohort_counts=c['cohort_updates'],complete_weighted_cycles_verified=full,all_step_contracts_verified=True,exact_schedule_and_episode_sequence_verified=True,branch_rng_sequence_verified=True,no_duplicate_episodes_across_continuation=True,checkpoint_hash_verified=True,complete_evaluations=complete,pending_evaluations=pending,audit_note='Sparse episodes are shuffled within the weighted schedule; continuity checked against schedule_entry and duplicate-free pairs, not monotonically increasing episode numbers.')))
