"""Continue each host with unchanged objective, optimizer and episode stream."""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_support import checkpoint, restore_rng
from tabu_lab.curriculum_v53.artifacts import atomic_json, append_event, finite_state, load_checkpoint, sha256
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.protocol import load_v55_plan, schedule_entry
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v53 import V53LossConfig
from tabu_lab.models.restoration_v6 import V6Model, score_training_episode
from tabu_lab.restoration_optimizers import adamw
from training_support import episode_seeds


def main(args):
    base = Path(__file__).resolve().parent
    root = base/args.segment/'run'
    root.mkdir(parents=True, exist_ok=False)
    receipt = dict(outcome='starting', host=args.host, parent=args.parent,
                   parent_sha256=args.parent_sha, successful_update_seconds_target=args.seconds,
                   continuation='new_stage_preserve_weights_optimizer_rng_sampling_cursor_episode_positions',
                   supervision=args.supervision,
                   sampling='each OpenML table once plus 8 old618 per 20 updates; OpenML window2048')
    atomic_json(root/'campaign.json', receipt)
    try:
        runtime = configure_runtime(args.device)
        if args.device=='mps': torch.mps.set_per_process_memory_fraction(0.5)
        else: torch.cuda.set_per_process_memory_fraction(0.5)
        receipt['allocator_fraction_cap']=0.5
        old_plan = load_v55_plan(args.manifest)
        digest = sha256(args.parent)
        donor = torch.load(args.parent, map_location='cpu', weights_only=False)
        assert donor['schema'] == 'tabu.v6.weights-only-checkpoint.v1'
        assert digest == args.parent_sha and donor['purpose'] == 'training'
        assert donor['identity']['supervision'] == args.supervision
        assert donor['model_config'] == old_plan.config.as_dict()
        assert donor['identity']['parent_manifest_sha256'] == sha256(args.manifest)
        offsets = copy.deepcopy(donor['state']['table_episode_offsets'])

        reset_sampling = args.reset_sampling_cursor
        if reset_sampling:
            consumed = {t.name:0 for t in old_plan.tables}
            for cursor in range(donor['state']['cursor']):
                table, episode = schedule_entry(old_plan,0,cursor)
                consumed[table.name] = max(consumed[table.name],episode+1)
            offsets = {name:offsets[name]+consumed[name] for name in offsets}
        spec = copy.deepcopy(old_plan.spec)
        spec['stages'][0]['sampling'] = [dict(cohort='focus2',episodes=2),
                                        dict(cohort='other10',episodes=10),dict(cohort='history618',episodes=8)]
        # The time budget governs stopping; keep the safety cursor cap out of its way.
        spec['stages'][0]['max_updates'] = max(spec['stages'][0]['max_updates'],1000000)
        spec['experiment_id'] = f'v6-window2048-replay40-30m-{args.host}-20260928'
        spec['description'] = 'Random training window capped at 2048; random one-third target Query; 40 percent old618 replay; preserve optimizer and host loss.'
        for table in spec['tables']:
            if table['cohort'] != 'history618': table['window_rows'] = 2048
            table['path'] = os.path.relpath((Path(args.manifest).parent/table['path']).resolve(), base)
        manifest = base/'balanced-manifest.json'
        atomic_json(manifest, spec)
        plan = load_v55_plan(manifest)
        stage = plan.spec['stages'][0]
        assert {k:v for k,v in stage.items() if k not in ('sampling','max_updates')} == {k:v for k,v in old_plan.spec['stages'][0].items() if k not in ('sampling','max_updates')}
        first_cycle = [schedule_entry(plan,0,i)[0] for i in range(20)]
        targets = [t.name for t in first_cycle if t.cohort!='history618']
        assert len(targets)==len(set(targets))==12
        assert sum(t.cohort=='history618' for t in first_cycle)==8
        assert all(t.window_rows==2048 for t in first_cycle if t.cohort!='history618')

        model = V6Model(plan.config, supervision=args.supervision).to(args.device, dtype=execution_dtype(args.device))
        model.load_state_dict(donor['model'], strict=True)
        optimizer = adamw(model, plan.optimizer)
        optimizer.load_state_dict(donor['optimizer'])
        restore_rng(donor['rng'])
        state = dict(update=donor['state']['update'], cursor=(0 if reset_sampling else donor['state']['cursor']), successful_update_seconds=0.0,
                     exposure={'focus2':0,'other10':0,'history618':0},
                     parent_update=donor['state']['update'], parent_cursor=donor['state']['cursor'],
                     table_episode_offsets=offsets, table_updates={t.name:0 for t in plan.tables})
        files = [base/'train_jointall.py', base/'training_support.py',
                 base/'source/src/tabu_lab/models/restoration_v53/readout.py',
                 *sorted((base/'source/src/tabu_lab/models/restoration_v6').glob('*.py'))]
        identity = dict(schema='tabu.v6.jointall-continuation.v1', supervision=args.supervision,
                        parent_sha256=digest, parent_identity_sha256=donor['identity']['sha256'],
                        model_config=plan.config.as_dict(), runtime=runtime,
                        parent_manifest_sha256=sha256(manifest), sampling=stage['sampling'],
                        code={str(f.relative_to(base)):sha256(f) for f in files})
        identity['sha256'] = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        del donor
        receipt.update(outcome='admitted', runtime=runtime, identity=identity,
                       parent_update=state['parent_update'], balanced_manifest=str(manifest),
                       optimizer_restored=True, rng_restored=True, schedule_cursor_reset=reset_sampling,
                       per_table_episode_positions_preserved=True, first_cycle=targets)
        atomic_json(root/'campaign.json', receipt)
        seeds = episode_seeds(plan)
        loss_config = V53LossConfig(**stage['loss'])
        last_save = 0
        while state['successful_update_seconds'] < args.seconds:
            table, relative_episode = schedule_entry(plan,0,state['cursor'])
            episode = offsets[table.name] + relative_episode
            start = time.monotonic()
            inputs, request, truth, info = build_episode(table,stage['recipe'][table.kind],episode,
                seeds,args.device,epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
            if table.cohort != 'history618': assert inputs.query.shape[0] == min(2048, table.train_rows)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            score=(score_training_episode(model,inputs,truth) if args.supervision=='joint_all'
                   else score_training_episode(model,inputs,truth,request=request,loss_config=loss_config))
            score.loss.backward()
            gradients=[p for p in model.parameters() if p.grad is not None]
            if not gradients or not finite_state([p.grad for p in gradients]):
                raise FloatingPointError('missing or nonfinite gradients')
            norm=torch.nn.utils.clip_grad_norm_(gradients,plan.optimizer.grad_clip,error_if_nonfinite=True)
            optimizer.step()
            if not finite_state(model.state_dict()) or not finite_state(optimizer.state_dict()):
                raise FloatingPointError('nonfinite state')
            torch.mps.synchronize() if args.device=='mps' else torch.cuda.synchronize()
            seconds=time.monotonic()-start
            state['successful_update_seconds']+=seconds
            state['update']+=1
            state['cursor']+=1
            state['exposure'][table.cohort]+=1
            state['table_updates'][table.name]+=1
            append_event(root/'updates.jsonl',dict(update=state['update'],table=table.name,
                cohort=table.cohort,episode_index=episode,seconds=seconds,
                successful_update_seconds=state['successful_update_seconds'],loss=float(score.loss.detach()),
                gradient_norm=float(norm),query_rows=score.query_rows,scored_cells=score.scored_cells,
                train_rows=inputs.query.shape[0],total_train_rows=table.train_rows,support_rows=inputs.query.shape[0]-score.query_rows,
                accelerator_allocated_bytes=(torch.mps.current_allocated_memory() if args.device=='mps' else torch.cuda.memory_allocated()),
                accelerator_reserved_bytes=(torch.mps.driver_allocated_memory() if args.device=='mps' else torch.cuda.memory_reserved())))
            if state['successful_update_seconds']-last_save>=300 or state['successful_update_seconds']>=args.seconds:
                path,checksum=checkpoint(root,model,optimizer,state,identity)
                receipt.update(outcome='training_completed' if state['successful_update_seconds']>=args.seconds else 'running',
                    checkpoint=str(path),checkpoint_sha256=checksum,checkpoint_update=state['update'],
                    successful_update_seconds=state['successful_update_seconds'],cohort_updates=state['exposure'],
                    openml_table_updates={t.name:state['table_updates'][t.name] for t in plan.tables if t.cohort!='history618'})
                atomic_json(root/'campaign.json',receipt)
                last_save=state['successful_update_seconds']
                print(json.dumps({k:receipt[k] for k in ('outcome','checkpoint_update','successful_update_seconds')}),flush=True)
    except Exception as error:
        receipt.update(outcome='failed',error_type=type(error).__name__,error=str(error))
        atomic_json(root/'campaign.json',receipt)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('host','device','parent','parent-sha','manifest'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--supervision', choices=['target_only','joint_all'], required=True)
    parser.add_argument('--segment', choices=['half1','half2'], required=True)
    parser.add_argument('--seconds',type=float,default=1800)
    parser.add_argument('--reset-sampling-cursor',action='store_true')
    main(parser.parse_args())
