"""Adapt dual718 endpoints to OpenML12 with each host's original V6 objective.

This is a new objective/sampling stage, not strict resume. Model, optimizer, RNG
and table episode positions are inherited; only half1 resets the mixture cursor.
"""
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
from tabu_lab.curriculum_v53.artifacts import atomic_json, append_event, finite_state, sha256
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.protocol import load_v55_plan, schedule_entry
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v53 import V53LossConfig
from tabu_lab.models.restoration_v6 import V6Model, score_training_episode
from tabu_lab.restoration_optimizers import adamw
from training_support import episode_seeds



OPENML_COHORTS = ('focus2', 'other10')
REPLAY_COHORT = 'history718'
OBJECTIVE = 'v6_single_branch_openml12_adaptation'


def settled_offsets(plan, donor, names=None):
    """Advance saved table origins past all updates represented by donor cursor."""
    wanted = {t.name for t in plan.tables} if names is None else set(names)
    consumed = {name: 0 for name in wanted}
    for cursor in range(donor['state']['cursor']):
        table, episode = schedule_entry(plan, 0, cursor)
        if table.name in consumed:
            consumed[table.name] = max(consumed[table.name], episode + 1)
    return {name: donor['state']['table_episode_offsets'][name] + used
            for name, used in consumed.items()}


def rewrite_spec(spec, path_base, base, host):
    """Use the complete 730-table template; do not add any replay entries."""
    spec = copy.deepcopy(spec)
    ids = [t['id'] for t in spec['tables']]
    assert len(ids) == len(set(ids)) == 730
    assert sum(t['cohort'] == REPLAY_COHORT for t in spec['tables']) == 718
    assert sum(t['cohort'] in OPENML_COHORTS for t in spec['tables']) == 12
    assert sum(t['id'].startswith('sparse_') for t in spec['tables']
               if t['cohort'] == REPLAY_COHORT) == 100
    sampling = [dict(cohort='focus2', episodes=2), dict(cohort='other10', episodes=10),
                dict(cohort=REPLAY_COHORT, episodes=8)]
    assert spec['stages'][0]['sampling'] == sampling
    spec['stages'][0]['max_updates'] = 1000000
    spec['experiment_id'] = f'v6-dual718-openml12-adapt60m-{host}-20260928'
    spec['description'] = ('OpenML12 adaptation after dual718: host original V6 objective, '
                           'window512, 40 percent equal-table history718 replay; '
                           'new stage retaining weights optimizer RNG and episode positions.')
    for table in spec['tables']:
        if table['cohort'] in OPENML_COHORTS:
            table['window_rows'] = 512
        else:
            assert table['cohort'] == REPLAY_COHORT and table['window_rows'] == 204
        # Resolve even absolute source paths before making them relative to this run.
        table['path'] = os.path.relpath((path_base / table['path']).resolve(), base)
    return spec


def validate_plan(plan):
    assert len(plan.tables) == 730
    assert sum(t.cohort == REPLAY_COHORT for t in plan.tables) == 718
    assert all(t.window_rows == (512 if t.cohort in OPENML_COHORTS else 204)
               for t in plan.tables)
    stage = plan.spec['stages'][0]
    assert all(r['kind'] == 'supervised_row' for r in stage['recipe'].values())
    first_cycle = [schedule_entry(plan, 0, i)[0] for i in range(20)]
    targets = [t.name for t in first_cycle if t.cohort in OPENML_COHORTS]
    assert len(targets) == len(set(targets)) == 12
    assert sum(t.cohort == REPLAY_COHORT for t in first_cycle) == 8
    return targets


def main(args):
    base = Path(__file__).resolve().parent
    root = base/args.segment/'run'
    root.mkdir(parents=True, exist_ok=False)
    receipt = dict(outcome='starting', host=args.host, parent=args.parent,
                   parent_sha256=args.parent_sha, successful_update_seconds_target=args.seconds,
                   continuation=('new_objective_stage_preserve_weights_optimizer_rng_episode_positions'
                                 if args.reset_sampling_cursor else 'continue_adaptation_stage_preserve_cursor'),
                   strict_resume=False, objective=OBJECTIVE, forward_passes_per_update=1,
                   optimizer_steps_per_update=1,
                   supervision=args.supervision,
                   sampling='each OpenML table once plus 8 equal-table history718 per 20 updates; OpenML window512')
    atomic_json(root/'campaign.json', receipt)
    try:
        runtime = configure_runtime(args.device)
        cap=0.75 if args.host=='gongqian-mini' else 0.5
        if args.device=='mps': torch.mps.set_per_process_memory_fraction(cap)
        else: torch.cuda.set_per_process_memory_fraction(cap)
        receipt['allocator_fraction_cap']=cap
        old_plan = load_v55_plan(args.manifest)
        digest = sha256(args.parent)
        donor = torch.load(args.parent, map_location='cpu', weights_only=False)
        assert donor['schema'] == 'tabu.v6.weights-only-checkpoint.v1'
        assert digest == args.parent_sha and donor['purpose'] == 'training'
        assert donor['model_config'] == old_plan.config.as_dict()
        assert donor['identity']['parent_manifest_sha256'] == sha256(args.manifest)
        reset_sampling = args.reset_sampling_cursor
        assert reset_sampling == (args.segment == 'half1')
        template_provenance = None
        if reset_sampling:
            assert len(old_plan.tables) == 718
            assert donor['identity']['objective'] == '0.5*v6_target_only+0.5*v55_query_squared'
            assert donor['identity']['weights'] == [0.5, 0.5]
            assert all((args.openml_template_manifest, args.openml_template_checkpoint,
                        args.openml_template_sha))
            template = load_v55_plan(args.openml_template_manifest)
            assert sha256(args.openml_template_checkpoint) == args.openml_template_sha
            history = torch.load(args.openml_template_checkpoint, map_location='cpu', weights_only=False)
            assert history['model_config'] == donor['model_config']
            assert history['identity']['supervision'] == args.supervision
            assert history['identity']['parent_manifest_sha256'] == sha256(args.openml_template_manifest)
            assert template.optimizer == old_plan.optimizer
            assert episode_seeds(template) == episode_seeds(old_plan)
            replay_names = {t.name for t in template.tables if t.cohort == REPLAY_COHORT}
            assert replay_names == {t.name for t in old_plan.tables}
            offsets = settled_offsets(old_plan, donor)
            openml_names = {t.name for t in template.tables if t.cohort in OPENML_COHORTS}
            offsets.update(settled_offsets(template, history, openml_names))
            spec = rewrite_spec(template.spec, Path(args.openml_template_manifest).parent, base, args.host)
            template_provenance = dict(checkpoint_sha256=args.openml_template_sha,
                                       manifest_sha256=sha256(args.openml_template_manifest),
                                       source='pre_dual_OpenML12_episode_positions_and_host_objective')
            del history
        else:
            assert donor['identity']['objective'] == OBJECTIVE
            assert donor['identity']['supervision'] == args.supervision
            offsets = copy.deepcopy(donor['state']['table_episode_offsets'])
            spec = rewrite_spec(old_plan.spec, Path(args.manifest).parent, base, args.host)
            template_provenance = donor['identity'].get('openml_template')
        assert set(offsets) == {t['id'] for t in spec['tables']}
        manifest = base / 'balanced-replay718-manifest.json'
        atomic_json(manifest, spec)
        plan = load_v55_plan(manifest)
        stage = plan.spec['stages'][0]
        assert plan.optimizer == old_plan.optimizer
        targets = validate_plan(plan)
        replay_cohort = REPLAY_COHORT

        model = V6Model(plan.config, supervision=args.supervision).to(args.device, dtype=execution_dtype(args.device))
        model.load_state_dict(donor['model'], strict=True)
        optimizer = adamw(model, plan.optimizer)
        optimizer.load_state_dict(donor['optimizer'])
        restore_rng(donor['rng'])
        state = dict(update=donor['state']['update'], cursor=(0 if reset_sampling else donor['state']['cursor']), successful_update_seconds=0.0,
                     exposure={'focus2':0,'other10':0,replay_cohort:0},
                     parent_update=donor['state']['update'], parent_cursor=donor['state']['cursor'],
                     table_episode_offsets=offsets, table_updates={t.name:0 for t in plan.tables})
        files = [base/'train.py', base/'training_support.py',
                 base/'source/src/tabu_lab/models/restoration_v53/readout.py',
                 *sorted((base/'source/src/tabu_lab/models/restoration_v6').glob('*.py'))]
        identity = dict(schema='tabu.v6.dual718-openml12-adaptation.v1', supervision=args.supervision,
                        objective=OBJECTIVE, forward_passes_per_update=1, optimizer_steps_per_update=1,
                        strict_resume=False, openml_template=template_provenance,
                        parent_sha256=digest, parent_identity_sha256=donor['identity']['sha256'],
                        model_config=plan.config.as_dict(), runtime=runtime,
                        parent_manifest_sha256=sha256(manifest), sampling=stage['sampling'],
                        code={str(f.relative_to(base)):sha256(f) for f in files})
        identity['sha256'] = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        del donor
        receipt.update(outcome='admitted', runtime=runtime, identity=identity,
                       parent_update=state['parent_update'], balanced_manifest=str(manifest),
                       optimizer_restored=True, rng_restored=True, schedule_cursor_reset=reset_sampling,
                       per_table_episode_positions_preserved=True, first_cycle=targets,
                       openml_template=template_provenance, replay_table_count=718, total_table_count=730)
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
            assert inputs.query.shape[0] == min(table.window_rows, table.train_rows)
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
                cohort=table.cohort,replay_family=('sparse100' if table.name.startswith('sparse_') else 'old618') if table.cohort==replay_cohort else None,episode_index=episode,seconds=seconds,
                successful_update_seconds=state['successful_update_seconds'],loss=float(score.loss.detach()),
                gradient_norm=float(norm),query_rows=score.query_rows,scored_cells=score.scored_cells,
                train_rows=inputs.query.shape[0],total_train_rows=table.train_rows,support_rows=inputs.query.shape[0]-score.query_rows,
                forward_passes=1,optimizer_steps=1,supervision=args.supervision,
                accelerator_allocated_bytes=(torch.mps.current_allocated_memory() if args.device=='mps' else torch.cuda.memory_allocated()),
                accelerator_reserved_bytes=(torch.mps.driver_allocated_memory() if args.device=='mps' else torch.cuda.memory_reserved())))
            if state['successful_update_seconds']-last_save>=300 or state['successful_update_seconds']>=args.seconds:
                path,checksum=checkpoint(root,model,optimizer,state,identity)
                receipt.update(outcome='training_completed' if state['successful_update_seconds']>=args.seconds else 'running',
                    checkpoint=str(path),checkpoint_sha256=checksum,checkpoint_update=state['update'],
                    successful_update_seconds=state['successful_update_seconds'],cohort_updates=state['exposure'],
                    openml_table_updates={t.name:state['table_updates'][t.name] for t in plan.tables if t.cohort in OPENML_COHORTS},
                    table_updates=state['table_updates'],table_coverage=sum(v>0 for v in state['table_updates'].values()),
                    replay_family_updates={family:sum(state['table_updates'][t.name] for t in plan.tables
                        if t.cohort==REPLAY_COHORT and ('sparse100' if t.name.startswith('sparse_') else 'old618')==family)
                        for family in ('old618','sparse100')})
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
    for name in ('openml-template-manifest','openml-template-checkpoint','openml-template-sha'):
        parser.add_argument('--'+name)
    main(parser.parse_args())
