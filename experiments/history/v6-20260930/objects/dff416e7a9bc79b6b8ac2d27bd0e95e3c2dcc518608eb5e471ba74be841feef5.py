"""Bernoulli branch sampling: one complete branch loss, one forward, one update."""
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
from tabu_lab.models.restoration_v6 import score_training_episode
from dual import DualModel, prepare, score as score_branch
from branch_sampling import BranchSampler, BRANCH_CONFIG
from tabu_lab.restoration_optimizers import adamw
from training_support import episode_seeds



OPENML_COHORTS = ('focus2', 'other10')
REPLAY_COHORT = 'history718'
OBJECTIVE = 'v6_v55_bernoulli50_all730'


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
                dict(cohort=REPLAY_COHORT, episodes=718)]
    spec['stages'][0]['sampling'] = sampling
    spec['stages'][0]['max_updates'] = 1000000
    spec['experiment_id'] = f'v6-branchsample50-all730-2h-{host}-20260929'
    spec['stages'][0]['loss'] = dict(state_weights=[0.0,1.0,0.0,0.0], discrete_weight=1.0)
    spec['description'] = ('Equal table all730 stochastic broadcast/unbroadcast branch, '
                           'OpenML window512, other tables204, no replay/foreground weighting; '
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
    first_cycle = [schedule_entry(plan, 0, i)[0] for i in range(730)]
    targets = [t.name for t in first_cycle if t.cohort in OPENML_COHORTS]
    assert len(targets) == len(set(targets)) == 12
    assert sum(t.cohort == REPLAY_COHORT for t in first_cycle) == 718
    assert len({t.name for t in first_cycle}) == 730
    return targets


def main(args):
    base = Path(__file__).resolve().parent
    root = base/args.segment/'run'
    root.mkdir(parents=True, exist_ok=False)
    receipt = dict(outcome='starting', host=args.host, parent=args.parent,
                   parent_sha256=args.parent_sha, successful_update_seconds_target=args.seconds,
                   continuation='new_branchsample50_stage_preserve_optimizer_rng_episode_cursor',
                   strict_resume=False, objective=OBJECTIVE, forward_passes_per_update=1,
                   optimizer_steps_per_update=1,
                   supervision=args.supervision,
                   sampling='each of all730 tables once per shuffled730-update cycle; OpenML512 other204')
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
        assert not reset_sampling, 'Continuation preserves the existing sampling cursor'
        template_provenance = None
        if reset_sampling:
            assert len(old_plan.tables) == 730
            offsets = settled_offsets(old_plan, donor)
        else:
            assert donor['identity']['objective'] in (OBJECTIVE, 'v6_all730_host_selected_loss')
            offsets = copy.deepcopy(donor['state']['table_episode_offsets'])
        spec = rewrite_spec(old_plan.spec, Path(args.manifest).parent, base, args.host)
        assert set(offsets) == {t['id'] for t in spec['tables']}
        manifest = base / 'all730-manifest.json'
        atomic_json(manifest, spec)
        plan = load_v55_plan(manifest)
        stage = plan.spec['stages'][0]
        assert plan.optimizer == old_plan.optimizer
        targets = validate_plan(plan)
        replay_cohort = REPLAY_COHORT

        model = DualModel(plan.config).to(args.device, dtype=execution_dtype(args.device))
        model.load_state_dict(donor['model'], strict=True)
        optimizer = adamw(model, plan.optimizer)
        optimizer.load_state_dict(donor['optimizer'])
        restore_rng(donor['rng'])
        state = dict(update=donor['state']['update'], cursor=(0 if reset_sampling else donor['state']['cursor']), successful_update_seconds=0.0,
                     exposure={'focus2':0,'other10':0,replay_cohort:0},
                     parent_update=donor['state']['update'], parent_cursor=donor['state']['cursor'],
                     table_episode_offsets=offsets, table_updates={t.name:0 for t in plan.tables})
        files = [base/'train.py', base/'training_support.py', base/'experiment.json', base/'dual.py', base/'branch_sampling.py',
                 base/'source/src/tabu_lab/models/restoration_v53/readout.py',
                 *sorted((base/'source/src/tabu_lab/models/restoration_v6').glob('*.py'))]
        identity = dict(branch_sampling=BRANCH_CONFIG, schema='tabu.v6.branchsample50.v1', supervision=args.supervision,
                        objective=OBJECTIVE, loss_config=stage['loss'], loss_scope=('query_rows_all_cells' if args.supervision=='joint_all' else 'query_target_cells'), response=('own_plus_target_encoding' if args.supervision=='joint_all' else 'own_cell_encoding'), forward_passes_per_update=1, optimizer_steps_per_update=1,
                        strict_resume=False, openml_template=template_provenance,
                        parent_sha256=digest, parent_identity_sha256=donor['identity']['sha256'],
                        model_config=plan.config.as_dict(), runtime=runtime,
                        parent_manifest_sha256=sha256(manifest), sampling=stage['sampling'],
                        code={str(f.relative_to(base)):sha256(f) for f in files})
        identity['sha256'] = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        selector = BranchSampler(donor['state'].get('branch_rng_state'))
        state['branch_rng_state'] = selector.state()
        state['branch_counts'] = {'v6':0,'v55':0}
        selector_restored = 'branch_rng_state' in donor['state']
        del donor
        receipt.update(outcome='admitted', runtime=runtime, identity=identity,
                       parent_update=state['parent_update'], balanced_manifest=str(manifest),
                       optimizer_restored=True, rng_restored=True, schedule_cursor_reset=reset_sampling, branch_selector_restored=selector_restored, branch_sampling=BRANCH_CONFIG,
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
            branch=selector.draw()
            prepared=prepare(model,inputs,request,truth,loss_config)
            score=score_branch(model,prepared,loss_config,branch)
            qrows=inputs.query.any(1)
            expected=int(((truth.states>=0)&qrows[:,None]).sum()) if args.supervision=='joint_all' else int(inputs.query.sum())
            assert score.scored_cells == expected
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
            state['branch_counts'][branch]+=1
            state['branch_rng_state']=selector.state()
            state['update']+=1
            state['cursor']+=1
            state['exposure'][table.cohort]+=1
            state['table_updates'][table.name]+=1
            append_event(root/'updates.jsonl',dict(update=state['update'],table=table.name,
                cohort=table.cohort,replay_family=('sparse100' if table.name.startswith('sparse_') else 'old618') if table.cohort==replay_cohort else None,episode_index=episode,seconds=seconds,
                successful_update_seconds=state['successful_update_seconds'],loss=float(score.loss.detach()),
                gradient_norm=float(norm),query_rows=score.query_rows,scored_cells=score.scored_cells,
                scored_visible_cells=(int(((truth.states==0)&qrows[:,None]).sum()) if args.supervision=='joint_all' else 0),scored_query_cells=int(inputs.query.sum()),
                columns=inputs.query.shape[1],objective=OBJECTIVE,branch=branch,selected_loss_weight=1.0,branch_counts=dict(state['branch_counts']),
                train_rows=inputs.query.shape[0],total_train_rows=table.train_rows,support_rows=inputs.query.shape[0]-score.query_rows,
                forward_passes=1,optimizer_steps=1,supervision=args.supervision,
                accelerator_allocated_bytes=(torch.mps.current_allocated_memory() if args.device=='mps' else torch.cuda.memory_allocated()),
                accelerator_reserved_bytes=(torch.mps.driver_allocated_memory() if args.device=='mps' else torch.cuda.memory_reserved())))
            if state['successful_update_seconds']-last_save>=300 or state['successful_update_seconds']>=args.seconds:
                path,checksum=checkpoint(root,model,optimizer,state,identity)
                receipt.update(outcome='training_completed' if state['successful_update_seconds']>=args.seconds else 'running',
                    checkpoint=str(path),checkpoint_sha256=checksum,checkpoint_update=state['update'],
                    successful_update_seconds=state['successful_update_seconds'],cohort_updates=state['exposure'],branch_counts=state['branch_counts'],
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
    parser.add_argument('--supervision', choices=['target_only'], required=True)
    parser.add_argument('--segment', choices=['stage1','stage2','stage3','stage4'], required=True)
    parser.add_argument('--seconds',type=float,default=3600)
    parser.add_argument('--reset-sampling-cursor',action='store_true')
    for name in ('openml-template-manifest','openml-template-checkpoint','openml-template-sha'):
        parser.add_argument('--'+name)
    main(parser.parse_args())
