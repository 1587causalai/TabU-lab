"""Independent per-table fine-tuning with a successful-update-time budget."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import signal
import sys
import time
from pathlib import Path

import torch

from tabu_lab.curriculum_v53 import runner
from tabu_lab.curriculum_v53.artifacts import atomic_json, load_checkpoint, sha256
from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.models.restoration._dtype import execution_dtype

sys.path.insert(0, str(Path(__file__).parent))
from frozen_evaluate import make_input, metrics, state_hash


def json_read(path):
    return json.loads(Path(path).read_text())


def make_plans(args, root, bank, parent_spec):
    (root/'data').mkdir()
    (root/'manifests').mkdir()
    parent_stage = parent_spec['stages'][0]
    paths = {}
    conversion = []
    for entry in bank['tables']:
        name = entry['name']
        original = Path(args.bank_root)/entry['path']
        assert sha256(original) == entry['sha256']
        data = json_read(original)
        # Historical TAR defaults all predictors to numeric. Declare that same
        # schema explicitly for V5.5; values and split row addresses are untouched.
        if data.get('features') is None:
            data['features'] = [dict(kind='numeric', domain=[]) for _ in range(entry['width'])]
        dest = root/'data'/f'{name}.json'
        atomic_json(dest, data)
        check = json_read(dest)
        assert check['values'] == json_read(original)['values']
        assert check['splits'] == json_read(original)['splits']
        window = entry['train_n'] if entry['cohort'] == 'old3' else min(204, entry['train_n'])
        fraction = math.ceil(window/3)/window
        manifest = dict(schema='tabu.curriculum.v55.v1',
            experiment_id=f'openml12-independent-{args.host}-{name}-20260927',
            description='Independent table adaptation from joint618; fixed final endpoint; no test-driven selection',
            model=parent_spec['model'], optimizer=parent_spec['optimizer'],
            seeds=dict(model=20260908, order=20260909, masks=20260910,
                       codes=20260911, windows=20260912, evaluation=20260913),
            tables=[dict(id=name, path=f'../data/{name}.json', sha256=sha256(dest),
                         cohort='target', kind='real', role='train',
                         window_rows=window, target_column=entry['width']-1)],
            probes=[], stages=[dict(name='target_finetune', question='Adapt to target train split',
                max_updates=100000, max_seconds=max(1800.0, args.seconds*8),
                sampling=[dict(cohort='target', episodes=1)],
                recipe=dict(real=dict(kind='supervised_row', fraction=fraction)),
                optimizer='adamw', evaluate_every=100000, checkpoint_every=250,
                probes=[], loss=parent_stage['loss'],
                objective=parent_stage.get('objective', dict(kind='squared')))])
        path = root/'manifests'/f'{name}.json'
        atomic_json(path, manifest)
        plan = load_v55_plan(path)
        paths[name] = path
        conversion.append(dict(name=name, original_sha256=entry['sha256'],
                               training_sha256=sha256(dest), values_and_splits_equal=True,
                               window=window, query_count=math.ceil(window/3),
                               identity_sha256=plan.identity['sha256']))
    atomic_json(root/'data-adaptation.json', conversion)
    return paths


def evaluate_one(args, root, entry, plan, checkpoint, output):
    payload, digest = load_checkpoint(checkpoint)
    assert payload['identity'] == plan.identity
    model = make_model(plan).to(device=args.device, dtype=execution_dtype(args.device))
    model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    before = state_hash(model)
    data = json_read(Path(args.bank_root)/entry['path'])
    rows, seen = [], set()
    for trace in entry['traces']:
        inputs, request = make_input(data, trace, entry['name'], args.device, execution_dtype(args.device))
        with torch.inference_mode():
            result = model(inputs, request)
        assert len(result.columns) == 1 and result.columns[0].result.status == 'ok'
        col = result.columns[0]
        values = col.decoded.detach().cpu().tolist()
        positions = col.target_indices.detach().cpu().tolist()
        targets = request.targets.detach().cpu().tolist()
        for value, pos in zip(values, positions):
            local_row, column = targets[pos]
            assert column == entry['width']-1 and math.isfinite(value)
            row_id = trace['query_row_ids'][local_row-len(trace['context_row_ids'])]
            assert row_id not in seen
            seen.add(row_id)
            rows.append(dict(row_id=row_id, target=float(data['values'][row_id][-1]),
                             prediction=value, episode_id=trace['episode_id']))
        del result, col, inputs, request
    assert seen == set(data['splits']['test']) and len(rows) == entry['test_n']
    rows.sort(key=lambda row: row['row_id'])
    assert state_hash(model) == before and sha256(checkpoint) == digest
    atomic_json(output/'test-predictions.json', rows)
    result = dict(outcome='completed', checkpoint_sha256=digest,
                  checkpoint_update=payload['state']['update'], metrics=metrics(rows),
                  historical_trace_sha256=entry['trace_sha256'], model_state_unchanged=True,
                  optimizer_updates_during_evaluation=0)
    atomic_json(output/'test-result.json', result)
    del model, payload
    gc.collect()
    if args.device == 'mps':
        torch.mps.empty_cache()
    return result


def run(args):
    root = Path(args.root)
    assert not (root/'campaign.json').exists(), 'Existing campaign must not be overwritten'
    runtime = runner.configure_runtime(args.device)
    parent, parent_digest = load_checkpoint(args.parent)
    assert parent_digest == args.parent_sha
    parent_plan = load_v55_plan(args.parent_manifest)
    assert parent['identity'] == parent_plan.identity
    bank = json_read(Path(args.bank_root)/'bank.json')
    paths = make_plans(args, root, bank, json_read(args.parent_manifest))
    receipt = dict(schema='tabu.openml12.independent-finetune.v1', outcome='prepared',
        host=args.host, parent_checkpoint=args.parent, parent_sha256=parent_digest,
        parent_update=parent['state']['update'], runtime=runtime,
        successful_update_seconds_per_table=args.seconds, mode='independent_per_table',
        continuation='weights-only per table; strict resume after one-update admission',
        bank_sha256=sha256(Path(args.bank_root)/'bank.json'),
        script_sha256=sha256(__file__), datasets={},
        selection='Fixed budget final checkpoint; no test-based tuning or selection',
        exposure_audit='Not performed, per owner request')
    atomic_json(root/'campaign.json', receipt)
    if args.prepare_only:
        print(json.dumps(dict(outcome='prepared', tables=len(paths))), flush=True)
        return
    del parent
    native_train_step = runner.train_step
    try:
        for entry in bank['tables']:
            name = entry['name']
            receipt.update(outcome='running', active_table=name)
            atomic_json(root/'campaign.json', receipt)
            table_root = root/'runs'/name
            table_root.mkdir(parents=True)
            plan = load_v55_plan(paths[name])
            admission = runner.run(plan, table_root/'admission', device=args.device,
                initialize_from=args.parent, max_updates_this_invocation=1)
            assert admission['outcome'] == 'stopped' and admission['durable_update'] == 1
            first = json.loads((table_root/'admission/updates.jsonl').read_text().splitlines()[0])
            budget = dict(successful_seconds=first['seconds'], updates=1,
                          stop_requested=False, recent=[first['seconds']])

            def timed_step(*a, **kw):
                row = native_train_step(*a, **kw)
                budget['successful_seconds'] += row['seconds']
                budget['updates'] += 1
                budget['recent'] = (budget['recent']+[row['seconds']])[-10:]
                # Leave approximately one successful step of headroom; avoid
                # a polling delay or saving time charging the update budget.
                margin = max(budget['recent'])*1.1
                if budget['successful_seconds'] >= args.seconds-margin:
                    budget['stop_requested'] = True
                    signal.raise_signal(signal.SIGTERM)
                return row

            runner.train_step = timed_step
            terminal = runner.run(plan, table_root/'main', device=args.device,
                resume=table_root/'admission/checkpoint-progress.pt',
                max_updates_this_invocation=100000)
            runner.train_step = native_train_step
            assert terminal['outcome'] == 'interrupted' and budget['stop_requested'], terminal.get('error')
            assert terminal['durable_update'] == budget['updates']
            atomic_json(table_root/'budget.json', budget)
            checkpoint = table_root/'main/checkpoint-progress.pt'
            test = evaluate_one(args, root, entry, plan, checkpoint, table_root)
            receipt['datasets'][name] = dict(**test,
                successful_update_seconds=budget['successful_seconds'],
                parent_checkpoint_sha256=parent_digest,
                checkpoint=str(checkpoint.resolve()))
            atomic_json(root/'campaign.json', receipt)
            print(json.dumps(dict(dataset=name, **test['metrics'],
                  updates=budget['updates'], successful_seconds=budget['successful_seconds'])), flush=True)
            gc.collect()
            if args.device == 'mps':
                torch.mps.empty_cache()
        scores = sorted(d['metrics']['r2'] for d in receipt['datasets'].values())
        receipt.update(outcome='completed', active_table=None,
                       macro_r2=math.fsum(scores)/12, median_r2=(scores[5]+scores[6])/2)
    except Exception as error:
        receipt.update(outcome='failed', error_type=type(error).__name__, error=str(error))
        atomic_json(root/'campaign.json', receipt)
        raise
    finally:
        runner.train_step = native_train_step
    atomic_json(root/'campaign.json', receipt)
    print(json.dumps(dict(outcome=receipt['outcome'], macro_r2=receipt['macro_r2'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('root','bank-root','parent','parent-sha','parent-manifest','device','host'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--prepare-only', action='store_true')
    run(parser.parse_args())
