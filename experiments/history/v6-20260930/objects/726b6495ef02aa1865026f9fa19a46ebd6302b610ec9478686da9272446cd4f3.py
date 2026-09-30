"""Frozen V5.5 inference on the exact saved Gen-4 OpenML12 episode bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from tabu_lab.curriculum_v53.artifacts import atomic_json, load_checkpoint, sha256
from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v55 import ColumnSchema, RestorationInput, RestorationRequest


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def make_input(data, trace, name, device, dtype, *, poison=False):
    support, query = trace['context_row_ids'], trace['query_row_ids']
    ids = support + query
    # Truth is not supplied to the model or its preparation/scoring functions.
    rows = [list(data['values'][i]) for i in ids]
    for row in rows[len(support):]:
        row[-1] = 1234567.0 if poison else 0.0
    tensor = torch.tensor(rows, dtype=dtype, device=device)
    width = tensor.shape[1]
    features = data.get('features') or [dict(kind='numeric', domain=[]) for _ in range(width)]
    schema = tuple(ColumnSchema(f'{name}/column-{i}', f['kind'],
                  None if f['kind'] == 'numeric' else len(f['domain']),
                  tuple(f['order']) if f.get('order') is not None else None)
                   for i, f in enumerate(features))
    values = tuple(tensor[:, i] if f.kind == 'numeric' else tensor[:, i].long()
                   for i, f in enumerate(schema))
    visible = torch.ones(tensor.shape, dtype=torch.bool, device=device)
    visible[len(support):, -1] = False
    query_mask = torch.zeros_like(visible)
    query_mask[len(support):, -1] = True
    inputs = RestorationInput(schema, values, visible, query_mask, trace['codebook_seed'])
    request = RestorationRequest(query_mask.nonzero())
    assert bool((inputs.values[-1][len(support):] == 0).all())
    request.validate(inputs)
    return inputs, request


def metrics(rows):
    n = len(rows)
    avg = math.fsum(row['target'] for row in rows) / n
    sse = math.fsum((row['target'] - row['prediction']) ** 2 for row in rows)
    sst = math.fsum((row['target'] - avg) ** 2 for row in rows)
    return dict(n=n, r2=1-sse/sst, rmse=math.sqrt(sse/n),
                mae=math.fsum(abs(row['target']-row['prediction']) for row in rows)/n)


def run(args):
    root = Path(args.root)
    out = root / args.output
    out.mkdir(exist_ok=False)
    report = dict(schema='tabu.openml12.frozen-v55.v1', status='local_unissued',
                  outcome='running', optimizer_updates=0, datasets={},
                  bank_sha256=sha256(root/'bank.json'), evaluator_sha256=sha256(__file__),
                  scope='Frozen historical OpenML12 diagnostic; exposure audit is separate; no clean unseen-table claim.')
    atomic_json(out/'started.json', report)
    try:
        tick = time.monotonic()
        runtime = configure_runtime(args.device)
        plan = load_v55_plan(args.manifest)
        payload, digest = load_checkpoint(args.checkpoint)
        assert digest == args.expected_sha and payload['purpose'] == 'training'
        assert payload['identity'] == plan.identity, 'parent identity/source/data drift'
        model = make_model(plan).to(device=args.device, dtype=execution_dtype(args.device))
        assert model.config.as_dict() == payload['model_config'], 'resolved model config drift'
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        model.requires_grad_(False)
        before = state_hash(model)
        report.update(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=digest,
                      checkpoint_update=payload['state']['update'], runtime=runtime,
                      model_config=payload['model_config'], identity=plan.identity,
                      initial_state_sha256=before, lineage=payload['lineage'])
        atomic_json(out/'resolved.json', report)
        del payload
        bank = json.loads((root/'bank.json').read_text())
        total_predictions = 0
        for entry in bank['tables']:
            name = entry['name']
            path = root/entry['path']
            assert sha256(path) == entry['sha256']
            data = json.loads(path.read_text())
            train, test = set(data['splits']['train']), set(data['splits']['test'])
            assert not train & test
            assert train | test == set(range(len(data['values'])))
            rows, seen = [], set()
            table_tick = time.monotonic()
            for ep_index, trace in enumerate(entry['traces']):
                support, query = trace['context_row_ids'], trace['query_row_ids']
                assert len(support) == len(set(support)) and set(support) <= train
                assert len(query) == len(set(query)) and set(query) <= test
                assert not seen & set(query)
                inputs, request = make_input(data, trace, name, args.device, execution_dtype(args.device))
                if ep_index == 0:
                    twin, twin_request = make_input(data, trace, name, args.device,
                                                    execution_dtype(args.device), poison=True)
                    assert torch.equal(request.targets, twin_request.targets)
                    for a, b in zip(inputs.values, twin.values):
                        assert torch.equal(a, b), 'Query truth leaked into forward input'
                    assert torch.equal(inputs.visible, twin.visible)
                    assert torch.equal(inputs.query, twin.query)
                    assert inputs.schema == twin.schema and inputs.code_seed == twin.code_seed
                    del twin, twin_request
                with torch.inference_mode():
                    result = model(inputs, request)
                assert len(result.columns) == 1
                prediction = result.columns[0]
                assert prediction.column == entry['width']-1 and prediction.result.status == 'ok'
                values = prediction.decoded.detach().cpu().tolist()
                positions = prediction.target_indices.detach().cpu().tolist()
                targets = request.targets.detach().cpu().tolist()
                assert len(values) == len(query)
                for value, pos in zip(values, positions):
                    row, col = targets[pos]
                    assert col == entry['width']-1 and math.isfinite(value)
                    row_id = query[row-len(support)]
                    assert row_id not in seen
                    seen.add(row_id)
                    rows.append(dict(row_id=row_id, episode_id=trace['episode_id'],
                                     target=float(data['values'][row_id][-1]), prediction=value))
                del result, prediction, inputs, request
                if args.device == 'mps':
                    torch.mps.synchronize()
                elif args.device == 'cuda:0':
                    torch.cuda.synchronize()
            assert seen == test and len(rows) == entry['test_n']
            rows.sort(key=lambda x: x['row_id'])
            score = metrics(rows)
            atomic_json(out/f'{name}-predictions.json', rows)
            report['datasets'][name] = dict(metrics=score, data_sha256=entry['sha256'],
                trace_sha256=entry['trace_sha256'], train_n=entry['train_n'],
                context_n=len(entry['traces'][0]['context_row_ids']),
                query_chunk=max(len(t['query_row_ids']) for t in entry['traces']),
                episodes=len(entry['traces']), query_label_invariance=True,
                seconds=time.monotonic()-table_tick)
            total_predictions += len(rows)
            atomic_json(out/'progress.json', report)
            print(json.dumps(dict(dataset=name, **score, seconds=time.monotonic()-table_tick)), flush=True)
        assert total_predictions == 7894 and len(report['datasets']) == 12
        assert state_hash(model) == before, 'model mutated'
        assert sha256(args.checkpoint) == digest, 'checkpoint mutated'
        r2 = sorted(d['metrics']['r2'] for d in report['datasets'].values())
        report.update(outcome='completed', model_state_unchanged=True, checkpoint_unchanged=True,
                      total_predictions=total_predictions, macro_r2=math.fsum(r2)/len(r2),
                      median_r2=(r2[5]+r2[6])/2, seconds=time.monotonic()-tick)
        if args.device == 'cuda:0':
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
        elif args.device == 'mps':
            report['final_driver_allocated_gib'] = torch.mps.driver_allocated_memory()/2**30
    except Exception as error:
        report.update(outcome='failed', error_type=type(error).__name__, error=str(error))
        atomic_json(out/'terminal.json', report)
        raise
    atomic_json(out/'terminal.json', report)
    print(json.dumps({k: report[k] for k in ('outcome','macro_r2','median_r2','seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root','manifest','checkpoint','expected-sha','device'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--output', default='evaluation-01')
    run(parser.parse_args())
