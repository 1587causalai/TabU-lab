"""Pair final joint618 metrics with audited support-only XGBoost predictions."""
import collections
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
PREP = Path('/Users/cms/.codex/worktrees/245d/workspace/projects/causal-superintelligence/TabU/preparations')
OLD = PREP / 'tri100-90min-dgx2-20260926/baselines/xgboost-simple'
NEW = PREP / 'joint618-60min-20260926/baselines/xgboost-simple'
MINI = Path('/Users/cms/.codex/worktrees/379d/workspace/projects/causal-superintelligence/TabU/preparations/joint618-60min-20260926')
HOSTS = ('dgx2', 'dustinstudio', 'gongqian-mini')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_new_prediction(record, entry, manifest_dir):
    """Independently rescore saved predictions against the original source truth."""
    path = (manifest_dir / entry['path']).resolve()
    assert sha(path) == record['data_sha256'] == entry['sha256']
    data = read(path)
    target = record['target_column']
    spec = data['features'][target]
    kind = spec['kind']
    visits = collections.defaultdict(list)
    order = spec.get('order') or list(range(len(spec.get('domain') or [])))
    ranks = {label: i / max(len(order) - 1, 1) for i, label in enumerate(order)}
    for m in record['mask_records']:
        support = m['support_rows']
        query = [a[0] for a in m['query_addresses']]
        assert len(set(support)) == 153 and len(set(query)) == 51
        assert set(support).isdisjoint(query)
        assert set(support) | set(query) == set(data['splits']['train'])
        assert all(a[1] == target for a in m['query_addresses'])
        assert len(m['predictions']) == len(m['truths']) == 51
        support_y = [data['values'][row][target] for row in support]
        if kind == 'numeric':
            support_y = list(map(float, support_y))
        digest = hashlib.sha256(json.dumps(support_y, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        assert digest == m['fit_target_sha256']
        if kind != 'numeric':
            assert sorted(set(support_y)) == m['visible_classes']
        for row, pred, truth in zip(query, m['predictions'], m['truths']):
            assert math.isfinite(pred) and truth == data['values'][row][target]
            if kind != 'numeric':
                assert pred in m['visible_classes']
            visits[row].append((pred, truth))
    if kind == 'numeric':
        truth = [items[0][1] for items in visits.values()]
        variance = mean((y - mean(truth)) ** 2 for y in truth)
        metrics = {'query_numeric_r2': 1 - mean(mean((p-t)**2 for p,t in items) for items in visits.values()) / variance}
    else:
        metrics = {'query_discrete_accuracy': mean(mean(p == t for p,t in items) for items in visits.values())}
        if kind == 'ordinal':
            metrics['query_ordinal_rank_mae'] = mean(mean(abs(ranks[p]-ranks[t]) for p,t in items) for items in visits.values())
    assert all(math.isclose(x, record['metrics'][k], rel_tol=1e-12, abs_tol=1e-12) for k,x in metrics.items())


def compare(include_new=False):
    snapshots = {h: read(HERE / (h + '-final-snapshot.json')) for h in HOSTS}
    dgx = snapshots['dgx2']
    bank = {t['table']: t for t in read(PREP / 'joint618-60min-20260926/evidence/qualification/fixed-query-bank.json')['tables']}
    mini_bank = {name: (probe, masks) for name, probe, masks in read(MINI / 'evidence/actual-evaluation-bank-addresses.json')['tables']}
    assert len(bank) == len(mini_bank) == 618
    fields = ('mask_seed', 'code_seed', 'window_seed', 'query_addresses', 'query_rows', 'support_rows')
    for name, t in bank.items():
        p, masks = mini_bank[name]
        assert t['probe'] == p and len(t['masks']) == len(masks)
        for a, b in zip(t['masks'], masks):
            assert a['mask_index'] == b['index']
            assert all(a[k] == b[k] for k in fields), name
    roster = {e['id']: (e['sha256'], e['cohort'], e['target_column']) for e in dgx['manifest_tables']}
    for h, s in snapshots.items():
        assert s['source_hashes'] == dgx['source_hashes']
        assert s['seeds'] == dgx['seeds'] and s['probes'] == dgx['probes']
        assert {e['id']: (e['sha256'], e['cohort'], e['target_column']) for e in s['manifest_tables']} == roster
        assert set(s['tables']) == set(bank)
        for name, row in s['tables'].items():
            masks = bank[name]['masks']
            queries = sorted({q for mask in masks for q in mask['query_rows']})
            assert row['probe'] == bank[name]['probe']
            assert row['query_rows'] == queries and row['unique_query_cells'] == len(queries)
            assert row['query_exposures'] == len(masks) * 51
    old_plan = read(OLD / 'plan.json')
    old_audit = read(OLD / 'prediction-audit.json')
    assert old_audit['status'] == 'passed'
    records = {}
    for table in old_plan['tables']:
        name = table['table']
        p = OLD / 'results' / (name + '.json')
        assert sha(p) == old_audit['original_prediction_receipts_sha256'][name]
        r = read(p)
        assert r['data_sha256'] == roster[name][0]
        assert r['plan_sha256'] == sha(OLD / 'plan.json')
        assert len(bank[name]['masks']) == len(table['masks'])
        for a, b in zip(bank[name]['masks'], table['masks']):
            assert a['mask_index'] == b['index']
            assert all(a[k] == b[k] for k in fields)
        records[name] = r
    assert len(records) == 498
    if include_new:
        new_plan = read(NEW / 'plan.json')
        new_plan_sha = sha(NEW / 'plan.json')
        completion = read(NEW / 'completion.json')
        audit = read(NEW / 'prediction-audit.json')
        assert completion['status'] == 'completed' and completion['tables'] == 120 and completion['masks'] == 240
        assert completion['plan_sha256'] == new_plan_sha
        assert audit['status'] == 'passed' and audit['tables'] == 618 and audit['masks'] == 1238
        for key in ('classifier', 'regressor', 'preprocessing', 'classifier_policy'):
            assert new_plan[key] == old_plan[key]
        manifest_path = PREP / 'joint618-60min-20260926/manifests/joint618.json'
        entries = {e['id']: e for e in read(manifest_path)['tables']}
        for p in sorted((NEW / 'results').glob('new120__*.json')):
            r = read(p)
            name = r['table']
            assert r['plan_sha256'] == new_plan_sha
            assert sha(p) == audit['original_prediction_receipts_sha256'][name]
            assert name not in records and r['data_sha256'] == roster[name][0]
            masks = bank[name]['masks']
            assert len(r['mask_records']) == len(masks) == 2
            for a, b in zip(masks, r['mask_records']):
                assert a['mask_index'] == b['index']
                assert a['query_addresses'] == b['query_addresses'] and a['support_rows'] == b['support_rows']
            audit_new_prediction(r, entries[name], manifest_path.parent)
            records[name] = r
        assert len(records) == 618
    groups = collections.defaultdict(list)
    for name, r in records.items():
        kind = r['target_kind']
        metrics = ['query_numeric_r2'] if kind == 'numeric' else ['query_discrete_accuracy']
        if kind == 'ordinal':
            metrics.append('query_ordinal_rank_mae')
        for metric in metrics:
            row = {'table': name, 'xgboost': r['metrics'][metric]}
            row.update({h: s['tables'][name]['metrics'][metric] for h, s in snapshots.items()})
            groups[r['cohort'] + '/' + kind + '/' + metric].append(row)
    result = {'coverage': len(records), 'alignment': '618 data identities and evaluation source/seeds/probes agree; exact DGX2/Mini masks and all three final Query unions/exposures checked; original prediction file hashes match prior rescore audit', 'checkpoint_metadata': {h: {k: s[k] for k in ['evaluation_path', 'evaluation_file_sha256', 'update', 'checkpoint_sha256']} for h, s in snapshots.items()}, 'groups': {}}
    for key, pairs in sorted(groups.items()):
        stats = {h: {'mean': mean(p[h] for p in pairs), 'median': median(p[h] for p in pairs)} for h in ('xgboost',) + HOSTS}
        for h in HOSTS:
            ds = [(p[h] - p['xgboost']) * (-1 if key.endswith('rank_mae') else 1) for p in pairs]
            stats[h]['win_tie_loss'] = [sum(d > 1e-12 for d in ds), sum(abs(d) <= 1e-12 for d in ds), sum(d < -1e-12 for d in ds)]
        result['groups'][key] = {'n': len(pairs), 'summary': stats, 'by_table': pairs}
    output = HERE / ('comparison-618.json' if include_new else 'comparison-498.json')
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'coverage': len(records), 'groups': {k: {'n': v['n'], 'summary': v['summary']} for k, v in result['groups'].items()}}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-new', action='store_true')
    compare(parser.parse_args().include_new)
