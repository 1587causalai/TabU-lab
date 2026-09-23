#!/usr/bin/env python3
"""Read-only old120 SSH -> W&B mirror for fresh runs and saved-state extensions.

The local SQLite outbox is authoritative for mirror delivery, never for fit.
No tensors, data rows, code, environment values, or checkpoint files are sent.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v54-old120-wandb-20260922"))
from cycle_mirror import CycleMixin, replay_enabled
from cycle_loss import initial_replay_state, advance_replay, REPLAY_GROUPS, WEIGHTED_REPLAY_KIND
from terminal_lifecycle import finish_telemetry

REMOTE = r'''
import hashlib,json,re,sys
from pathlib import Path
q=json.loads(sys.stdin.readline()); p=Path(q['output'])
z=json.loads((p/'resolved.json').read_text())
rows=[]; offset=q['offset']; f=p/'updates.jsonl'
if f.exists():
 if f.stat().st_size < offset: raise ValueError('updates journal shrank')
 with f.open('rb') as h:
  h.seek(offset)
  for _ in range(20000):
   line=h.readline()
   if not line or not line.endswith(b'\n'): break
   u=json.loads(line)
   row={k:u[k] for k in ('update','table','loss','gradient_norm','seconds','total_seconds')}
   row.update({k:u[k] for k in ('update_kind','normal_update','extra_updates') if k in u})
   rows.append(row)
   offset=h.tell()
latest=rows[-1]['update'] if rows else q['last_update']
evaluations=[]
for f in ([] if q.get('cycles_only') else sorted(p.glob('evaluation-*.json'))):
 if f.name in q['seen_evaluations']: continue
 e=json.loads(f.read_text())
 if e['update']>latest: continue
 b=e['probes']['train_fit']
 evaluations.append({'file':f.name,'update':e['update'],'complete':b['complete'],
  'origin':'current_output',
  'tables':[{'table':r['table'],'metrics':{k:v for k,v in r['metrics'].items() if k.startswith('query_')},
   'query_exposures':r['query_exposures'],'unique_query_cells':r['unique_query_cells']} for r in b['by_table']]})
f=p/'terminal.json'; terminal=None; runtime=None
if f.exists() and not q.get('cycles_only'):
 t=json.loads(f.read_text())
 runtime=t.get('runtime')
 if t['update']<=latest:
  terminal={k:t.get(k) for k in ('outcome','update','durable_update','checkpoint_sha256','total_seconds','error_type','normal_update','extra_updates')}
  # Error messages are local evidence, never uploaded verbatim. Some failures
  # (notably BudgetExhausted) have only error, without error_type.
  terminal['error_present']=t.get('error') not in (None,'')
  terminal['terminal_identity_matches']=t.get('identity',{}).get('sha256')==z['identity']['sha256']
  terminal['checkpoint_verified']=False
  terminal['checkpoint_verification']='not_checked'
  if t.get('outcome') in ('stopped','interrupted'):
   try:
    expected=t.get('checkpoint_sha256'); name=t.get('checkpoint')
    if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
     raise ValueError('invalid checkpoint digest')
    if not isinstance(name,str) or not name or Path(name).is_absolute():
     raise ValueError('invalid checkpoint pointer')
    pointer=p/name; checkpoint=pointer.resolve(strict=True)
    if not checkpoint.is_relative_to(p.resolve()) or not checkpoint.is_file():
     raise ValueError('checkpoint outside output or not regular file')
    before=checkpoint.stat()
    if before.st_size<=0: raise ValueError('empty checkpoint')
    digest=hashlib.sha256()
    with checkpoint.open('rb') as handle:
     for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    after=checkpoint.stat()
    stable=(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    named_ok=not re.fullmatch('[0-9a-f]{64}',checkpoint.stem) or checkpoint.stem==expected
    terminal['checkpoint_verified']=stable and pointer.resolve(strict=True)==checkpoint and named_ok and digest.hexdigest()==expected
    terminal['checkpoint_verification']='sha256_match' if terminal['checkpoint_verified'] else 'checksum_or_stability_mismatch'
   except (OSError,ValueError,RuntimeError):
    terminal['checkpoint_verification']='missing_or_invalid_checkpoint'
print(json.dumps({'identity_sha256':z['identity']['sha256'],'source_sha256':z['identity']['source']['sha256'],
 'declared_model':z['spec']['model'],'terminal_runtime':runtime,
 'tables':sorted(z['identity']['datasets']),'offset':offset,'updates':rows,'evaluations':evaluations,'terminal':terminal}))
'''


def safe_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def replay_base_counts(config):
    return config.get('base_extra_counts', {table: 0 for table in config['base_counts']})


def validate_replay_config(config):
    if not replay_enabled(config):
        return
    base = config['initial_update']
    normal = config.get('initial_normal_update')
    policy = config['loss_replay']
    limit, start = policy.get('normal_max_updates'), policy.get('start_normal_cursor')
    if (type(normal) is not int or normal < 0 or normal > base or normal % 120
            or type(limit) is not int or limit <= normal or limit % 120
            or type(start) is not int or start != normal):
        raise ValueError('invalid replay initial normal cursor or normal budget')
    extras = replay_base_counts(config)
    if (not isinstance(extras, dict) or set(extras) != set(config['base_counts'])
            or any(type(v) is not int or v < 0 or v > config['base_counts'][k] for k, v in extras.items())
            or sum(extras.values()) != base - normal):
        raise ValueError('base extra exposure disagrees with actual and normal cursors')
    initial_extra = base - normal
    if policy['kind'] == WEIGHTED_REPLAY_KIND and 'start_extra_updates' not in policy:
        raise ValueError('weighted replay must explicitly declare inherited start_extra_updates')
    for value in (policy.get('start_extra_updates', initial_extra), config.get('initial_extra_updates', initial_extra)):
        if type(value) is not int or value != initial_extra:
            raise ValueError('inherited extra counter differs from initial actual minus normal')
    extra_per_cycle = sum(count * passes for _, count, passes in REPLAY_GROUPS[policy['kind']])
    actual_new = (limit - normal) // 120 * (120 + extra_per_cycle)
    if config.get('new_max_updates', actual_new) != actual_new:
        raise ValueError('actual new budget disagrees with declared normal/extra schedule')
    if config.get('global_max_updates', base + actual_new) != base + actual_new:
        raise ValueError('actual global budget disagrees with replay schedule')


class Outbox(CycleMixin):
    def __init__(self, path, config):
        validate_replay_config(config)
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, event_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS exposure (update_id INTEGER PRIMARY KEY, counts TEXT NOT NULL, seconds REAL NOT NULL)')
        marker = {k: config[k] for k in ('run_id', 'identity_sha256', 'remote_output', 'initial_update', 'base_counts', 'base_total_seconds', 'evaluate_every')}
        if replay_enabled(config):
            marker.update(loss_replay=config['loss_replay'],
                          initial_normal_update=config['initial_normal_update'],
                          base_extra_counts=replay_base_counts(config))
            self.db.execute('CREATE TABLE IF NOT EXISTS replay_exposure (update_id INTEGER PRIMARY KEY, extra_counts TEXT NOT NULL, normal_update INTEGER NOT NULL)')
        previous = self.get('identity')
        if previous is not None and previous != marker:
            raise ValueError('mirror identity differs from existing cursor')
        with self.db:
            self.put('identity', marker)

    def get(self, key, default=None):
        row = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, json.dumps(value)))

    def emit(self, event_id, payload):
        if self.db.execute('SELECT 1 FROM events WHERE event_id=?', (event_id,)).fetchone():
            return
        seq = self.db.execute('SELECT COALESCE(MAX(seq),-1)+1 FROM events').fetchone()[0]
        self.db.execute('INSERT INTO events VALUES (?,?,?)', (seq, event_id, json.dumps(payload)))

    def request(self, config):
        return {'output': config['remote_output'], 'offset': self.get('offset', 0),
                'last_update': self.get('last_update', config['initial_update']),
                'seen_evaluations': self.get('seen_evaluations', [])}

    def ingest(self, snapshot, config, references):
        if snapshot['identity_sha256'] != config['identity_sha256']:
            raise ValueError('remote run identity mismatch')
        if snapshot['source_sha256'] != config['source_sha256']:
            raise ValueError('remote source identity mismatch')
        tables = snapshot['tables']
        if len(tables) != 120 or set(tables) != set(references):
            raise ValueError('fixed-query reference table set mismatch')
        base = config['initial_update']
        base_counts = config['base_counts']
        if set(base_counts) != set(tables) or sum(base_counts.values()) != base:
            raise ValueError('base old120 exposure does not match the initial global update')
        counts = self.get('counts', base_counts.copy())
        last = self.get('last_update', base)
        replay = replay_enabled(config)
        if replay:
            base_extras = replay_base_counts(config)
            extra_counts = self.get('extra_counts', base_extras.copy())
            normal_base = config['initial_normal_update']
            normal = self.get('normal_update', normal_base)
            state = self.get('raw_replay_state')
            if state is None:
                state = initial_replay_state(tables, base, normal_base, policy_kind=config['loss_replay']['kind'])
            # Validate the entire all-step stream before committing any scalar
            # events. The independent cycle cursor also supports late backfill.
            replay_state, _ = advance_replay(state, snapshot['updates'])
        seen = set(self.get('seen_evaluations', []))
        best = self.get('best', {})
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO exposure VALUES (?,?,?)',
                            (base, json.dumps(base_counts), config['base_total_seconds']))
            if replay:
                self.db.execute('INSERT OR IGNORE INTO replay_exposure VALUES (?,?,?)',
                                (base, json.dumps(base_extras), normal_base))
            for row in snapshot['updates']:
                update = row['update']
                if update != last + 1:
                    raise ValueError('non-contiguous new-experiment updates')
                last = update
                counts[row['table']] += 1
                kind = row.get('update_kind', 'normal')
                if replay:
                    normal = row['normal_update']
                    if kind != 'normal':
                        extra_counts[row['table']] += 1
                    if 'extra_updates' in row and (type(row['extra_updates']) is not int
                                                   or row['extra_updates'] != update - normal):
                        raise ValueError('journal extra counter disagrees with actual minus normal updates')
                if update % config['evaluate_every'] == 0:
                    self.db.execute('INSERT OR IGNORE INTO exposure VALUES (?,?,?)',
                                    (update, json.dumps(counts), row['total_seconds']))
                    if replay:
                        self.db.execute('INSERT OR IGNORE INTO replay_exposure VALUES (?,?,?)',
                                        (update, json.dumps(extra_counts), normal))
                new_update = update - base
                if new_update % 100 == 0:
                    payload = {'train/update': new_update, 'train/new_update': new_update,
                               'train/global_update': update, 'progress/new_updates': new_update,
                               'wall/new_seconds': max(0., row['total_seconds'] - config['base_total_seconds']),
                               'wall/old120_seconds': row['total_seconds'],
                               'train/table': row['table'], 'train/table_new_exposure': counts[row['table']] - base_counts[row['table']],
                               'train/table_cumulative_exposure': counts[row['table']],
                               'train/update_kind': kind}
                    if replay:
                        payload.update({'train/normal_update': normal,
                                        'train/new_normal_update': normal - normal_base,
                                        'train/extra_updates': update - normal,
                                        'train/new_extra_updates': new_update - (normal - normal_base),
                                        'progress/new_normal_updates': normal - normal_base,
                                        'train/table_extra_exposure': extra_counts[row['table']]})
                    for key in ('loss', 'gradient_norm', 'seconds'):
                        if safe_number(row[key]):
                            payload['train/' + key] = row[key]
                    payload['train/finite'] = int(all(safe_number(row[k]) for k in ('loss', 'gradient_norm', 'seconds')))
                    self.emit(f'train:{update}', payload)
                if replay and kind != 'normal':
                    self.emit(f'replay:{update}', {
                        'replay/update': new_update, 'replay/global_update': update,
                        'replay/normal_update': normal, 'replay/new_normal_update': normal - normal_base,
                        'replay/loss': row['loss'], 'replay/group': kind, 'replay/table': row['table'],
                        'replay/extra_updates': update - normal,
                        'replay/new_extra_updates': new_update - (normal - normal_base),
                        'replay/table_extra_exposure': extra_counts[row['table']],
                        'replay/table_new_extra_exposure': extra_counts[row['table']] - base_extras[row['table']]})
            # Needed for a terminal evaluation away from a periodic boundary.
            if snapshot['terminal']:
                self.db.execute('INSERT OR IGNORE INTO exposure VALUES (?,?,?)',
                                (last, json.dumps(counts), snapshot['terminal']['total_seconds']))
                if replay:
                    if snapshot['terminal']['update'] != last:
                        raise ValueError('terminal and replay journal cursor disagree')
                    self.db.execute('INSERT OR IGNORE INTO replay_exposure VALUES (?,?,?)',
                                    (last, json.dumps(extra_counts), normal))
            for evaluation in snapshot['evaluations']:
                update = evaluation['update']
                if update < base:
                    raise ValueError('completed-parent historical evaluation must not enter this continuation')
                if not evaluation['complete'] or len(evaluation['tables']) != 120:
                    raise ValueError('incomplete fixed Query evaluation')
                exposure = self.db.execute('SELECT counts,seconds FROM exposure WHERE update_id=?', (update,)).fetchone()
                if not exposure:
                    raise ValueError('no exact exposure checkpoint for evaluation')
                ec = json.loads(exposure[0])
                payload = {'fixed/update': update - base, 'fixed/new_update': update - base,
                           'fixed/global_update': update,
                           'fixed/pre_evaluation_seconds': max(0., exposure[1] - config['base_total_seconds']),
                           'fixed/complete': 1, 'fixed/table_count': 120,
                           'fixed/origin': evaluation.get('origin', 'current_output')}
                if replay:
                    ex = self.db.execute('SELECT extra_counts,normal_update FROM replay_exposure WHERE update_id=?', (update,)).fetchone()
                    if ex is None:
                        raise ValueError('no exact normal/extra exposure checkpoint for evaluation')
                    extra_ec, fixed_normal = json.loads(ex[0]), ex[1]
                    payload.update({'fixed/normal_update': fixed_normal,
                                    'fixed/new_normal_update': fixed_normal - normal_base,
                                    'fixed/extra_updates': update - fixed_normal,
                                    'fixed/new_extra_updates': (update - base) - (fixed_normal - normal_base)})
                if {r['table'] for r in evaluation['tables']} != set(tables):
                    raise ValueError('fixed Query evaluation table set mismatch')
                for row in evaluation['tables']:
                    name = row['table']
                    prefix = f'fixed/{name}/'
                    for src, dst in (('query_numeric_r2', 'r2'), ('query_numeric_normalized_mse', 'nmse'), ('query_discrete_accuracy', 'accuracy')):
                        value = row['metrics'].get(src)
                        if safe_number(value):
                            payload[prefix + dst] = value
                            best_key = name + '/' + dst
                            previous = best.get(best_key, value)
                            best[best_key] = min(previous, value) if dst == 'nmse' else max(previous, value)
                            payload[prefix + 'best_' + dst] = best[best_key]
                    payload[prefix + 'new_exposure'] = ec[name] - base_counts[name]
                    payload[prefix + 'cumulative_exposure'] = ec[name]
                    payload[prefix + 'base_exposure'] = base_counts[name]
                    if replay:
                        payload[prefix + 'cumulative_extra_exposure'] = extra_ec[name]
                        payload[prefix + 'new_extra_exposure'] = extra_ec[name] - base_extras[name]
                        payload[prefix + 'cumulative_normal_exposure'] = ec[name] - extra_ec[name]
                        payload[prefix + 'new_normal_exposure'] = ec[name] - extra_ec[name] - (base_counts[name] - base_extras[name])
                    payload[prefix + 'query_exposures'] = row['query_exposures']
                    payload[prefix + 'query_unique_cells'] = row['unique_query_cells']
                    unique_targets = row['metrics'].get('query_unique_targets')
                    if safe_number(unique_targets):
                        payload[prefix + 'query_unique_targets'] = unique_targets
                    for key in ('r2', 'nmse', 'accuracy'):
                        value = references[name].get(key)
                        if safe_number(value):
                            payload[prefix + 'reference_' + key] = value
                # A fixed bank at one global update is one curve point, even
                # if the runner has duplicate initial/final receipt names.
                self.emit('evaluation:train_fit:' + str(update), payload)
                seen.add(evaluation['file'])
            terminal = snapshot['terminal']
            if terminal:
                if replay:
                    for key, expected in (('normal_update', normal), ('extra_updates', last - normal)):
                        if terminal.get(key) is not None and terminal[key] != expected:
                            raise ValueError('terminal normal/extra counter disagrees with journal')
                self.emit('terminal', {'terminal/' + k: v for k, v in terminal.items() if v is not None}
                          | {'terminal/global_update': terminal['update'], 'terminal/new_update': terminal['update'] - base}
                          | ({'terminal/normal_update': normal, 'terminal/new_normal_update': normal - normal_base,
                              'terminal/extra_updates': last - normal,
                              'terminal/new_extra_updates': (last - base) - (normal - normal_base)} if replay else {}))
                self.put('terminal', terminal)
            self.put('offset', snapshot['offset'])
            self.put('last_update', last)
            self.put('counts', counts)
            if replay:
                self.put('extra_counts', extra_counts)
                self.put('normal_update', normal)
                self.put('raw_replay_state', replay_state)
            self.put('seen_evaluations', sorted(seen))
            self.put('best', best)
            self.put('remote_declared_model', snapshot.get('declared_model'))
            self.put('remote_terminal_runtime', snapshot.get('terminal_runtime'))


def read_remote(config, request):
    command = 'python3 -c ' + shlex.quote(REMOTE)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o',
                             f"ConnectTimeout={config.get('ssh_connect_timeout', 10)}",
                             config['host'], command],
                            input=json.dumps(request) + '\n', text=True,
                            capture_output=True, timeout=config.get('ssh_read_timeout', 60))
    if result.returncode:
        raise RuntimeError('remote read failed')  # do not echo arbitrary SSH output
    return json.loads(result.stdout)


def connect(config, panel):
    import wandb  # monitoring environment only; no torch or trainer import
    allowed = ('identity_sha256', 'source_sha256', 'parent_checkpoint_sha256', 'parent_update',
               'parent_model', 'model', 'model_label', 'initialization', 'heads', 'backbone_layers', 'unit_layers',
               'initial_update', 'initial_normal_update', 'initial_extra_updates', 'base_total_seconds', 'loss_replay',
               'function_preserving_expected', 'function_preserving_verified', 'function_preserving_verification',
               'new_max_updates', 'global_max_updates', 'updates_per_table', 'table_count',
               'training_started_utc', 'monitor_attached_utc',
               'episode_policy', 'codec_version', 'device', 'dtype', 'evaluate_every')
    metadata = {k: config[k] for k in allowed if k in config}
    metadata.update(claim_boundary='training-row fixed Query fit; no reserved or final_test',
                    mirror='read-only SSH scalar observer; local receipts authoritative',
                    training_sample_every=100, model_step_semantics='global old120 update; new_update subtracts initial_update',
                    default_chart_axis='new updates within the declared run or continuation',
                    best_includes_initial_boundary_evaluation=True,
                    cumulative_exposure_scope=config.get('cumulative_exposure_scope',
                                                         'old120 only; excludes prior joint5 exposure'),
                    base_exposure_min=min(config['base_counts'].values()),
                    base_exposure_max=max(config['base_counts'].values()))
    if replay_enabled(config):
        policy = config['loss_replay']
        normal_new = policy['normal_max_updates'] - config['initial_normal_update']
        weighted = policy['kind'] == WEIGHTED_REPLAY_KIND
        strategy = ('120 balanced normal updates, then frozen original-loss top2 for3 complete passes, '
                    'top6 for2 complete passes, top24 once; nested groups are additive, highest2 tables receive6 extras'
                    if weighted else '120 balanced normal updates, then original-loss top6 once and top24 once; top6 overlap is intentional')
        extra_per_cycle = sum(count * passes for _, count, passes in REPLAY_GROUPS[policy['kind']])
        metadata.update(loss_replay_strategy=strategy, extra_updates_per_cycle=extra_per_cycle,
                        actual_updates_per_cycle=120 + extra_per_cycle,
                        initial_extra_updates=config['initial_update'] - config['initial_normal_update'],
                        new_extra_max_updates=normal_new // 120 * extra_per_cycle,
                        best_includes_initial_boundary_evaluation=config.get('best_includes_initial_boundary_evaluation', False),
                        best_scope='current output fixed evaluations only; no inherited parent metrics',
                        normal_max_updates=policy['normal_max_updates'],
                        new_normal_max_updates=normal_new,
                        new_actual_max_updates=config['new_max_updates'],
                        actual_global_max_updates=config['initial_update'] + config['new_max_updates'],
                        train_cycle_axis='normal updates since initial_normal_update; extra updates excluded',
                        train_and_replay_axis='actual optimizer updates since initial_update',
                        fixed_axis='actual optimizer updates since initial_update; evaluation cadence unchanged',
                        base_extra_exposure_sum=sum(replay_base_counts(config).values()))
    run = wandb.init(entity=config.get('entity', 'zj3712'), project=config.get('project', 'restoration'),
                     id=config['run_id'], name=config.get('name', config['run_id']),
                     group=config.get('group', 'v54-old120-existing-runs-20260922'), resume='allow',
                     mode='online', config=metadata, dir=str(panel),
                     settings=wandb.Settings(disable_git=True, disable_code=True, console='off', silent=True,
                                             x_disable_stats=True, x_disable_meta=True, init_timeout=45))
    run.define_metric('train_cycle/update')
    run.define_metric('train_cycle/*', step_metric='train_cycle/update')
    run.define_metric('train/update')
    run.define_metric('fixed/update')
    run.define_metric('train/*', step_metric='train/update')
    run.define_metric('fixed/*', step_metric='fixed/update')
    run.define_metric('progress/new_updates')
    run.define_metric('wall/*', step_metric='progress/new_updates')
    cycle_definition = 'equal-table mean of all120 original per-update losses in one anchored complete balanced cycle'
    if replay_enabled(config):
        run.define_metric('replay/update')
        run.define_metric('replay/*', step_metric='replay/update')
        cycle_definition = ('equal-table mean of 120 original NORMAL update losses per balanced cycle; '
                            'all replay extra losses excluded; not a fixed-checkpoint evaluation')
    distributions = ['train_cycle/median_loss', 'train_cycle/p05_loss', 'train_cycle/p95_loss']
    if config.get('loss_replay', {}).get('kind') == WEIGHTED_REPLAY_KIND:
        distributions.append('train_cycle/p99_loss')
    run.config.update({'cycle_loss_definition': cycle_definition, 'cycle_size': 120, 'cycle_loss_metric': 'train_cycle/mean_loss', 'raw_loss_metric': 'train/loss', 'raw_loss_chart_sampling': 100, 'all_raw_steps_preserved_in_training_journal': True, 'cycle_distribution_metrics': distributions, 'cycle_quantile_definition': 'sorted120 per-table normal raw losses; linear interpolation at (n-1)*q; median q=0.5'}, allow_val_change=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent, help='local experiment panel')
    parser.add_argument('--config', type=Path, required=True, help='verified explicit experiment and lineage metadata')
    parser.add_argument('--references', type=Path, help='default ROOT/naive-query-reference.json')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fixture', type=Path, help='local synthetic snapshot, requires --dry-run')
    parser.add_argument('--once', action='store_true', help='one dry-run poll only')
    args = parser.parse_args()
    if (args.fixture or args.once) and not args.dry_run:
        parser.error('--fixture/--once require --dry-run')
    panel = args.root.resolve()
    config = json.loads(args.config.read_text())
    if 'initial_update' not in config and 'base_update' in config:
        config['initial_update'] = config['base_update']
    required = ('run_id', 'entity', 'project', 'host', 'remote_output', 'identity_sha256', 'source_sha256',
                'initial_update', 'base_counts', 'base_total_seconds', 'evaluate_every',
                'heads', 'backbone_layers', 'unit_layers', 'device', 'dtype', 'model_label',
                'initialization', 'new_max_updates')
    for key in required:
        if key not in config:
            raise ValueError('required monitoring config field missing: ' + key)
    if type(config['initial_update']) is not int or config['initial_update'] < 0:
        raise ValueError('initial_update must be a non-negative integer')
    if type(config['evaluate_every']) is not int or config['evaluate_every'] <= 0:
        raise ValueError('invalid evaluation interval')
    if not safe_number(config['base_total_seconds']) or config['base_total_seconds'] < 0:
        raise ValueError('invalid base cumulative wall time')
    if not isinstance(config['base_counts'], dict) or len(config['base_counts']) != 120:
        raise ValueError('base_counts must declare the exact 120-table scope')
    if any(type(v) is not int or v < 0 for v in config['base_counts'].values()):
        raise ValueError('invalid base per-table exposure')
    if sum(config['base_counts'].values()) != config['initial_update']:
        raise ValueError('base exposure sum differs from old120 update counter')
    references_path = args.references or panel / 'naive-query-reference.json'
    refs = {r['table']: r['metrics'] for r in json.loads(references_path.read_text())['records']}
    prefix = 'wandb-dry-run' if args.dry_run else 'wandb-mirror'
    lock = (panel / (prefix + '.lock')).open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    box = Outbox(panel / (prefix + '.sqlite3'), config)
    if box.get('finished'):
        print(json.dumps({'mirror_already_finished': True, 'terminal': box.get('terminal')}))
        return
    run = None
    next_step = None
    while True:
        phase = 'remote_read'
        try:
            snapshot = json.loads(args.fixture.read_text()) if args.fixture else read_remote(config, box.request(config))
            phase = 'ingest_identity_and_metrics'
            box.ingest(snapshot, config, refs)
            phase = 'cycle_read'
            cycle_snapshot = json.loads(args.fixture.read_text()) if args.fixture else read_remote(config, box.cycle_request(config))
            phase = 'cycle_ingest'
            box.ingest_cycles(cycle_snapshot, config, refs)
            if not args.dry_run:
                if run is None:
                    phase = 'wandb_connect'
                    run = connect(config, panel)
                    # W&B resume's next internal step reconciles the durable outbox
                    # after a crash between SDK enqueue, upload, and cursor writes.
                    next_step = int(run.step)
                    run.summary['training_terminal_confirmed'] = bool(box.get('terminal'))
                    (panel / 'wandb-launch.json').write_text(json.dumps({'run_id': run.id, 'url': run.url,
                        'pid': os.getpid(), 'resumed_next_step': next_step, 'identity_sha256': config['identity_sha256']}, indent=2) + '\n')
                phase = 'wandb_enqueue'
                for seq, payload in box.db.execute('SELECT seq,payload FROM events WHERE seq>=? ORDER BY seq', (next_step,)):
                    data = json.loads(payload)
                    run.log(data, step=seq, commit=True)
                    next_step = seq + 1
                    with box.db:
                        box.put('last_enqueued_step', seq)
                    if 'terminal/outcome' in data:
                        run.summary.update({'training_terminal_confirmed': True, **data})
                if box.cycles_caught_up_to_terminal(config):
                    phase = 'wandb_finish'
                    closure = finish_telemetry(run, box.get('terminal'), config)
                    with box.db:
                        box.put('finished', True)
                        box.put('telemetry_closure', closure)
                    (panel / (prefix + '-status.json')).write_text(json.dumps({'utc_unix': time.time(),
                        'last_global_update': box.get('last_update'), 'last_new_update': box.get('last_update') - config['initial_update'], 'terminal': box.get('terminal'),
                        'mirror_finished': True, 'sdk_finish_completed': True,
                        'telemetry_closure': closure}, indent=2) + '\n')
                    box.db.close()
                    lock.close()
                    return
            status = {'utc_unix': time.time(), 'dry_run': args.dry_run, 'last_global_update': box.get('last_update'), 'last_new_update': box.get('last_update') - config['initial_update'],
                      'event_count': box.db.execute('SELECT COUNT(*) FROM events').fetchone()[0],
                      'next_internal_step': next_step, 'terminal': box.get('terminal')}
            status.update(box.cycle_status(config))
            (panel / (prefix + '-status.json')).write_text(json.dumps(status, indent=2) + '\n')
            if args.once:
                print(json.dumps(status))
                return
        except Exception as error:
            # Deliberately do not print exception content (SDK errors can include secrets).
            retryable = (phase in ('remote_read', 'cycle_read')
                         and isinstance(error, subprocess.TimeoutExpired))
            (panel / (prefix + '-error.json')).write_text(json.dumps({'utc_unix': time.time(), 'phase': phase,
                                                                      'error_type': type(error).__name__,
                                                                      'retryable': retryable}) + '\n')
            if args.dry_run:
                raise
            if retryable:
                # A slow/contended host is not a failed training run. Both SSH reads
                # are read-only; the durable offsets and outbox let the next poll
                # retry without losing or duplicating a training update.
                time.sleep(max(5, args.poll_seconds))
                continue
            # A live trainer is never signalled, restarted, or made to wait.
            if run is not None or phase in ('ingest_identity_and_metrics', 'cycle_ingest'):
                raise RuntimeError('mirror delivery failed; inspect local outbox and relaunch this mirror only') from None
        time.sleep(max(5, args.poll_seconds))


if __name__ == '__main__':
    main()
