"""Read local monitor receipts and atomically render the adaptation result report.

No SSH, training, configuration changes, or notification-state changes occur here.
Only RESULT.md and summary.json are generated beside this file.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
HOSTS = ('dustinstudio', 'dgx2', 'gongqian-mini')
METRICS = ('r2', 'slog')
STAGES = ('parent', 'before', 'half1', 'half2')
STAGE_NAMES = {'parent': '原父点', 'before': '本轮起点', 'half1': '+30分钟', 'half2': '+60分钟', 'final': '+60分钟后'}
FILES = {
    'openml12': {'parent': 'original-parent-eval.json', 'before': 'before-eval.json',
                 'half1': 'half1-eval.json', 'half2': 'half2-eval.json'},
    'sparse100': {'parent': 'original-parent-sparse-eval.json', 'before': 'before-sparse-eval.json',
                  'final': 'sparse-final-eval.json'},
}


def read(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def atomic_text(path, text):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_evaluation(ev, family, train, expected_bank, baseline=None):
    count, tables = (7894, 12) if family == 'openml12' else (54400, 100)
    require(ev.get('outcome') == 'completed', 'evaluation is not completed')
    require(train is not None and train.get('outcome') == 'training_completed',
            'matching completed training receipt is missing')
    require(ev.get('checkpoint_sha256') == train.get('checkpoint_sha256'), 'checkpoint SHA mismatch')
    require(ev.get('optimizer_updates') == 0, 'evaluation optimizer_updates is not zero')
    require(ev.get('evaluation_supervision', ev.get('supervision')) == 'target_only',
            'evaluation readout is not target_only')
    require(ev.get('total_predictions') == count, 'total prediction count mismatch')
    require(len(ev.get('tables', {})) == tables, 'table count mismatch')
    require(bool(expected_bank) and ev.get('bank_sha256') == expected_bank, 'fixed bank SHA mismatch')
    require(sum(t['n'] for t in ev['tables'].values()) == count, 'per-table predictions do not sum to total')
    if baseline is not None:
        require(set(ev['tables']) == set(baseline['tables']), 'table identities changed')
    for name, row in ev['tables'].items():
        require(all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in METRICS),
                f'{name}: nonfinite or undefined reported metric')
        if family == 'openml12':
            require(row['query_rows'] == row['n'], f'{name}: Query count mismatch')
            require(row['forward_rows'] == row['support_rows'] + row['query_rows'],
                    f'{name}: complete context row count mismatch')
            if baseline is not None:
                for key in ('n', 'query_rows', 'support_rows', 'forward_rows'):
                    require(row[key] == baseline['tables'][name][key], f'{name}: context changed')
        else:
            require(row['n'] == 544 and row['support_rows_per_forward'] == 136
                    and row['query_rows_per_forward'] == 68, f'{name}: sparse window/count mismatch')
    for metric in METRICS:
        computed = statistics.fmean(t[metric] for t in ev['tables'].values())
        require(math.isclose(computed, ev['macro'][metric], abs_tol=1e-10, rel_tol=1e-10),
                f'macro {metric} does not match equal-table mean')


def compact_eval(ev):
    return {k: ev[k] for k in ('checkpoint_sha256', 'checkpoint_update', 'bank_sha256',
                                'total_predictions', 'optimizer_updates', 'macro', 'tables')}


def compare(current, reference):
    result = {'macro_delta': {k: current['macro'][k] - reference['macro'][k] for k in METRICS}}
    result['tables'] = {name: {k: current['tables'][name][k] - row[k] for k in METRICS}
                        for name, row in reference['tables'].items()}
    result['counts'] = {}
    result['largest_changes'] = {}
    for metric in METRICS:
        changes = [(name, row[metric]) for name, row in result['tables'].items()]
        result['counts'][metric] = dict(improved=sum(d > 1e-12 for _, d in changes),
                                       declined=sum(d < -1e-12 for _, d in changes),
                                       tied=sum(abs(d) <= 1e-12 for _, d in changes))
        ordered = sorted(changes, key=lambda pair: pair[1])
        result['largest_changes'][metric] = {'lowest': ordered[:5], 'highest': ordered[-5:][::-1]}
    return result


def read_host(host, banks):
    folder = BASE / 'monitor' / host
    controller = read(folder / 'controller.json') or {}
    observer = read(folder / 'status.json') or {}
    originals = {
        'parent': read(BASE.parent / 'v6-openml12-recover30m-20260928' / 'monitor' / host / 'half2-train.json'),
        'before': read(BASE.parent / 'v6-v55-dual718-continue60m-20260928' / 'monitor' / host / 'half2-train.json'),
    }
    trains = {phase: read(folder / (phase + '-train.json')) for phase in ('half1', 'half2')}
    expected_supervision = 'joint_all' if host == 'dustinstudio' else 'target_only'
    errors = []
    for phase, train in trains.items():
        if not train:
            continue
        try:
            require(train.get('supervision') == expected_supervision, f'{phase}: training loss mismatch')
            require(train.get('forward_passes_per_update') == 1 and train.get('optimizer_steps_per_update') == 1,
                    f'{phase}: training is not one forward/one update')
            source = originals['before'] if phase == 'half1' else trains['half1']
            require(source is not None and train.get('parent_sha256') == source.get('checkpoint_sha256'),
                    f'{phase}: training parent mismatch')
            if train.get('outcome') in ('running', 'training_completed', 'admitted'):
                require(train.get('total_table_count') == 730 and train.get('replay_table_count') == 718,
                        f'{phase}: 12+718 table receipt mismatch')
                require(train.get('schedule_cursor_reset') is (phase == 'half1'), f'{phase}: cursor reset mismatch')
        except ValueError as exc:
            errors.append(str(exc))
        if train.get('outcome') == 'failed':
            errors.append(f"{phase} training failed: {train.get('error_type', '')}: {train.get('error', '')}")
    data = {'host': host, 'training_supervision': expected_supervision,
            'controller_outcome': controller.get('outcome', 'pending'), 'phase': controller.get('phase'),
            'errors': errors, 'evaluations': {}, 'comparisons': {}}
    for family, files in FILES.items():
        evaluations = {}
        raw_baseline = read(folder / files['parent'])
        for stage, filename in files.items():
            ev = read(folder / filename)
            if ev is None:
                evaluations[stage] = {'status': 'pending'}
                continue
            if ev.get('outcome') != 'completed':
                evaluations[stage] = {'status': ev.get('outcome', 'pending'), 'error': ev.get('error')}
                if ev.get('outcome') == 'failed':
                    errors.append(f"{family}/{stage} evaluation failed: {ev.get('error', '')}")
                continue
            source = originals[stage] if stage in originals else trains['half2' if stage == 'final' else stage]
            try:
                validate_evaluation(ev, family, source, banks[family], raw_baseline)
                evaluations[stage] = {'status': 'completed', **compact_eval(ev)}
            except (ValueError, KeyError, TypeError) as exc:
                evaluations[stage] = {'status': 'invalid', 'error': str(exc)}
                errors.append(f'{family}/{stage}: {exc}')
        data['evaluations'][family] = evaluations
        comparisons = {}
        for stage, ev in evaluations.items():
            if stage == 'parent' or ev['status'] != 'completed':
                continue
            comparisons[stage] = {f'vs_{ref}': compare(ev, evaluations[ref]) for ref in ('parent', 'before')
                                  if stage != ref and evaluations[ref]['status'] == 'completed'}
            if stage == 'half2' and evaluations['half1']['status'] == 'completed':
                comparisons[stage]['vs_half1'] = compare(ev, evaluations['half1'])
        data['comparisons'][family] = comparisons
    cumulative = {}
    seconds = 0.0
    updates = 0
    phase_stats = {}
    for phase, train in trains.items():
        if not train:
            phase_stats[phase] = {'status': 'pending'}
            continue
        seconds += float(train.get('successful_update_seconds', 0.0))
        table_updates = train.get('table_updates', {})
        for name, count in table_updates.items():
            cumulative[name] = cumulative.get(name, 0) + count
        stage_updates = train.get('checkpoint_update', train.get('parent_update', 0)) - train.get('parent_update', 0)
        updates += stage_updates
        phase_stats[phase] = dict(status=train.get('outcome', 'pending'),
                                  successful_seconds=train.get('successful_update_seconds', 0.0),
                                  updates=stage_updates, cumulative_successful_seconds=seconds,
                                  coverage=sum(v > 0 for v in table_updates.values()),
                                  openml_coverage=sum(v > 0 for v in train.get('openml_table_updates', {}).values()),
                                  cohort_updates=train.get('cohort_updates', {}),
                                  replay_family_updates=train.get('replay_family_updates', {}))
    data['training'] = dict(phases=phase_stats, successful_seconds_in_saved_receipts=seconds,
                            latest_observed_successful_seconds=observer.get('additional_successful_seconds'),
                            successful_updates_in_saved_receipts=updates,
                            total_table_coverage=sum(v > 0 for v in cumulative.values()),
                            expected_total_tables=730)
    if controller.get('outcome') == 'failed':
        errors.append(f"controller failed: {controller.get('error_type', '')}: {controller.get('error', '')}")
    data['complete'] = (all((t or {}).get('outcome') == 'training_completed' for t in trains.values())
                        and seconds >= 3600 and not errors
                        and data['evaluations']['openml12']['half2']['status'] == 'completed'
                        and data['evaluations']['sparse100']['final']['status'] == 'completed')
    return data


def fmt(value, signed=False):
    if value is None:
        return 'pending'
    return format(value, '+.4f' if signed else '.4f')


def metric_at(host, family, stage, metric, table=None):
    ev = host['evaluations'][family][stage]
    if ev['status'] != 'completed':
        return None
    return ev['macro'][metric] if table is None else ev['tables'][table][metric]


def report(summary):
    lines = ['# 双损失后 OpenML12 再适应：结果', '',
             f"生成时间：{summary['generated_utc']}。仅使用本地 monitor 已落盘回执，未完成阶段标为 pending。", '',
             '原父点已经做过真实12表专项；本轮仅对双损失最终点追加60分钟真实表微调，原父点没有追加训练。'
             '这是实际完整流程的恢复与承接价值比较，不是等新增训练预算的初始化因果对照，也不能单独归因于双损失。', '',
             '三机均使用广播 target_only 读出评估；训练时 Dustin 为 joint_all，DGX2/Mini 为 target_only。'
             'OpenML12 为完整 train support＋完整 test Query，共7894预测；稀疏100为原固定测试bank，共54400预测，测试world与训练相同。'
             '稀疏100只补终点评估，没有本轮半程稀疏结果。旧618本轮未做留存评估，40%回放不能代替留存成绩。'
             '不同主机只与自身父点/起点比较，不做跨机因果归因。', '',
             '## OpenML12 宏指标', '',
             '| 主机 | 节点 | R² | S_log | ΔR² 原父点 | ΔS_log 原父点 | ΔR² 本轮起点 | ΔS_log 本轮起点 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for host in summary['hosts'].values():
        for stage in STAGES:
            ev = host['evaluations']['openml12'][stage]
            cp = host['comparisons']['openml12'].get(stage, {})
            dp = cp.get('vs_parent', {}).get('macro_delta', {})
            db = cp.get('vs_before', {}).get('macro_delta', {})
            cells = [host['host'], STAGE_NAMES[stage], fmt(metric_at(host, 'openml12', stage, 'r2')),
                     fmt(metric_at(host, 'openml12', stage, 'slog')),
                     fmt(dp.get('r2'), True) if stage != 'parent' else '—',
                     fmt(dp.get('slog'), True) if stage != 'parent' else '—',
                     fmt(db.get('r2'), True) if stage not in ('parent', 'before') else '—',
                     fmt(db.get('slog'), True) if stage not in ('parent', 'before') else '—']
            if ev['status'] not in ('completed', 'pending'):
                cells[1] += ' (' + ev['status'] + ')'
            lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## 已验证训练进度与逐表变化数量', '',
              '| 主机 | 阶段 | 保存回执成功秒数 | 更新数 | 本段覆盖/730 | 本段OpenML覆盖/12 |',
              '|---|---|---:|---:|---:|---:|']
    for host in summary['hosts'].values():
        for phase, t in host['training']['phases'].items():
            lines.append(f"| {host['host']} | {phase}: {t['status']} | {t.get('successful_seconds', 0):.3f} | {t.get('updates', 0)} | {t.get('coverage', 0)} | {t.get('openml_coverage', 0)} |")
    lines += ['', '训练进度表为已保存campaign的证据，不把评估等待时间计入训练；running回执不是训练终态。', '',
              '| 主机 | 数据组 | 节点 | 对比 | R²改善/下降/持平 | S_log改善/下降/持平 |',
              '|---|---|---|---|---|---|']
    for host in summary['hosts'].values():
        for family, stages in host['comparisons'].items():
            for stage, refs in stages.items():
                for ref, change in refs.items():
                    nums = ['/'.join(str(change['counts'][m][k]) for k in ('improved', 'declined', 'tied')) for m in METRICS]
                    lines.append('| ' + ' | '.join([host['host'], family, STAGE_NAMES[stage], ref, *nums]) + ' |')
    lines += ['', '## 稀疏100测试宏指标（仅终点补测）', '',
              '| 主机 | 节点 | R² | S_log | ΔR² 原父点 | ΔS_log 原父点 | ΔR² 本轮起点 | ΔS_log 本轮起点 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for host in summary['hosts'].values():
        for stage in ('parent', 'before', 'final'):
            cp = host['comparisons']['sparse100'].get(stage, {})
            dp = cp.get('vs_parent', {}).get('macro_delta', {}); db = cp.get('vs_before', {}).get('macro_delta', {})
            cells = [host['host'], STAGE_NAMES[stage], fmt(metric_at(host, 'sparse100', stage, 'r2')),
                     fmt(metric_at(host, 'sparse100', stage, 'slog')),
                     *[fmt(dp.get(k), True) if stage != 'parent' else '—' for k in METRICS],
                     *[fmt(db.get(k), True) if stage == 'final' else '—' for k in METRICS]]
            lines.append('| ' + ' | '.join(cells) + ' |')
    for host in summary['hosts'].values():
        lines += ['', '## ' + host['host'] + ' 完整逐表', '',
                  f"控制器：{host['controller_outcome']}；当前阶段：{host['phase']}；完整终态：{host['complete']}。"]
        for family, stages in (('openml12', STAGES), ('sparse100', ('parent', 'before', 'final'))):
            lines += ['', '### ' + family, '']
            baseline = host['evaluations'][family]['parent']
            if baseline['status'] != 'completed':
                lines += ['父点评估尚未有效就绪；不生成未验证逐表比较。']
                continue
            for metric in METRICS:
                lines += ['', f'**{metric}**', '', '| 表 | ' + ' | '.join(STAGE_NAMES[s] for s in stages) + ' |',
                          '|---|' + '---:|' * len(stages)]
                for name in sorted(baseline['tables']):
                    values = [fmt(metric_at(host, family, s, metric, name)) for s in stages]
                    lines.append('| ' + ' | '.join([name, *values]) + ' |')
        if host['errors']:
            lines += ['', '### 回执异常', ''] + ['- ' + e for e in host['errors']]
    lines += ['', '## 验证范围', '',
              '所有标记completed的评估均验证：对应训练检查点一致、零优化器更新、target_only读出、固定bank一致、'
              '表数与预测数完整、宏指标等于逐表均值。未到达或不完整回执不会写成完成。', '',
              f'JSON详表：[summary.json]({BASE / "summary.json"})。不以逐表跨主机最佳值拼接共享模型成绩。', '']
    return '\n'.join(lines)


def main():
    banks = {}
    for family, files in FILES.items():
        for host in HOSTS:
            ev = read(BASE / 'monitor' / host / files['parent'])
            if ev and ev.get('outcome') == 'completed':
                banks[family] = ev['bank_sha256']
                break
        banks.setdefault(family, None)
    summary = dict(schema='tabu.dual718.openml12.adaptation.report.v1',
                   generated_utc=datetime.now(timezone.utc).isoformat(), bank_sha256=banks,
                   comparison='practical_full_pipeline_not_equal_additional_budget', old618_retention_evaluated=False,
                   hosts={h: read_host(h, banks) for h in HOSTS})
    summary['all_complete'] = all(h['complete'] for h in summary['hosts'].values())
    atomic_text(BASE / 'summary.json', json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    atomic_text(BASE / 'RESULT.md', report(summary))
    print(json.dumps({'all_complete': summary['all_complete'], 'hosts': {
        h: {'ready': {family: [stage for stage, ev in data['evaluations'][family].items() if ev['status'] == 'completed']
                       for family in FILES}, 'openml12': {
                           stage: {'macro': ev['macro'], 'comparison': {ref: {'macro_delta': c['macro_delta'], 'counts': c['counts']}
                                                                    for ref, c in data['comparisons']['openml12'].get(stage, {}).items()}}
                           for stage, ev in data['evaluations']['openml12'].items() if stage in ('half1', 'half2') and ev['status'] == 'completed'},
            'errors': data['errors']} for h, data in summary['hosts'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
