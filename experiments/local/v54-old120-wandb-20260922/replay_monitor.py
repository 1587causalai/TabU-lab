"""Read-only SSH monitor for old120 normal120/replay continuations.

Only local status JSON/Markdown are written. Never launches, stops or modifies a
remote trainer. Normal updates, replay updates and inherited model exposure are
reported separately. Best fixed scores cover this output only, not its parents.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess


REMOTE = r'''
from pathlib import Path
from collections import Counter, deque
import datetime, hashlib, json, math, statistics, subprocess
r = Path(CONFIG['remote_root'])
out = Path(CONFIG['remote_output'])
d = out if out.is_absolute() else r / out
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
ps = [line for line in subprocess.check_output(
    ['ps', '-axo', 'pid,ppid,user,etime,args'], text=True).splitlines()
    if 'tabu_lab.cli curriculum-v54 run' in line]
x = dict(utc=now, host=CONFIG['host'], root=str(r),
         output=str(d), processes=ps, ready=(d/'resolved.json').exists())
if x['ready']:
    z = json.loads((d/'resolved.json').read_text())
    identity = z['identity']; stage = z['spec']['stages'][0]
    normal, extra, kinds = Counter(), Counter(), Counter()
    tail = deque(maxlen=120); finite = True; malformed = 0; errors = []
    last_update = int(CONFIG['initial_update'])
    last_normal = int(CONFIG['initial_normal_update'])
    base_extra = last_update - last_normal
    last_extra = base_extra
    policy = stage['loss_replay']
    groups = {'normal120_top5_top20_v1': ('extra_top6', 'extra_top24'),
              'normal120_p99x3_p95x2_p80x1_v2': ('extra_top2', 'extra_top6', 'extra_top24')}
    allowed_kinds = ('normal',) + groups.get(policy['kind'], ())
    if policy['kind'] not in groups:
        errors.append(dict(error='unsupported replay policy'))
    if (CONFIG.get('initial_extra_updates', base_extra) != base_extra
            or CONFIG.get('loss_replay', {}).get('start_extra_updates', base_extra) != base_extra):
        errors.append(dict(error='initial extra counter differs from actual minus normal'))
    known = {entry['id'] for entry in z['spec']['tables'] if entry['role'] == 'train'}
    if (d/'updates.jsonl').exists():
        with (d/'updates.jsonl').open() as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    u = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                kind = u.get('update_kind'); table = u.get('table')
                if table not in known or kind not in allowed_kinds:
                    if len(errors) < 20: errors.append(dict(line=line_number, error='unknown table/update_kind'))
                    continue
                expected_normal = last_normal + int(kind == 'normal')
                expected_extra = last_extra + int(kind != 'normal')
                if (u['update'] != last_update + 1 or u.get('normal_update') != expected_normal
                        or u.get('extra_updates') != expected_extra
                        or u['update'] != u['normal_update'] + u['extra_updates']):
                    if len(errors) < 20: errors.append(dict(line=line_number, error='counter discontinuity', update=u['update']))
                normal[table] += int(kind == 'normal')
                extra[table] += int(kind != 'normal'); kinds[kind] += 1
                finite &= all(isinstance(u.get(k), (int,float)) and math.isfinite(u[k])
                              for k in ('loss','gradient_norm','seconds'))
                last_update = u['update']; last_normal = u['normal_update']; last_extra = u['extra_updates']
                tail.append(u)
    curves = []
    for p in sorted(d.glob('evaluation-*.json')):
        e = json.loads(p.read_text()); bank = e['probes']['train_fit']
        curves.append(dict(update=e['update'], normal_update=e.get('normal_update'),
            extra_updates=e.get('extra_updates'), complete=bank['complete'],
            seconds=bank['seconds'], file=p.name,
            tables={t['table']: {k:v for k,v in t['metrics'].items() if k.startswith('query_')}
                    for t in bank['by_table']}))
    curves.sort(key=lambda e: e['update'])
    terminal = None
    if (d/'terminal.json').exists():
        t = json.loads((d/'terminal.json').read_text())
        terminal = {k:t.get(k) for k in ('outcome','update','durable_update','normal_update',
            'extra_updates','checkpoint','checkpoint_sha256','runtime','error','error_type','stage_verdicts','exposure')}
        if t.get('checkpoint'):
            generation = (d/t['checkpoint']).resolve(strict=True)
            terminal['checkpoint_hash_verified'] = hashlib.sha256(generation.read_bytes()).hexdigest() == t['checkpoint_sha256']
    checkpoint = None
    pointer = d/'checkpoint-progress.pt'
    if pointer.exists():
        generation = pointer.resolve(strict=True)
        sidecar = json.loads(pointer.with_suffix('.json').read_text())
        digest = hashlib.sha256(generation.read_bytes()).hexdigest()
        checkpoint = dict(generation=str(generation), sha256=digest,
            content_address_verified=digest == generation.stem,
            sidecar_matches_generation=sidecar['sha256'] == digest,
            sidecar_update=sidecar['update'], sidecar_cursor=sidecar['cursor'])
    drift = []
    for relative, expected in identity['source']['files'].items():
        path = r/'src/tabu_lab'/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            drift.append(relative)
    stdout = r/'receipts/run.stdout'
    stdout_tail = ''
    if stdout.exists():
        with stdout.open('rb') as handle:
            handle.seek(max(0, stdout.stat().st_size - 2000))
            stdout_tail = handle.read().decode(errors='replace')
    x.update(identity_sha256=identity['sha256'], source_sha256=identity['source']['sha256'],
        data_sha256=identity['data_sha256'], source_files_verified=len(identity['source']['files']),
        source_drift=drift, normal_max_updates=stage['loss_replay']['normal_max_updates'],
        actual_max_updates=stage['max_updates'], update=last_update, normal_update=last_normal,
        extra_updates=last_extra, initial_extra_updates=base_extra,
        new_extra_updates=last_extra-base_extra, loss_replay=policy,
        normal_counts=dict(normal), extra_counts=dict(extra),
        update_kind_counts=dict(kinds), all_logged_updates_finite=finite if tail else None,
        incomplete_jsonl_lines=malformed, journal_errors=errors,
        seconds_per_update=statistics.mean(u['seconds'] for u in tail) if tail else None,
        last_total_seconds=tail[-1]['total_seconds'] if tail else None,
        curves=curves, terminal=terminal, checkpoint=checkpoint, stdout_tail=stdout_tail)
print(json.dumps(x, allow_nan=False))
'''


def read_remote(decisions):
    missing = [key for key in ("host", "remote_root", "remote_output", "initial_update",
                               "initial_normal_update") if key not in decisions]
    if missing:
        return {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "ready": False, "preparation_missing": missing,
                "host": decisions.get("host", "unknown"), "processes": []}
    script = "import json\nCONFIG = json.loads(" + repr(json.dumps(decisions)) + ")\n" + REMOTE
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={decisions.get('ssh_connect_timeout', 8)}",
         decisions.get("host", "unknown"), "python3 -"],
        input=script, text=True, capture_output=True, check=True,
        timeout=decisions.get("ssh_read_timeout", 60))
    return json.loads(result.stdout)


def base_count(value):
    if isinstance(value, dict):
        return int(value["updates"])
    return int(value)


def initial_extra_count(decisions):
    extra = decisions['initial_update'] - decisions['initial_normal_update']
    if extra < 0 or any(type(value) is not int or value != extra for value in (
            decisions.get('initial_extra_updates', extra),
            decisions.get('loss_replay', {}).get('start_extra_updates', extra))):
        raise ValueError('initial actual/normal/extra counters disagree')
    return extra


def is_training_interpreter(process_line):
    """ps columns are pid,ppid,user,etime,args; docker's CLI is only a launcher."""
    columns = process_line.split(None, 4)
    if len(columns) != 5:
        return False
    try:
        args = shlex.split(columns[4])
    except ValueError:
        return False
    return bool(args and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?",
                                     Path(args[0]).name.lower()))


def build_status(snapshot, decisions, checks, references, previous=None):
    """Derive local evidence without changing any input or touching SSH/files."""
    status = dict(snapshot)
    status["model_label"] = decisions["model_label"]
    status["start_mode"] = decisions.get("start_mode", "continuation")
    status["exposure_scope"] = decisions["exposure_scope"]
    status["scope"] = decisions.get("scope", decisions["model_label"] + " replay continuation; train-row fixed Query fit")
    status["fixed_best_scope"] = "evaluations in this output only, not parent best"
    if not snapshot["ready"]:
        status["state"] = ("not_ready" if decisions.get("state") in ("preparing", "launching")
                           else "needs_investigation")
        return status
    status["identity_matches"] = snapshot["identity_sha256"] == decisions.get("identity_sha256")
    status["source_matches"] = (snapshot["source_sha256"] == decisions.get("source_sha256")
                                and not snapshot["source_drift"])
    status["data_matches"] = snapshot["data_sha256"] == decisions.get("data_sha256")
    status["loss_replay"] = snapshot.get("loss_replay", decisions.get("loss_replay", {"kind": "normal120_top5_top20_v1"}))
    status["replay_policy_matches"] = (not decisions.get("loss_replay") or not snapshot.get("loss_replay")
                                       or snapshot["loss_replay"] == decisions["loss_replay"])
    status["initial_extra_updates"] = initial_extra_count(decisions)
    status["new_actual_updates"] = snapshot["update"] - decisions["initial_update"]
    status["new_normal_updates"] = snapshot["normal_update"] - decisions["initial_normal_update"]
    status["new_extra_updates"] = snapshot["extra_updates"] - status["initial_extra_updates"]
    status["inherited_exposure_matches"] = True
    if len(checks) == 120:
        names = {item['table'] for item in checks}
        totals = decisions.get('base_counts', {})
        extras = decisions.get('base_extra_counts', {name: 0 for name in names})
        status["inherited_exposure_matches"] = (
            set(totals) == names and set(extras) == names
            and sum(base_count(value) for value in totals.values()) == decisions['initial_update']
            and sum(base_count(value) for value in extras.values()) == status['initial_extra_updates']
            and all(0 <= base_count(extras[name]) <= base_count(totals[name]) for name in names))
    matching = [line for line in snapshot["processes"]
                if snapshot["output"] in line and snapshot["root"] in line]
    active = [line for line in matching if is_training_interpreter(line)]
    status["matching_launcher_processes"] = [line for line in matching if line not in active]
    status["active_training_processes"] = active
    terminal = snapshot["terminal"]
    problems = (not all(status[k] for k in ("identity_matches", "source_matches", "data_matches"))
                or snapshot["journal_errors"] or snapshot["all_logged_updates_finite"] is False
                or not status["replay_policy_matches"]
                or not status["inherited_exposure_matches"]
                or min(status["new_actual_updates"], status["new_normal_updates"], status["new_extra_updates"]) < 0
                or status["new_actual_updates"] != status["new_normal_updates"] + status["new_extra_updates"]
                or snapshot["update"] != snapshot["normal_update"] + snapshot["extra_updates"]
                or len(active) > 1
                or (terminal is not None and not terminal.get("checkpoint_hash_verified", False)))
    status["state"] = ("needs_investigation" if problems else terminal["outcome"] if terminal
                       else "running" if active else "needs_investigation")
    if previous and previous.get("ready") and previous.get("identity_sha256") == snapshot["identity_sha256"]:
        status["previous_readback_utc"] = previous["utc"]
        status["update_delta_since_previous_readback"] = snapshot["update"] - previous["update"]
        status["normal_delta_since_previous_readback"] = snapshot["normal_update"] - previous["normal_update"]
        status["extra_delta_since_previous_readback"] = snapshot["extra_updates"] - previous["extra_updates"]
    complete = [c for c in snapshot["curves"] if c["complete"] and len(c["tables"]) == 120]
    latest = complete[-1] if complete else None
    status["latest_fixed_update"] = latest["update"] if latest else None
    rows = []
    for item in checks:
        table = item["table"]; numeric = item["target_kind"] == "numeric"
        metric = "query_numeric_r2" if numeric else "query_discrete_accuracy"
        points = [(c["update"], c["tables"][table][metric]) for c in complete
                  if c["tables"].get(table, {}).get(metric) is not None]
        best = max(points, key=lambda pair: pair[1]) if points else None
        metrics = latest["tables"].get(table, {}) if latest else {}
        nmse = [c["tables"][table]["query_numeric_normalized_mse"] for c in complete
                if c["tables"].get(table, {}).get("query_numeric_normalized_mse") is not None]
        normal = snapshot["normal_counts"].get(table, 0)
        extra = snapshot["extra_counts"].get(table, 0)
        inherited = base_count(decisions["base_counts"].get(table, 0))
        inherited_extra = base_count(decisions.get("base_extra_counts", {}).get(table, 0))
        if inherited_extra < 0 or inherited_extra > inherited:
            raise ValueError('per-table inherited extra exposure exceeds inherited total')
        rows.append(dict(table=table, target_column=item["target_column"],
            target_kind=item["target_kind"], normal_new=normal, extra_new=extra,
            inherited_model_updates=inherited, cumulative_model_updates=inherited+normal+extra,
            inherited_normal_updates=inherited-inherited_extra, inherited_extra_updates=inherited_extra,
            cumulative_normal_updates=inherited-inherited_extra+normal,
            cumulative_extra_updates=inherited_extra+extra,
            current=metrics.get(metric), best=best[1] if best else None,
            best_update=best[0] if best else None,
            current_nmse=metrics.get("query_numeric_normalized_mse"),
            lowest_nmse=min(nmse) if nmse else None,
            query_unique_targets=metrics.get("query_unique_targets"),
            reference=references[table]["metrics"]))
    status["tables"] = rows
    return status


def number(value):
    return "—" if value is None else f"{value:.6f}"


def render_status(status):
    lines = [f"# {status['model_label']}：高损失加训 old120 当前读回", "",
             f"UTC {status['utc']}；{status.get('host', 'unknown')}；{status['state']}。", ""]
    if not status["ready"]:
        return "\n".join(lines + ["任务尚未就绪，本次只读回查未启动或恢复训练。", ""])
    lines += [f"模型累计实际更新 {status['update']:,} / {status['actual_max_updates']:,}；"
              f"正常更新 {status['normal_update']:,} / {status['normal_max_updates']:,}；"
              f"加训累计 {status['extra_updates']:,}（继承 {status['initial_extra_updates']:,}）。", "",
              f"{'本任务新增' if status.get('start_mode') == 'weights_only' else '本接续新增'}：正常 {status['new_normal_updates']:,}，加训 {status['new_extra_updates']:,}；"
              f"合计 {status['new_actual_updates']:,}。", "",
              f"身份匹配 {status['identity_matches']}；源码匹配 {status['source_matches']}；"
              f"数据身份匹配 {status['data_matches']}；已记录 loss/gradient 有限 {status['all_logged_updates_finite']}。", ""]
    if status.get("update_delta_since_previous_readback") is not None:
        lines += [f"较 UTC {status['previous_readback_utc']}：实际 +{status['update_delta_since_previous_readback']:,}，"
                  f"正常 +{status['normal_delta_since_previous_readback']:,}，加训 +{status['extra_delta_since_previous_readback']:,}。", ""]
    if status.get("seconds_per_update") is not None:
        lines += [f"最近最多120个实际优化步平均 {status['seconds_per_update']:.3f} 秒/步（包含normal及extra）。", ""]
    weighted = status.get('loss_replay', {}).get('kind') == 'normal120_p99x3_p95x2_p80x1_v2'
    schedule = ('每轮120表正常覆盖，然后top2完整3遍、top6完整2遍、top24完整1遍；嵌套叠加共42次加训，最高2表各加训6次。'
                if weighted else '每轮120表正常覆盖，然后top6及top24各加训一次，共30次加训。')
    lines += [f"{status['model_label']}；{schedule}"
              f"train_cycle只统计正常120步。曝光范围：{status['exposure_scope']}。", "",
              f"最新完整固定Query评估：{status['latest_fixed_update'] if status['latest_fixed_update'] is not None else '本输出尚无完整评估'}。"
              "当前/最佳均仅来自本次输出的固定训练行Query评估，最佳不代表父任务最佳。"
              "每表8掩码、408次target Query曝光；未使用reserved/final_test。", "",
              "| 表 | 正常新增 | 加训新增 | 加训累计 | 模型累计 | 目标列/类型 | 当前 R² / Acc | 本输出最佳 @实际步 | 当前 / 最低 NMSE | 可见支持参考 | Query唯一地址 |",
              "|---|---:|---:|---:|---:|---|---:|---|---|---|---:|"]
    for row in status["tables"]:
        numeric = row["target_kind"] == "numeric"; ref = row["reference"]
        reference = (f"均值 R² {ref['r2']:.4f}; NMSE {ref['nmse']:.4f}" if numeric
                     else f"多数类 Acc {ref['accuracy']:.4f}")
        best = "—" if row["best"] is None else f"{row['best']:.6f} @{row['best_update']}"
        nmse = f"{number(row['current_nmse'])} / {number(row['lowest_nmse'])}" if numeric else "—"
        lines.append(f"| {row['table']} | {row['normal_new']} | {row['extra_new']} | {row['cumulative_extra_updates']} | {row['cumulative_model_updates']} | "
                     f"{row['target_column']}/{row['target_kind']} | {number(row['current'])} | {best} | {nmse} | "
                     f"{reference} | {row['query_unique_targets'] if row['query_unique_targets'] is not None else '—'} |")
    if status.get("journal_errors"):
        lines += ["", "日志计数存在异常，详见JSON；未自动重启训练。"]
    return "\n".join(lines) + "\n"


def main(panel):
    panel = Path(panel).resolve()
    decisions = json.loads((panel / "decisions.json").read_text())
    previous_files = sorted(panel.glob("status-*.json"))
    previous = json.loads(previous_files[-1].read_text()) if previous_files else None
    snapshot = read_remote(decisions)
    checks = json.loads((panel / "parent-preparation.json").read_text())["table_checks"]
    references = {item["table"]: item for item in json.loads(
        (panel / "naive-query-reference.json").read_text())["records"]}
    status = build_status(snapshot, decisions, checks, references, previous)
    stamp = datetime.datetime.fromisoformat(status["utc"]).strftime("%Y%m%dT%H%M%SZ")
    (panel / f"status-{stamp}.json").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    text = render_status(status)
    (panel / f"status-{stamp}.md").write_text(text)
    (panel / "CURRENT.md").write_text(text)
    print(json.dumps({key: status.get(key) for key in (
        "utc", "state", "update", "normal_update", "extra_updates", "new_actual_updates", "new_extra_updates",
        "latest_fixed_update", "identity_matches", "source_matches", "data_matches")}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
