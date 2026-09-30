"""Finish the 30m trial with paired OpenML12 and old618 retention evidence."""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from collect_wandb_30m import HOSTS, MONITOR, push


HERE = Path(__file__).parent
BASE = json.loads((HERE / "result-squared.json").read_text())


def remote(host, code):
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "python3", "-"],
                          input=code, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


STAGE_CODE = '''import json,pathlib
h=pathlib.Path.home()/"experiments"
r=h/"openml12-joint30m-squared-20260927"
b=h/"openml12-frozen-icl-20260927"
out={}
for n in (1,2):
 c=r/("half%d/campaign.json"%n)
 e=b/("evaluation-joint30m-squared-half%d-full-20260927/metrics-r2-slog.json"%n)
 out["half%d"%n]={"campaign":None,"metrics":None}
 if c.exists():
  d=json.loads(c.read_text())
  out["half%d"%n]["campaign"]={k:d.get(k) for k in ("outcome","parent_sha256","parent_update","checkpoint_sha256","checkpoint_update","added_updates","added_successful_seconds","old618_tables_with_gradient","error_type","error")}
 if e.exists(): out["half%d"%n]["metrics"]=json.loads(e.read_text())
print(json.dumps(out))
'''


def retention_status(host, label):
    code = '''import json,pathlib
p=pathlib.Path.home()/"experiments/openml12-joint30m-squared-20260927/retention-%s/terminal.json"
if p.exists():
 d=json.loads(p.read_text())
 print(json.dumps({k:d.get(k) for k in ("outcome","error_type","error","bank_checked_masks","checkpoint_sha256","checkpoint_update")}))
else: print(json.dumps({"outcome":"pending"}))
''' % label
    return remote(host, code)


def retention_summary(host):
    code = '''import json,pathlib
r=pathlib.Path.home()/"experiments/openml12-joint30m-squared-20260927"
out={}
for label in ("pre","mid","end"):
 d=json.loads((r/("retention-"+label)/"terminal.json").read_text())
 assert d["outcome"]=="completed" and d["bank_checked_masks"]==1238
 groups={}
 for name,probe in d["probes"].items():
  tables={}
  for t in probe["by_table"]:
   kinds={v["kind"] for k,v in t["by_column"].items() if k.startswith("query/")}
   assert len(kinds)==1
   kind=next(iter(kinds))
   tables[t["table"]]={"kind":kind,"metrics":{k:v for k,v in t["metrics"].items() if k.startswith("query_")}}
  groups[name]={"tables":tables,"macro":probe["macro"]}
 out[label]={"checkpoint_sha256":d["checkpoint_sha256"],"checkpoint_update":d["checkpoint_update"],"bank_checked_masks":d["bank_checked_masks"],"groups":groups}
print(json.dumps(out))
'''
    return remote(host, code)


def wait_tests():
    while True:
        state = {host: remote(host, STAGE_CODE) for host in HOSTS}
        failures = {host: item for host, item in state.items()
                    if any(item[f"half{n}"]["campaign"] and
                           item[f"half{n}"]["campaign"]["outcome"] == "failed"
                           for n in (1, 2))}
        if failures:
            (HERE / "finalize-30m-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
            raise RuntimeError("one or more strict-resume segments failed")
        if all(item["half2"]["metrics"] for item in state.values()):
            for host, item in state.items():
                for n in (1, 2):
                    c, m = item[f"half{n}"]["campaign"], item[f"half{n}"]["metrics"]
                    assert c["outcome"] == "training_completed" and m["outcome"] == "completed"
                    assert c["checkpoint_sha256"] == m["checkpoint_sha256"]
                assert item["half1"]["campaign"]["parent_sha256"] == BASE["hosts"][host]["checkpoint_sha256"]
                assert item["half2"]["campaign"]["parent_sha256"] == item["half1"]["campaign"]["checkpoint_sha256"]
            return state
        time.sleep(30)


def run_retention(label):
    print(json.dumps({"retention": label, "status": "launching"}), flush=True)
    subprocess.run([sys.executable, str(HERE / "launch_old618_eval.py"), "--label", label],
                   cwd=HERE, check=True)
    while True:
        state = {host: retention_status(host, label) for host in HOSTS}
        failed = {host: value for host, value in state.items() if value["outcome"] == "failed"}
        if failed:
            (HERE / f"retention-{label}-failures.json").write_text(json.dumps(failed, indent=2) + "\n")
            raise RuntimeError(f"old618 retention {label} failed")
        if all(value["outcome"] == "completed" for value in state.values()):
            assert all(value["bank_checked_masks"] == 1238 for value in state.values())
            print(json.dumps({"retention": label, "status": "completed"}), flush=True)
            return
        time.sleep(20)


def avg(values):
    valid = [v for v in values if v is not None]
    return statistics.fmean(valid) if valid else None


def aggregate_retention(stage):
    types = {"numeric": [], "nominal": [], "ordinal": []}
    groups = {}
    for probe_name, group in stage["groups"].items():
        groups[probe_name] = {k: v for k, v in group["macro"].items()
                              if k in ("query_numeric_r2", "query_discrete_accuracy",
                                       "query_ordinal_rank_mae")}
        for table in group["tables"].values():
            types[table["kind"]].append(table["metrics"])
    return dict(groups=groups,
                numeric_r2=avg(x.get("query_numeric_r2") for x in types["numeric"]),
                nominal_accuracy=avg(x.get("query_discrete_accuracy") for x in types["nominal"]),
                ordinal_accuracy=avg(x.get("query_discrete_accuracy") for x in types["ordinal"]),
                ordinal_rank_mae=avg(x.get("query_ordinal_rank_mae") for x in types["ordinal"]),
                type_counts={k:len(v) for k,v in types.items()})


def make_result(stages, retention):
    result = {"schema":"openml12-joint30m-squared-result-v1","outcome":"completed",
              "hosts":{},"claim_boundary":"OpenML12 frozen test; old618 fixed-Query training-row retention"}
    for host in HOSTS:
        first, second = stages[host]["half1"], stages[host]["half2"]
        old = retention[host]
        assert old["pre"]["checkpoint_sha256"] == BASE["hosts"][host]["checkpoint_sha256"]
        assert old["mid"]["checkpoint_sha256"] == first["campaign"]["checkpoint_sha256"]
        assert old["end"]["checkpoint_sha256"] == second["campaign"]["checkpoint_sha256"]
        assert sum(len(g["tables"]) for g in old["pre"]["groups"].values()) == 618
        result["hosts"][host] = dict(
            start=dict(update=BASE["hosts"][host]["checkpoint_update"],
                       checkpoint_sha256=BASE["hosts"][host]["checkpoint_sha256"],
                       macro=BASE["hosts"][host]["terminal_macro"],
                       retention=aggregate_retention(old["pre"])),
            mid=dict(update=first["campaign"]["checkpoint_update"],
                     checkpoint_sha256=first["campaign"]["checkpoint_sha256"],
                     added_successful_seconds=first["campaign"]["added_successful_seconds"],
                     macro=first["metrics"]["macro"], retention=aggregate_retention(old["mid"])),
            end=dict(update=second["campaign"]["checkpoint_update"],
                     checkpoint_sha256=second["campaign"]["checkpoint_sha256"],
                     added_successful_seconds=second["campaign"]["added_successful_seconds"],
                     macro=second["metrics"]["macro"], retention=aggregate_retention(old["end"])))
    return result


def report(result):
    lines = ["# OpenML12 平方损失续训 30 分钟", "",
             "三台各从两小时终点检查点按相同 identity 严格恢复；分两段各约 900 秒成功参数更新时间。",
             "OpenML12 测试每表一次前向：整张 train 作 support、整张 test 作 Query，报告 R² 与 S_log。",
             "旧618使用原 1,238-mask 固定 Query bank，报告的是训练行保持，不是未见表泛化。", "",
             "| 主机 | 起点 R² / S_log | 15 分钟 R² / S_log | 30 分钟 R² / S_log |",
             "| --- | ---: | ---: | ---: |"]
    def pair(d): return f'{d["r2"]:.4f} / {d["slog"]:.4f}'
    for host, item in result["hosts"].items():
        lines.append(f'| {host} | {pair(item["start"]["macro"])} | {pair(item["mid"]["macro"])} | {pair(item["end"]["macro"])} |')
    lines += ["", "旧618各类固定 Query 保持（数值越高越好；ordinal rank MAE 越低越好）：", "",
              "| 主机 | 阶段 | numeric R² | nominal accuracy | ordinal accuracy | ordinal rank MAE |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for host, item in result["hosts"].items():
        for label in ("start","mid","end"):
            r=item[label]["retention"]
            lines.append(f'| {host} | {label} | {r["numeric_r2"]:.4f} | {r["nominal_accuracy"]:.4f} | {r["ordinal_accuracy"]:.4f} | {r["ordinal_rank_mae"]:.4f} |')
    lines += ["", "精确 checkpoint SHA、每阶段成功更新秒和六组 old618 宏指标见 `result-30m.json`。"]
    return "\n".join(lines)+"\n"


def finish_wandb(result):
    for host in HOSTS:
        target = result["hosts"][host]["end"]["update"]
        while True:
            code = '''import json,pathlib
p=pathlib.Path(%r)/"wandb-%s.json"
print(json.dumps(json.loads(p.read_text()) if p.exists() else {}))
''' % (MONITOR, host)
            state = remote("dustinstudio", code)
            if state.get("last_update", 0) >= target:
                break
            time.sleep(10)
        item = result["hosts"][host]
        push(host, dict(kind="final", host=host, update=target,
                        additional_successful_seconds=(
                            item["mid"]["added_successful_seconds"] +
                            item["end"]["added_successful_seconds"]),
                        r2_macro=item["end"]["macro"]["r2"],
                        slog_macro=item["end"]["macro"]["slog"]))


def main():
    stages = wait_tests()
    (HERE / "stage30m-receipts.json").write_text(json.dumps(stages, indent=2) + "\n")
    for label in ("pre","mid","end"):
        run_retention(label)
    retention = {host: retention_summary(host) for host in HOSTS}
    for host, item in retention.items():
        (HERE / f"retention-summary-{host}.json").write_text(json.dumps(item, indent=2) + "\n")
    result = make_result(stages, retention)
    (HERE / "result-30m.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "RESULT-30M.md").write_text(report(result))
    finish_wandb(result)
    print(json.dumps({"outcome":"completed","end_macro":{
        host:result["hosts"][host]["end"]["macro"] for host in HOSTS}}), flush=True)


if __name__ == "__main__":
    main()
