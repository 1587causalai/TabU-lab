"""Collect terminal R²/S_log, compare baselines, and close the W&B runs."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from collect_wandb_squared import HOSTS, push


HERE = Path(__file__).parent
BASE = json.loads((HERE.parent / "openml12-joint-fit-20260927/metrics-r2-slog.json").read_text())
REMOTE = '''import json,pathlib
home=pathlib.Path.home()
run=home/"experiments/openml12-joint2h-squared-20260927/campaign.json"
eval=home/"experiments/openml12-frozen-icl-20260927/evaluation-joint2h-squared-full-20260927/metrics-r2-slog.json"
out={"campaign":None,"metrics":None}
if run.exists():
 c=json.loads(run.read_text())
 out["campaign"]={k:c.get(k) for k in ("outcome","objective","parent_sha256","parent_update","checkpoint_sha256","checkpoint_update","successful_update_seconds","cohort_updates","old_tables_with_gradient","error_type","error")}
if eval.exists(): out["metrics"]=json.loads(eval.read_text())
print(json.dumps(out))
'''


def remote(host):
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "python3", "-"],
                          input=REMOTE, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


def monitoring_update(host):
    code = '''import json,pathlib
r=pathlib.Path.home()/"experiments/openml12-joint2h-squared-monitor-20260927"
p=r/("wandb-%s.json")
print(json.dumps(json.loads(p.read_text()) if p.exists() else {}))
''' % host
    done = subprocess.run(["ssh", "dustinstudio", "python3", "-"],
                          input=code, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


def build(state):
    names = list(BASE["tables"])
    result = {"schema": "openml12-joint2h-squared-result-v1", "outcome": "completed",
              "objective": "squared", "hosts": {}, "tables": {},
              "xgboost_macro": BASE["macro"]["xgboost"]}
    for host in HOSTS:
        campaign, metrics = state[host]["campaign"], state[host]["metrics"]
        assert campaign["outcome"] == metrics["outcome"] == "completed" or (
            campaign["outcome"] == "training_completed" and metrics["outcome"] == "completed")
        assert campaign["objective"] == {"kind": "squared"}
        assert campaign["checkpoint_sha256"] == metrics["checkpoint_sha256"]
        assert campaign["checkpoint_update"] == metrics["checkpoint_update"]
        assert set(metrics["tables"]) == set(names)
        expected = BASE["macro"][f"{host}_tuned"]
        result["hosts"][host] = dict(parent_checkpoint_sha256=campaign["parent_sha256"],
                                     parent_update=campaign["parent_update"],
                                     checkpoint_sha256=campaign["checkpoint_sha256"],
                                     checkpoint_update=campaign["checkpoint_update"],
                                     successful_update_seconds=campaign["successful_update_seconds"],
                                     cohort_updates=campaign["cohort_updates"],
                                     old_tables_with_gradient=campaign["old_tables_with_gradient"],
                                     parent_macro=expected, terminal_macro=metrics["macro"])
    for name in names:
        result["tables"][name] = {
            "support_rows": BASE["tables"][name]["support_rows"],
            "query_rows": BASE["tables"][name]["query_rows"],
            "xgboost": BASE["tables"][name]["models"]["xgboost"],
            **{host: state[host]["metrics"]["tables"][name] for host in HOSTS}}
    return result


def report(result):
    lines = ["# OpenML12 共同微调：平方损失阶段", "",
             "三台主机各训练一个共享检查点，同时拟合 12 张 OpenML 表，并以 old618 进入梯度回放。",
             "每张表一次测试前向：全部训练行作为 support，全部测试行作为 Query；共 7,894 条测试预测。",
             "指标为测试 R² 与 S_log；S_log 使用训练目标的 median/MAD 固定标尺。",
             "本阶段从 24 分钟联合检查点仅继承权重，优化器、随机状态、游标及曝光重置；外层损失均为平方损失。", "",
             "| 模型 | 起点 R² | 终点 R² | 起点 S_log | 终点 S_log | 更新 | 成功更新秒 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for host in HOSTS:
        h = result["hosts"][host]
        p, t = h["parent_macro"], h["terminal_macro"]
        lines.append(f'| {host} | {p["r2"]:.4f} | {t["r2"]:.4f} | {p["slog"]:.4f} | {t["slog"]:.4f} | {h["checkpoint_update"]} | {h["successful_update_seconds"]:.1f} |')
    x = result["xgboost_macro"]
    lines.append(f'| XGBoost | — | {x["r2"]:.4f} | — | {x["slog"]:.4f} | — | — |')
    lines += ["", "old618 回放进入梯度；下表是 12 张真实表测试效果，不代表 old618 的固定 Query 留存效果。", "",
              "| 表 | support | Query | dgx2 R² / S_log | dustinstudio R² / S_log | gongqian-mini R² / S_log | XGBoost R² / S_log |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, t in result["tables"].items():
        def pair(d): return f'{d["r2"]:.3f} / {d["slog"]:.3f}'
        lines.append(f'| {name} | {t["support_rows"]} | {t["query_rows"]} | {pair(t["dgx2"])} | {pair(t["dustinstudio"])} | {pair(t["gongqian-mini"])} | {pair(t["xgboost"])} |')
    return "\n".join(lines) + "\n"


def main():
    while True:
        state = {}
        for host in HOSTS:
            try:
                state[host] = remote(host)
            except Exception as error:
                print(json.dumps({"host": host, "remote_error": type(error).__name__}), flush=True)
                state[host] = {}
        failures = {h: v["campaign"] for h, v in state.items()
                    if v.get("campaign") and v["campaign"]["outcome"] == "failed"}
        if failures:
            (HERE / "finalize-squared-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
            raise RuntimeError("one or more squared-loss training runs failed")
        if all(v.get("metrics") for v in state.values()):
            result = build(state)
            (HERE / "result-squared.json").write_text(json.dumps(result, indent=2) + "\n")
            (HERE / "RESULT-SQUARED.md").write_text(report(result))
            for host in HOSTS:
                target = result["hosts"][host]["checkpoint_update"]
                while monitoring_update(host).get("last_update", 0) < target:
                    time.sleep(10)
                h = result["hosts"][host]
                push(host, dict(kind="final", host=host, update=target,
                                additional_successful_seconds=h["successful_update_seconds"],
                                r2_macro=h["terminal_macro"]["r2"],
                                slog_macro=h["terminal_macro"]["slog"]))
            print(json.dumps({"outcome": "completed", "macro": {
                h: result["hosts"][h]["terminal_macro"] for h in HOSTS}}), flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
