"""Build the default full-train-support OpenML12 result from host receipts."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


HERE = Path(__file__).parent
BASE = HERE.parent / "openml12-frozen-icl-20260927" / "results.json"
HOSTS = ("dgx2", "dustinstudio", "gongqian-mini")


def get(host, name):
    path = f"experiments/openml12-frozen-icl-20260927/{name}/terminal.json"
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "cat", path],
                          capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


def main():
    baseline = json.loads(BASE.read_text())
    prior = json.loads((HERE / "status.json").read_text())
    both = json.loads((HERE / "metrics-r2-slog.json").read_text())
    assert both["schema"] == "openml12-full-context-r2-slog-v1"
    assert both["floor_activated_tables"] == []
    parent = {h: get(h, "evaluation-parent-fulltrain-fulltest-20260927") for h in HOSTS}
    tuned = {h: get(h, "evaluation-fulltrain-fulltest-20260927") for h in HOSTS}
    names = list(baseline["historical_reference"]["datasets"])
    reference = baseline["historical_reference"]["datasets"]
    assert set(both["tables"]) == set(names)
    for host in HOSTS:
        a, b = parent[host], tuned[host]
        assert a["outcome"] == b["outcome"] == "completed"
        assert a["total_predictions"] == b["total_predictions"] == 7894
        assert len(a["datasets"]) == len(b["datasets"]) == 12
        assert a["checkpoint_sha256"] == baseline["hosts"][host]["checkpoint_sha256"]
        assert b["checkpoint_sha256"] == prior[host]["campaign"]["checkpoint_sha256"]
        assert b["checkpoint_update"] == prior[host]["campaign"]["checkpoint_update"]
        assert a["model_state_unchanged"] and b["model_state_unchanged"]
        assert a["checkpoint_unchanged"] and b["checkpoint_unchanged"]
        for name in names:
            da, db = a["datasets"][name], b["datasets"][name]
            ntrain = baseline["hosts"]["dgx2"]["datasets"][name]["train_n"]
            ntest = baseline["hosts"]["dgx2"]["datasets"][name]["metrics"]["n"]
            assert da["support_rows"] == db["support_rows"] == ntrain
            assert da["query_rows"] == db["query_rows"] == ntest
            assert da["forwards"] == db["forwards"] == 1
            scored = both["tables"][name]["models"]
            for stage, receipt in (("parent", da), ("tuned", db)):
                item = scored[f"{host}_{stage}"]
                assert item["checkpoint_sha256"] == (a if stage == "parent" else b)["checkpoint_sha256"]
                assert abs(item["r2"] - receipt["metrics"]["r2"]) < 1e-10
            assert abs(scored["xgboost"]["r2"] - reference[name]["xgboost"]["r2"]) < 1e-10
    (HERE / "full-context-receipts.json").write_text(
        json.dumps(dict(parent=parent, tuned=tuned), ensure_ascii=False, indent=2) + "\n")
    macro = lambda kind: math.fsum(reference[n][kind]["r2"] for n in names) / 12
    lines = ["# OpenML12：单检查点联合拟合，全 train support 测试",
             "",
             "**默认测试口径已更正**：每张表一次前向，全部 train 行作为有标签 support，全部 test 行作为无标签 Query；Query 真值只用于事后计分。12 表共 31,551 个 train 行、7,894 个 test 行。原来的[有限 support 分块结果](RESULT-CANDIDATE.md)只是候选口径，不能当作本页结果。",
             "",
             "三台主机各用自己的 joint618 父 checkpoint 联合拟合 12 表，每台仅一个共享终点。每20次参数更新含重点2表各3次、其他10表各1次、旧618回放4次；每机约1,440秒成功更新时间。",
             "",
             "| 主机 | 微调前 R² / S_log | 微调后 R² / S_log | R² 变化 | S_log 变化 |",
             "|---|---:|---:|---:|---:|"]
    for h in HOSTS:
        a, b = parent[h]["macro_r2"], tuned[h]["macro_r2"]
        sa = both["macro"][f"{h}_parent"]["slog"]
        sb = both["macro"][f"{h}_tuned"]["slog"]
        lines.append(f"| {h} | {a:.4f} / {sa:.4f} | {b:.4f} / {sb:.4f} | {b-a:+.4f} | {sb-sa:+.4f} |")
    xgb = both["macro"]["xgboost"]
    lines += ["", f"同划分历史 XGBoost 宏平均：R² {xgb['r2']:.4f} / S_log {xgb['slog']:.4f}。历史 MLP 的 R² 为 {macro('mlp'):.4f}，本次未补算其 S_log。TabU 和 XGBoost 都能利用完整 train 池；但 TabU 是跨表共享且带预训练历史的模型，训练预算与模型单位仍不同。", "",
              "## 每表测试 R² / S_log 与样本数", "",
              "每个模型单元格依次为 **R² / S_log**；两项均越高越好。", "",
              "| 表 | 全 train support + 全 test Query | dgx2 | Dustin | Mini | XGBoost |",
              "|---|---:|---:|---:|---:|---:|"]
    for n in names:
        d = tuned["dgx2"]["datasets"][n]
        scores = both["tables"][n]["models"]
        fmt = lambda item: f"{item['r2']:.4f} / {item['slog']:.4f}"
        cells = [fmt(scores[f"{h}_tuned"]) for h in HOSTS] + [fmt(scores["xgboost"])]
        lines.append(f"| {n} | {d['support_rows']} + {d['query_rows']} | " + " | ".join(cells) + " |")
    lines.append("| **12 表宏平均** | — | " + " | ".join(
        f"{both['macro'][model]['r2']:.4f} / {both['macro'][model]['slog']:.4f}"
        for model in [*(f"{h}_tuned" for h in HOSTS), "xgboost"]) + " |")
    lines += ["", "## 口径与回执", "",
              "- 训练时每次仅取一张表：airfoil 801+401、concrete 549+275、qsar 484+242、bodyfat 134+67，其余8表各136+68；前一项为可见标签 support，后一项 Query 标签仅用于梯度损失。训练池逐表轮换，不是一次前向拼接12张表。",
              "- 测试时改为整张 train 表加整张 test 表。例如 airfoil 1,202+301；concrete 在实际冻结划分中为 824+206；kin8nm 为 6,553+1,639=8,192。每表一次前向，12表合计12次，无梯度更新。",
              "- XGBoost 对每表用该表完整 train 池拟合，再预测同一 test 池；这里测试地址、标签和指标对齐。按用户要求未审计测试行历史训练曝光，不作未见表泛化声明。",
              "- S_log 用训练集标签中位数和原始 MAD 标准化，在冻结测试集逐行预测上事后补算；预设正 floor 为 1e-12（原始目标单位），12 表均未触发。XGBoost 直接重评分历史保存的逐行预测，并复核其 R² 与原记录一致。这是探索性回顾指标，不是本轮预注册训练目标。",
              "- 三机旧618表均有实际梯度回放，但本次没有旧618终点保持评估。",
              "", "双指标逐表精确值和计分参数见 [metrics-r2-slog.json](metrics-r2-slog.json)；父/终点模型 SHA、每表 RMSE/MAE、前向行数和完整原始回执见 [full-context-receipts.json](full-context-receipts.json)；训练配方见 [PLAN.md](PLAN.md)。", ""]
    (HERE / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps({h: dict(parent=parent[h]["macro_r2"], tuned=tuned[h]["macro_r2"])
                      for h in HOSTS}))


if __name__ == "__main__":
    main()
