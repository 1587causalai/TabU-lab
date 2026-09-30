"""Read the three terminal receipts and write the joint OpenML12 comparison."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


HERE = Path(__file__).parent
PRETRAIN = HERE.parent / "openml12-frozen-icl-20260927" / "results.json"
HOSTS = ("dgx2", "dustinstudio", "gongqian-mini")


def remote_json(host, relative):
    command = ["ssh", "-o", "BatchMode=yes", host, "cat", f"experiments/{relative}"]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if completed.returncode or not completed.stdout.strip():
        return None
    return json.loads(completed.stdout)


def main():
    baseline = json.loads(PRETRAIN.read_text())
    rows = {}
    for host in HOSTS:
        campaign = remote_json(host, "openml12-joint-fit-20260927/campaign.json")
        budget = remote_json(host, "openml12-joint-fit-20260927/budget.json")
        test = remote_json(host, "openml12-frozen-icl-20260927/evaluation-openml12-joint-20260927/terminal.json")
        rows[host] = dict(campaign=campaign, budget=budget, test=test)
    (HERE / "status.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    if any(not r["campaign"] or r["campaign"]["outcome"] != "training_completed"
           or not r["test"] or r["test"]["outcome"] != "completed" for r in rows.values()):
        print(json.dumps({h: (r["campaign"] or {}).get("outcome") for h, r in rows.items()}))
        return
    for host, r in rows.items():
        campaign, test, budget = r["campaign"], r["test"], r["budget"]
        assert campaign["shared_checkpoint"] and test["total_predictions"] == 7894
        assert len(test["datasets"]) == 12
        assert test["checkpoint_sha256"] == campaign["checkpoint_sha256"]
        assert test["checkpoint_update"] == campaign["checkpoint_update"] == budget["updates"]
        assert campaign["update_counts"]["history618"] > 0
        assert abs(sum(campaign["update_counts"].values()) - budget["updates"]) == 0
        assert 1400 < budget["successful_seconds"] <= 1442
    names = list(baseline["historical_reference"]["datasets"])
    comp = dict(schema="tabu.openml12.shared-joint-fit-comparison.v1",
                mode="one shared checkpoint per host for all 12 tables",
                target_budget_seconds=1440,
                baseline=baseline, hosts=rows,
                historical_reference=baseline["historical_reference"])
    (HERE / "comparison.json").write_text(json.dumps(comp, ensure_ascii=False, indent=2) + "\n")
    ref = baseline["historical_reference"]["datasets"]
    macro = lambda kind: math.fsum(ref[n][kind]["r2"] for n in names)/12
    lines = ["# OpenML12：一个共享 checkpoint 联合拟合 12 表（有限 support 候选测试）",
             "",
             "**测试口径更正：本页沿用了历史固定有限 support（后9表约136行）和分块 Query，只是候选测试结果，不是用户指定的默认全 train support + 全 test Query 前向。默认口径已完成，见[主结果](RESULT.md)。**",
             "",
             "三台主机各从自己的 joint618 父 checkpoint 出发，各得到**一个**同时拟合 12 张 OpenML 回归表的终点 checkpoint。前一轮 [每表独立微调](../openml12-finetune-20260927/RESULT.md)不符合本轮实验单位，数据保留但不作为联合训练结果。",
             "",
             "每20次更新含 `kin8nm`、`pumadyn32nh` 各3次、其余10表各1次、旧618表回放4次，混合顺序固定 seed。终点不按测试结果选择。",
             "",
             "| 主机/模型 | 联合训练前 R² | 联合训练后 R² | 变化 | 有效更新秒 | 更新数 | 旧618回放更新 | 旧表实际获梯度 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for host in HOSTS:
        r = rows[host]
        pre = baseline["hosts"][host]["macro_r2"]
        post = r["test"]["macro_r2"]
        c, b = r["campaign"], r["budget"]
        model = "H4" if host == "gongqian-mini" else "H8"
        lines.append(f"| {host} {model} | {pre:.4f} | {post:.4f} | {post-pre:+.4f} | {b['successful_seconds']:.1f} | {b['updates']} | {c['update_counts']['history618']} | {c['old_tables_with_gradient']}/618 |")
    wins = []
    for host in HOSTS:
        datasets = rows[host]["test"]["datasets"]
        mlp = sum(datasets[n]["metrics"]["r2"] > ref[n]["mlp"]["r2"] for n in names)
        xgb = sum(datasets[n]["metrics"]["r2"] > ref[n]["xgboost"]["r2"] for n in names)
        wins.append(f"{host} 胜 MLP {mlp}/12、胜 XGBoost {xgb}/12")
    lines += ["", f"历史同划分基线：MLP 宏平均 R² {macro('mlp'):.4f}；XGBoost {macro('xgboost'):.4f}；旧 Gen4 TAR 联合训练 {macro('tar'):.4f}。这些模型的历史训练量与本轮不同。",
              "逐表胜出数：" + "；".join(wins) + "。", "",
              "## 每表固定测试 R²", "",
              "| 表 | dgx2 H8 | Dustin H8 | Mini H4 | MLP | XGBoost | 旧 TAR |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        scores = [rows[h]["test"]["datasets"][n]["metrics"]["r2"] for h in HOSTS]
        lines.append(f"| {n} | {scores[0]:.4f} | {scores[1]:.4f} | {scores[2]:.4f} | {ref[n]['mlp']['r2']:.4f} | {ref[n]['xgboost']['r2']:.4f} | {ref[n]['tar']['r2']:.4f} |")
    lines += ["", "## 证据与解释", "",
              "- 三机各自的父权重、H8/H4、loss 与 CUDA FP64 / MPS FP32 保持独立。新数据身份采用 weights-only 初始化；一更新 admission 后在同身份严格恢复。",
              "- 同一历史固定测试 bank 覆盖 12 表、7,894 行；每台仅加载自己的一个终点 checkpoint 推理，测试期间无参数更新。逐行预测保存在各远端 `~/experiments/openml12-frozen-icl-20260927/evaluation-openml12-joint-20260927/`。",
              "- 与 MLP/XGBoost 比的是相同表、划分和测试指标；MLP/XGBoost 逐表使用完整训练池，TabU 是已有预训练的共享模型，测试时沿用历史有限 support（后9表约136行）。训练算力、先验和推理可见训练行数未配平，因此属于探索性效果对照。",
              "- 旧618实际参与梯度，回放份额见上表。这里未运行旧618终点固定 Query 保持评估，不能仅凭回放次数断言没有遗忘。",
              "- 按用户要求未审计测试行是否曾进入历史训练，结果只描述这批测试地址上的拟合，不作未见表泛化声明。",
              "", "训练配方见 [PLAN.md](PLAN.md)，终点 SHA/逐表 RMSE、MAE 与完整原始回执见 [comparison.json](comparison.json)。", ""]
    (HERE / "RESULT-CANDIDATE.md").write_text("\n".join(lines))
    print(json.dumps({h: rows[h]["test"]["macro_r2"] for h in HOSTS}))


if __name__ == "__main__":
    main()
