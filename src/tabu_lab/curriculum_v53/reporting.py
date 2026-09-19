"""Human-readable projections of an attempt's frozen evaluation receipts.

No checkpoint, data tensor, parent attempt or model is loaded to write a report.
The machine-readable receipts remain authoritative.
"""

# Chinese report prose intentionally uses Chinese punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import json
import math
from pathlib import Path


def _cell(value) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _number(value, *, accuracy=False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{100 * value:.2f}%" if accuracy else f"{value:.6g}"


def _records(output: Path, terminal: dict):
    records, errors = [], []
    for path in sorted(output.glob("evaluation-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("probes"), dict):
                raise ValueError("evaluation receipt needs a probes mapping")
            records.append((value.get("update"), path.name, value["probes"]))
        except (OSError, ValueError) as error:
            errors.append(f"{path.name}: {type(error).__name__}")
    if isinstance(terminal.get("probes"), dict):
        records.append((terminal.get("checkpoint_update", terminal.get("update")),
                        "terminal.json", terminal["probes"]))
    records.sort(key=lambda item: (
        item[0] if type(item[0]) is int else -1, item[1]
    ))
    return records, errors


def _boundary(probe: dict) -> str:
    if probe.get("purpose") == "retrospective":
        return "历史已观察分割的回溯比较"
    if probe.get("partition") == "train":
        return "训练池固定 mask"
    if probe.get("partition") == "validation":
        return "validation；其他列可见的 transductive 协议"
    if probe.get("purpose") == "final_test":
        return "独立 final_test；其他列可见的 transductive 协议"
    return _cell(probe.get("partition", "未声明"))


def _stage_rows(plan, terminal):
    verdicts = {row["stage"]: row for row in terminal.get("stage_verdicts", [])}
    current = terminal.get("stage_index")
    seconds = terminal.get("stage_seconds", [])
    independent = "checkpoint_update" in terminal and current is None
    for index, stage in enumerate(plan.spec["stages"]):
        entry = verdicts.get(stage["name"])
        if entry is not None:
            count = stage["max_updates"]
            verdict = {
                "completed_no_performance_gate": "更新预算完成；未设置性能门槛",
                "passed": "validation gate 达标",
                "failed": "validation gate 未达标",
            }.get(entry["verdict"], entry["verdict"])
        elif independent:
            count, verdict = None, "本次仅评价；不判定训练阶段"
        elif current == index:
            count = terminal.get("cursor")
            verdict = f"未完成（{terminal.get('outcome', 'unknown')}）"
        elif isinstance(current, int) and index > current:
            count, verdict = 0, "未开始"
        else:
            count, verdict = None, "无阶段 verdict 记录"
        elapsed = seconds[index] if index < len(seconds) else None
        yield (stage["name"], stage["question"], stage["max_updates"], stage["max_seconds"],
               _number(count), _number(elapsed), verdict)


def write_report(output: Path, plan, terminal: dict) -> Path:
    """Write report.md from this attempt's receipts; return its path."""
    output = Path(output)
    records, read_errors = _records(output, terminal)
    identity = terminal.get("identity", plan.identity)
    runtime = terminal.get("runtime", {})
    device = runtime.get("device", "未记录")
    lines = [
        "# V5.3 课程实验报告", "",
        f"实验：`{_cell(plan.spec['experiment_id'])}`；状态："
        f"`{_cell(terminal.get('outcome', 'unknown'))}`；证据等级："
        f"`{_cell(terminal.get('status', 'local_unissued'))}`。", "",
        f"本次运行设备：`{_cell(device)}`。本报告只汇总此 attempt 的记录；"
        "累计更新、阶段用时和表曝光若来自 strict resume，则包含其继承链。"
        "固定 probe 起止仅使用此目录已有的完整评价，不补写父 attempt 的结果。", "",
        "CPU 记录仅支持该本地 CPU 配方；GPU 记录仅支持所记录设备与配方。"
        "两者都不自动证明大表、大模型、其他后端或分布式训练合格。", "",
        f"累计更新：{_number(terminal.get('update', terminal.get('checkpoint_update')))}；"
        f"最后持久化更新：{_number(terminal.get('durable_update'))}；"
        f"累计用时：{_number(terminal.get('total_seconds'))} 秒。", "",
        "## 阶段问题与实际结束状态", "",
        "| 阶段 | 问题 | 更新预算 | 秒数预算 | 实际更新 | 累计秒数 | 实际 verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |"
                 for row in _stage_rows(plan, terminal))
    lines += ["", "无性能 gate 的阶段只报告预算完成；不把运行完成写成质量达标。", "",
              "## 固定 probe 起止", "",
              "下表比较同名 probe 的首末完整记录。numeric normalized MSE 使用每个 episode "
              "forward-visible codec 的尺度，不等同于历史按固定目标方差定义的 NMSE；"
              "不同 probe 名称不拼成一条改善曲线。", "",
              "| Probe | 证据范围 | 更新起 → 止 | Query numeric normalized MSE | "
              "Query discrete accuracy | 记录 |",
              "|---|---|---|---:|---:|---|"]
    banks, final_test_complete = {}, False
    independent_final_test = False
    for update, filename, probes in records:
        for name, probe in probes.items():
            if not isinstance(probe, dict) or probe.get("complete") is not True:
                read_errors.append(f"{filename}: {_cell(name)} 缺少完整评价标记")
                continue
            banks.setdefault(name, []).append((update, filename, probe))
            final_test_complete |= probe.get("purpose") == "final_test"
            independent_final_test |= (
                probe.get("purpose") == "final_test" and filename == "terminal.json"
                and "checkpoint_update" in terminal and terminal.get("outcome") == "completed"
            )
    for name, items in sorted(banks.items()):
        first, last = items[0], items[-1]
        first_metrics, last_metrics = first[2].get("macro", {}), last[2].get("macro", {})
        metrics = []
        for metric, accuracy in (("query_numeric_normalized_mse", False),
                                 ("query_discrete_accuracy", True)):
            metrics.append(f"{_number(first_metrics.get(metric), accuracy=accuracy)} → "
                           f"{_number(last_metrics.get(metric), accuracy=accuracy)}")
        source = f"[起]({first[1]}) / [止]({last[1]})"
        row = (name, _boundary(last[2]), f"{_number(first[0])} → {_number(last[0])}",
               *metrics, source)
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    if not banks:
        lines.append("| — | 此 attempt 没有完整 probe 结果 | — | — | — | — |")
    lines += [""]
    if independent_final_test:
        lines += ["此次包含完整的独立 final_test 冻结评价。该评价不执行训练更新或阶段选择；"
                  "这不证明历史上从未查看过同一测试数据。", ""]
    elif final_test_complete:
        lines += ["已读取 final_test 指标，但此记录不能确认为已完成的独立冻结评价；"
                  "不据此判定其是否参与选择。", ""]
    else:
        lines += ["此次没有完整 final_test 评价，不报告最终测试结论。", ""]
    lines += ["## 按表训练曝光", "",
              "曝光从 terminal 的累计链状态读取；固定 probe 的评价曝光不计入训练更新。", "",
              "| 表 | 更新数 | Query cell 暴露 | 不同行访问数 | 不同 Query 行数 | 行访问总数 |",
              "|---|---:|---:|---:|---:|---:|"]
    exposure = terminal.get("exposure", {})
    for name, item in sorted(exposure.items()):
        visits, query_visits = item.get("row_visits", {}), item.get("query_row_visits", {})
        row = (name, item.get("updates", 0), item.get("query_cells", 0),
               len(visits), len(query_visits), sum(visits.values()))
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    if not exposure:
        lines.append("| — | 未记录训练曝光 | — | — | — | — |")
    lines += ["", "## 身份、checkpoint 与原始记录", "",
              f"- 实验身份 SHA-256：`{_cell(identity.get('sha256', '未记录'))}`。",
              f"- Manifest SHA-256：`{_cell(identity.get('manifest_sha256', '未记录'))}`。",
              f"- 数据身份 SHA-256：`{_cell(identity.get('data_sha256', '未记录'))}`。",
              f"- 源码 SHA-256：`{_cell(identity.get('source', {}).get('sha256', '未记录'))}`。",
              "- 权威结束记录与身份：[terminal.json](terminal.json)。"]
    checkpoint = terminal.get("checkpoint")
    if checkpoint:
        lines += [f"- 当前 checkpoint 指针：[{_cell(checkpoint)}]({_cell(checkpoint)})。",
                  "- 训练配置与身份：[resolved.json](resolved.json)。",
                  "- checkpoint 指针链接到 `checkpoints/<sha256>.pt` 不可变代际；"
                  "搬移时保留整个 attempt 目录和相对符号链接，不只复制指针文件。"]
    if terminal.get("checkpoint_sha256"):
        lines.append(f"- 冻结 checkpoint SHA-256：`{terminal['checkpoint_sha256']}`。")
    if terminal.get("checkpoint_source"):
        source = terminal["checkpoint_source"]
        lines.append(f"- 冻结 checkpoint 来源：[{_cell(source)}](<{source}>)。")
    if terminal.get("error"):
        lines += ["", f"结束原因：{_cell(terminal['error'])}。失败记录和最后有效 checkpoint 保留。"]
    if read_errors:
        lines += ["", "以下评价记录无法完整读取，未纳入指标摘要：", ""]
        lines.extend(f"- {_cell(error)}" for error in read_errors)
    lines += ["", "本文件是可重建的可读投影；指标、身份与结束原因以原始 JSON 记录为准。", ""]
    path = output / "report.md"
    temporary = output / ".report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path
