# Chinese report expectations intentionally use Chinese punctuation.
# ruff: noqa: RUF001

import json
from types import SimpleNamespace

from tabu_lab.curriculum_v53.reporting import write_report


def _plan():
    return SimpleNamespace(spec={
        "experiment_id": "report-fixture", "stages": [{
            "name": "fit", "question": "Can fixed tables fit?", "max_updates": 4,
            "max_seconds": 20,
        }],
    }, identity={"sha256": "identity", "source": {"sha256": "source"}})


def _probe(value, *, purpose="fit", partition="train"):
    return {"complete": True, "purpose": purpose, "partition": partition,
            "macro": {"query_numeric_normalized_mse": value,
                      "query_discrete_accuracy": None}}


def test_report_matches_same_probe_and_does_not_promote_no_gate(tmp_path):
    for update, value in ((0, 0.8), (4, 0.5)):
        (tmp_path / f"evaluation-000-{update:09d}.json").write_text(json.dumps({
            "update": update, "probes": {
                "fit-bank": _probe(value),
                "old-reserved": _probe(value + 1, purpose="retrospective", partition="test"),
            },
        }))
    terminal = {
        "outcome": "completed", "runtime": {"device": "cpu"}, "update": 4,
        "durable_update": 4, "stage_index": 1, "stage_seconds": [1.2],
        "stage_verdicts": [{"stage": "fit", "verdict": "completed_no_performance_gate"}],
        "exposure": {"a": {"updates": 4, "query_cells": 8,
                           "row_visits": {"0": 4, "1": 4}, "query_row_visits": {"1": 4}}},
        "checkpoint": "checkpoint-progress.pt", "checkpoint_sha256": "checkpoint",
    }
    report = write_report(tmp_path, _plan(), terminal).read_text()
    assert "0.8 → 0.5" in report
    assert "预算完成；未设置性能门槛" in report
    assert "validation gate 达标" not in report
    assert "历史已观察分割的回溯比较" in report
    assert "此次没有完整 final_test 评价" in report
    assert "该评价不执行训练更新或阶段选择" not in report
    assert "| a | 4 | 8 | 2 | 1 | 8 |" in report
    assert "不可变代际" in report


def test_independent_final_evaluation_reads_terminal_without_checkpoint(tmp_path):
    terminal = {"outcome": "completed", "runtime": {"device": "cuda:0"},
                "checkpoint_update": 4, "checkpoint_sha256": "known",
                "probes": {"final": _probe(0.2, purpose="final_test", partition="test")}}
    report = write_report(tmp_path, _plan(), terminal).read_text()
    assert "本次仅评价；不判定训练阶段" in report
    assert "此次包含完整的独立 final_test 冻结评价" in report
    assert "这不证明历史上从未查看过同一测试数据" in report
    assert "0.2 → 0.2" in report


def test_failed_attempt_preserves_read_errors_without_fabricating_metrics(tmp_path):
    (tmp_path / "evaluation-broken.json").write_text("{broken")
    terminal = {"outcome": "failed", "error": "nonfinite gradient", "stage_index": 0,
                "cursor": 1, "stage_seconds": [0.3]}
    report = write_report(tmp_path, _plan(), terminal).read_text()
    assert "未完成（failed）" in report
    assert "evaluation-broken.json: JSONDecodeError" in report
    assert "此 attempt 没有完整 probe 结果" in report
    assert "nonfinite gradient" in report
