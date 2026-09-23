"""Model labels and inherited exposure must not leak from the H4 monitor."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


spec = importlib.util.spec_from_file_location("replay_monitor", Path(__file__).with_name("replay_monitor.py"))
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


@pytest.mark.parametrize("label,host,base", [
    ("standard Small, 8 heads", "dgx2", 1000),
    ("Nano, 4 heads, Unit0", "dustinstudio", 1600),
])
def test_model_label_and_actual_normal_extra_exposure(label, host, base):
    snapshot = dict(utc="2026-09-22T04:00:00+00:00", host=host, ready=True,
                    identity_sha256="identity", source_sha256="source", data_sha256="data",
                    source_drift=[], output="/new/runs/replay", root="/new",
                    processes=["123 1 cms 00:10 python -m tabu_lab.cli curriculum-v54 run --output-root /new/runs/replay"],
                    terminal=None, journal_errors=[], all_logged_updates_finite=True,
                    update=base*120+150, normal_update=base*120+120, extra_updates=30,
                    normal_counts={"table": 1}, extra_counts={"table": 2}, curves=[],
                    normal_max_updates=983040, actual_max_updates=1200000)
    decisions = dict(model_label=label, exposure_scope="model old120 history only",
                     identity_sha256="identity", source_sha256="source", data_sha256="data",
                     initial_update=base*120, initial_normal_update=base*120,
                     base_counts={"table": base})
    status = monitor.build_status(snapshot, decisions,
                                  [dict(table="table", target_kind="numeric", target_column=1)],
                                  {"table": {"metrics": dict(r2=0., nmse=1.)}})
    assert status["state"] == "running"
    assert status["tables"][0]["cumulative_model_updates"] == base+3
    assert status["tables"][0]["inherited_model_updates"] == base
    assert status["new_normal_updates"] == 120
    assert status["new_actual_updates"] == 150
    text = monitor.render_status(status)
    assert label in text and "Small-H4" not in text and "H4累计" not in text
    assert "本输出尚无完整评估" in text

    # DGX's launcher carries the same full trainer command but isn't another trainer.
    snapshot["processes"].append("122 1 cms 00:11 docker run --entrypoint python3 IMAGE -m tabu_lab.cli curriculum-v54 run --output-root /new/runs/replay")
    with_launcher = monitor.build_status(snapshot, decisions, checks=[dict(table="table", target_kind="numeric", target_column=1)],
                                         references={"table": {"metrics": dict(r2=0., nmse=1.)}})
    assert with_launcher["state"] == "running"
    assert len(with_launcher["active_training_processes"]) == 1
    assert len(with_launcher["matching_launcher_processes"]) == 1

    snapshot["processes"].append("124 1 cms 00:01 /usr/bin/python3.12 -m tabu_lab.cli curriculum-v54 run --output-root /new/runs/replay")
    duplicate = monitor.build_status(snapshot, decisions, checks=[], references={})
    assert duplicate["state"] == "needs_investigation"
    snapshot["processes"] = with_launcher["matching_launcher_processes"]
    launcher_only = monitor.build_status(snapshot, decisions, checks=[], references={})
    assert launcher_only["state"] == "needs_investigation"


@pytest.mark.parametrize("executable", ["python3", "/usr/bin/python3.12", "/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python"])
def test_training_interpreter_platforms(executable):
    assert monitor.is_training_interpreter(f"123 1 cms 00:10 {executable} -m tabu_lab.cli curriculum-v54 run")


def test_unconfigured_panel_remains_read_only_not_ready(monkeypatch):
    monkeypatch.setattr(monitor.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("unexpected SSH"))
    snapshot = monitor.read_remote({"state": "preparing"})
    assert snapshot["ready"] is False and "host" in snapshot["preparation_missing"]


def test_weighted_inherited_extra_not_counted_as_new_or_normal():
    policy = dict(kind="normal120_p99x3_p95x2_p80x1_v2", normal_max_updates=983040,
                  start_normal_cursor=120000, start_extra_updates=600)
    snapshot = dict(utc="2026-09-22T05:00:00+00:00", host="dgx2", ready=True,
                    identity_sha256="identity", source_sha256="source", data_sha256="data",
                    source_drift=[], output="/new/runs/replay", root="/new",
                    processes=["123 1 cms 00:10 python -m tabu_lab.cli curriculum-v54 run --output-root /new/runs/replay"],
                    terminal=None, journal_errors=[], all_logged_updates_finite=True,
                    update=120762, normal_update=120120, extra_updates=642,
                    normal_counts={"table": 1}, extra_counts={"table": 6}, curves=[],
                    normal_max_updates=983040, actual_max_updates=1300000, loss_replay=policy)
    decisions = dict(model_label="Standard Small H8", exposure_scope="old120 cumulative",
                     identity_sha256="identity", source_sha256="source", data_sha256="data",
                     initial_update=120600, initial_normal_update=120000, initial_extra_updates=600,
                     base_counts={"table": 1010}, base_extra_counts={"table": 10}, loss_replay=policy)
    status = monitor.build_status(snapshot, decisions,
                [dict(table="table", target_kind="numeric", target_column=1)],
                {"table": {"metrics": dict(r2=0., nmse=1.)}})
    assert status["state"] == "running"
    assert status["new_actual_updates"] == 162
    assert status["new_normal_updates"] == 120
    assert status["new_extra_updates"] == 42
    row = status["tables"][0]
    assert row["normal_new"] == 1 and row["extra_new"] == 6
    assert row["inherited_extra_updates"] == 10
    assert row["cumulative_extra_updates"] == 16
    assert row["cumulative_normal_updates"] == 1001
    assert row["cumulative_model_updates"] == 1017
    text = monitor.render_status(status)
    assert "加训累计 642（继承 600）" in text
    assert "本接续新增：正常 120，加训 42" in text
    assert "top2完整3遍、top6完整2遍、top24完整1遍" in text
    assert "最高2表各加训6次" in text
    bad = monitor.build_status(snapshot | {"extra_updates": 42}, decisions, [], {})
    assert bad["state"] == "needs_investigation"


def test_remote_journal_starts_at_inherited_extra_and_accepts_top2(tmp_path):
    root = tmp_path / "source"
    out = root / "runs" / "replay"
    out.mkdir(parents=True)
    names = [f"table_{i:03d}" for i in range(120)]
    policy = dict(kind="normal120_p99x3_p95x2_p80x1_v2", normal_max_updates=360,
                  start_normal_cursor=120, start_extra_updates=30)
    resolved = {"identity": {"sha256": "identity", "source": {"sha256": "source", "files": {}}, "data_sha256": "data"},
                "spec": {"stages": [{"loss_replay": policy, "max_updates": 474}],
                         "tables": [{"id": t, "role": "train"} for t in names]}}
    (out / "resolved.json").write_text(json.dumps(resolved))
    rows, total, normal = [], 150, 120
    groups = [("normal", names)] + [("extra_top2", names[-2:])] * 3 + [("extra_top6", names[-6:])] * 2 + [("extra_top24", names[-24:])]
    for kind, tables in groups:
        for table in tables:
            total += 1
            normal += int(kind == "normal")
            rows.append(dict(update=total, normal_update=normal, extra_updates=total-normal,
                             update_kind=kind, table=table, loss=1., gradient_norm=1., seconds=.1, total_seconds=total*.1))
    (out / "updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = dict(remote_root=str(root), remote_output=str(out), host="synthetic", initial_update=150,
                  initial_normal_update=120, initial_extra_updates=30, loss_replay=policy)
    script = "CONFIG = " + repr(config) + "\n" + monitor.REMOTE
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    assert data["journal_errors"] == []
    assert data["update"] == 312 and data["normal_update"] == 240
    assert data["initial_extra_updates"] == 30
    assert data["extra_updates"] == 72 and data["new_extra_updates"] == 42
    assert data["update_kind_counts"] == dict(normal=120, extra_top2=6, extra_top6=12, extra_top24=24)
    assert data["extra_counts"][names[-1]] == 6
    inherited_extra = {name: 2 if i >= 114 else 1 if i >= 96 else 0 for i, name in enumerate(names)}
    decisions = config | dict(model_label="Nano", exposure_scope="old120", identity_sha256="identity",
                             source_sha256="source", data_sha256="data",
                             base_counts={name: 1 + inherited_extra[name] for name in names},
                             base_extra_counts=inherited_extra)
    data["processes"] = [f"123 1 cms 00:10 python -m tabu_lab.cli curriculum-v54 run --output-root {out}"]
    checks = [dict(table=name, target_kind="numeric", target_column=1) for name in names]
    refs = {name: {"metrics": dict(r2=0., nmse=1.)} for name in names}
    status = monitor.build_status(data, decisions, checks, refs)
    assert status["state"] == "running" and status["inherited_exposure_matches"]
    bad = decisions | {"base_extra_counts": {name: 0 for name in names}}
    status = monitor.build_status(data, bad, checks, refs)
    assert status["state"] == "needs_investigation" and not status["inherited_exposure_matches"]
