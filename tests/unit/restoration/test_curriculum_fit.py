import torch

from tabu_lab.models.restoration import ColumnSchema, RestorationConfig, RestorationModel
from tabu_lab.models.restoration.backbone import BackboneConfig
from tabu_lab.models.restoration.encoding import EncoderConfig
from tabu_lab.restoration_curriculum_fit import (
    REAL,
    SYNTHETIC,
    CurriculumTable,
    _episode_for,
    _schedule,
    _test_episode,
    evaluate,
)
from tabu_lab.restoration_optimizers import (
    OptimizerConfig,
    adamw,
    hidden_linear_weights,
    switch_to_muon,
)


def _table(name, cohort, kind=SYNTHETIC, rows=8, *, windowed=False):
    return CurriculumTable(
        name, cohort, kind, (torch.arange(rows, dtype=torch.float64),),
        # The target is numeric; this keeps the fixture independent of a corpus file.
        (ColumnSchema(f"{name}/target", "numeric"),), tuple(range(rows)), 2, windowed,
    )


def _stage(name, updates, mask="random_cell"):
    return {"name": name, "max_updates": updates, "mask": mask,
            "mask_fraction": .25, "synthetic_mask_fraction": .25,
            "real_mask_fraction": .5}


def test_stage_schedules_have_the_declared_cycle_shapes():
    old = [_table(f"old_{i}", "old120") for i in range(120)]
    recent = [_table(f"new_{i}", "recent120") for i in range(120)]
    assert len(list(_schedule(_stage("old120", 120), old, {"order": 3}))) == 120
    assert len(list(_schedule(_stage("recent120_replay", 150), old + recent, {"order": 3}))) == 150


def test_stage_three_schedule_reports_1200_step_cycles():
    synthetic = [_table(f"syn_{i}", "old120") for i in range(240)]
    real = [_table(f"real_{i}", "old3", REAL) for i in range(12)]
    entries = list(_schedule(_stage("openml12_mixed", 2400, "supervised_row"),
                             synthetic + real, {"order": 3}))
    assert len(entries) == 2400
    assert entries[0][1:] == (0, 0)
    assert entries[1199][1:] == (0, 1199)
    assert entries[1200][1:] == (1, 0)


def test_supervised_row_episode_queries_only_final_column():
    table = _table("real", "old3", REAL, rows=8)
    stage = _stage("openml12_mixed", 15, "supervised_row")
    config = RestorationConfig(encoder=EncoderConfig(width=128), backbone=BackboneConfig(
        width=128, layers=1, heads=2, ff_width=16, slots=2,
    ))
    episode, info = _episode_for(
        table, 0, stage, {"masks": 4, "windows": 5}, config, "cpu", evaluation=True,
    )
    target_states = episode[2].states[episode[1].targets[:, 0], episode[1].targets[:, 1]]
    query = episode[1].targets[target_states == 1]
    assert set(query[:, 1].tolist()) == {0}
    assert info["tail_guard_enabled"] is False


def test_muon_transition_covers_backbone_weights_without_overlap():
    config = RestorationConfig(encoder=EncoderConfig(width=128), backbone=BackboneConfig(
        width=128, layers=1, heads=2, ff_width=16, slots=2,
    ))
    model = RestorationModel(config).double()
    cfg = OptimizerConfig(1e-4, .01, (.9, .95), 1e-8, 1.0)
    optimizer = adamw(model, cfg)
    selected = hidden_linear_weights(model)
    mixed, partition = switch_to_muon(optimizer, model, cfg)
    assert partition["muon"] == list(selected)
    assigned = [id(parameter) for group in mixed.param_groups for parameter in group["params"]]
    assert len(assigned) == len(set(assigned)) == sum(1 for _ in model.parameters())


def _table_with_reserved(name, cohort, kind=SYNTHETIC, rows=8, reserved=3, windowed=False):
    return CurriculumTable(
        name, cohort, kind,
        (torch.arange(rows, dtype=torch.float64),),
        (ColumnSchema(f"{name}/target", "numeric"),),
        tuple(range(rows)), reserved, windowed,
        (torch.arange(rows, rows + reserved, dtype=torch.float64) + 0.5,),
        tuple(range(rows, rows + reserved)),
    )


def _small_config():
    return RestorationConfig(encoder=EncoderConfig(width=128), backbone=BackboneConfig(
        width=128, layers=1, heads=2, ff_width=16, slots=2,
    ))


def test_reserved_test_episode_queries_only_reserved_target_rows():
    table = _table_with_reserved("syn", "old120")
    episode, total = _test_episode(table, {"windows": 5, "codes": 6}, "old120", "cpu", 256)
    inputs, _, truth = episode
    assert total == 3
    assert inputs.visible.shape == (11, 1)
    assert int(inputs.query.sum()) == 3
    assert inputs.query[8:, 0].all() and not inputs.query[:8].any()
    assert int((truth.states == 1).sum()) == 3 and int((truth.states == 0).sum()) == 8
    again, _ = _test_episode(table, {"windows": 5, "codes": 6}, "old120", "cpu", 256)
    assert torch.equal(again[0].query, inputs.query)
    assert torch.equal(again[0].values[0], inputs.values[0])
    # Training episodes keep the 8-row training panel; reserved rows never enter.
    stage = _stage("old120", 120)
    train_episode, _ = _episode_for(
        table, 0, stage, {"masks": 4, "windows": 5}, _small_config(), "cpu"
    )
    assert train_episode[0].visible.shape == (8, 1)


def test_reserved_test_episode_caps_rows_and_windows_large_tables():
    table = _table_with_reserved("big", "new9", rows=8, reserved=100, windowed=True)
    episode, total = _test_episode(table, {"windows": 5, "codes": 6}, "openml12_mixed", "cpu", 16)
    inputs, _, _ = episode
    assert total == 100
    assert int(inputs.query.sum()) == 16  # capped by reserved_test_max_rows
    assert inputs.visible.shape == (8 + 16, 1)  # window-0 context (8) + subset (16)
    again, _ = _test_episode(table, {"windows": 5, "codes": 6}, "openml12_mixed", "cpu", 16)
    assert torch.equal(again[0].query, inputs.query)
    other, _ = _test_episode(table, {"windows": 5, "codes": 7}, "openml12_mixed", "cpu", 16)
    # Truth sidecar keeps the original values; the codes seed picks a different subset.
    assert not torch.equal(other[2].values[0][8:], episode[2].values[0][8:])


def test_evaluate_adds_reserved_test_block_only_when_enabled():
    tables = [
        _table_with_reserved("syn_a", "old120"),
        _table_with_reserved("real_b", "old3", REAL),
    ]
    config = _small_config()
    model = RestorationModel(config).double()
    stage = _stage("old120", 120)
    seeds = {"masks": 4, "windows": 5, "codes": 6}
    plain = evaluate(model, tables, stage, seeds, config, "cpu", masks=2)
    assert "test" not in plain
    assert plain["scope"].endswith("reserved rows excluded")
    report = evaluate(model, tables, stage, seeds, config, "cpu", masks=2, reserved_test=64)
    assert "forward-only" in report["scope"]
    test = report["test"]
    assert test["complete"] is True
    assert test["coverage"]["tables"] == 2
    assert test["coverage"]["query_cells"] == 6
    assert test["coverage"]["reserved_rows"] == 6
    assert test["by_state"]["query"]["count"] == 6
    assert test["by_state"]["query"]["encoding_mse"] is not None
    assert set(test["by_table"]) == {"syn_a", "real_b"}
    assert "forward-only" in test["scope"]
    # The fixed-bank evaluation is unaffected by the extra test pass.
    assert report["by_state"]["query"]["count"] == plain["by_state"]["query"]["count"]
    assert report["by_state"]["query"]["encoding_mse"] == plain["by_state"]["query"]["encoding_mse"]
