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
