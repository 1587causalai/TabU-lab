"""Configuration, actual mask sampling, data isolation and strict continuation."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest
import torch
import yaml

from tabu_lab.models.restoration_v7 import (
    MaskingSpec,
    V7Config,
    V7Model,
    checkpoint_state,
    load_checkpoint,
    load_typed_table,
    make_optimizer,
    sample_task,
    save_checkpoint,
)
from tabu_lab.models.restoration_v7.fit import run_fit


@pytest.fixture(autouse=True)
def isolated_runtime(request):
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    if parameters.get("device") == "mps":
        torch.use_deterministic_algorithms(False)
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def fixture(tmp_path, mode="mixed"):
    table_path = tmp_path / "toy.json"
    table_path.write_text(
        json.dumps(
            dict(
                schema="tabu.tar.typed-fit-table.1",
                dataset="toy",
                column_names=["x", "z", "y"],
                features=[{"kind": "numeric"}] * 3,
                values=[[i / 10, (i % 5) / 5, (i / 10) ** 2] for i in range(20)],
                splits=dict(train=list(range(15)), test=list(range(15, 20))),
            )
        )
    )
    cfg = dict(
        schema="tabu.restoration.v7-fit.v1",
        tables=[dict(path="toy.json")],
        steps=4,
        seed=17,
        device="cpu",
        dtype="float64",
        window_rows=10,
        query_rows=3,
        output_dir="run",
        masking=dict(mode=mode, columns=[0, 1, "target"]),
        model=dict(
            backbone=dict(width=64, layers=1, heads=2, ff_width=64, slots=3),
            rounds=2,
            unit_layers=1,
            query_init="donor",
            query_source=True,
            coupling_blocks=1,
            coupling_hidden=[16],
            center_chunk_size=4,
        ),
    )
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg, load_typed_table(table_path)


def test_masking_counts_endpoints_determinism_and_no_test_exposure(tmp_path):
    _, _, table = fixture(tmp_path)
    spec = MaskingSpec(columns=(0, 1, "target"))
    modes = set()
    for step in range(20):
        task, receipt = sample_task(table, spec, seed=42, step=step, window_rows=10, query_rows=3)
        _, repeated = sample_task(table, spec, seed=42, step=step, window_rows=10, query_rows=3)
        assert receipt == repeated
        modes.add(receipt["mode"])
        assert set(receipt["rows"]) <= set(table.train_rows)
        assert not (set(receipt["rows"]) & set(table.test_rows))
        assert receipt["query_cells"] == int(task.inputs.query.sum())
        assert int(task.inputs.query.any(0).sum()) == (1 if receipt["mode"] == "single" else 3)
    assert modes == {"single", "joint"}
    for p, mode in ((0, "single"), (1, "joint")):
        _, receipt = sample_task(
            table, replace(spec, joint_probability=p), seed=42, step=0, window_rows=10, query_rows=3
        )
        assert receipt["mode"] == mode
    with pytest.raises(ValueError, match="distinct"):
        replace(spec, columns=(2, "target")).eligible(table)
    with pytest.raises(ValueError, match="not enough"):
        replace(spec, columns=("target",)).eligible(table)


@pytest.mark.parametrize("mode", ["single", "joint", "mixed"])
@pytest.mark.parametrize("loss_mode", ["query_only", "balanced_reconstruction"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_configured_training_resume_and_evaluation(tmp_path, mode, device, loss_mode):
    path, cfg, table = fixture(tmp_path, mode)
    cfg.update(device=device, dtype="float32" if device == "mps" else "float64")
    cfg["model"]["loss_mode"] = loss_mode
    path.write_text(yaml.safe_dump(cfg))
    plan = run_fit(path)
    assert plan["status"] == "validated-not-run"
    assert not (tmp_path / "run").exists()
    complete = run_fit(path, execute=True)
    updates = [
        json.loads(line) for line in (tmp_path / "run/updates.jsonl").read_text().splitlines()
    ]
    for update in updates:
        (plan,) = update["auxiliary_plans"]
        if loss_mode == "query_only":
            assert plan is None
        else:
            assert plan["sampling"] == "all_visible_per_column"
            for column in plan["columns"]:
                assert column["n_aux"] == len(column["rows"])
                assert column["w_bal"] == pytest.approx(
                    column["n_query"] / (column["n_query"] + column["n_aux"])
                )
    evaluation = json.loads((tmp_path / "run/evaluation.json").read_text())
    assert {report["mode"] for report in evaluation} == {"target_only", "joint"}
    for report in evaluation:
        assert set(report["query_rows"]) <= set(table.test_rows)
        assert set(report["support_rows"]) <= set(table.train_rows)
    half = tmp_path / "half.yaml"
    half.write_text(yaml.safe_dump(cfg | dict(steps=2, output_dir="half")))
    prefix = run_fit(half, execute=True)
    resumed = run_fit(
        path, execute=True, resume=prefix["checkpoint"], output_dir=tmp_path / "resumed"
    )
    a = torch.load(complete["checkpoint"], weights_only=True)
    b = torch.load(resumed["checkpoint"], weights_only=True)
    assert a["sampling_totals"] == b["sampling_totals"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    if mode != "mixed":
        assert complete["sampling_totals"]["query_cells"] == (12 if mode == "single" else 36)
    drift = tmp_path / "drift.yaml"
    drift.write_text(
        yaml.safe_dump(cfg | dict(masking=asdict(MaskingSpec(mode="single", columns=(0,)))))
    )
    with pytest.raises(ValueError, match="manifest"):
        run_fit(drift, execute=True, resume=prefix["checkpoint"], output_dir=tmp_path / "bad")


def test_config_rejects_silent_typo_and_mps_fp64(tmp_path):
    path, cfg, _ = fixture(tmp_path)
    path.write_text(yaml.safe_dump(cfg | dict(masking={"mod": "mixed"})))
    with pytest.raises(ValueError, match="unknown masking"):
        run_fit(path)
    path.write_text(yaml.safe_dump(cfg | dict(device="mps")))
    with pytest.raises(ValueError, match="MPS requires"):
        run_fit(path)


def test_weights_only_parent_inherits_shape_and_starts_fresh_optimizer(tmp_path):
    path, cfg, _ = fixture(tmp_path, "single")
    cfg["steps"] = 2
    path.write_text(yaml.safe_dump(cfg))
    parent = run_fit(path, execute=True)
    child = cfg | dict(
        steps=1,
        output_dir="child",
        init_checkpoint=parent["checkpoint"],
        model=dict(query_source=True, rounds=4),
        masking=dict(mode="mixed"),
    )
    path.write_text(yaml.safe_dump(child))
    plan = run_fit(path)
    assert plan["manifest"]["model"]["backbone"]["width"] == 64
    assert plan["manifest"]["model"]["rounds"] == 4
    result = run_fit(path, execute=True)
    state = torch.load(result["checkpoint"], weights_only=True)
    assert state["step"] == 1
    assert all(float(value["step"]) == 1 for value in state["optimizer"]["state"].values())
    assert state["manifest"]["initialization"]["sha256"] == parent["checkpoint_sha256"]


@pytest.mark.parametrize(
    "optimizer",
    [
        {"lr": -1},
        {"lr": float("nan")},
        {"lr": float("inf")},
        {"lr": "0.1"},
        {"eps": -1},
        {"eps": float("nan")},
        {"weight_decay": -1},
        {"betas": [0.9]},
        {"betas": [0.9, 1.0]},
        {"betas": [0.9, float("nan")]},
        {"betas": [True, 0.9]},
        {"lr": True},
    ],
)
def test_dry_run_rejects_invalid_optimizer_before_model_creation(tmp_path, optimizer):
    path, cfg, _ = fixture(tmp_path)
    path.write_text(yaml.safe_dump(cfg | {"optimizer": optimizer}))
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="optimizer"):
        run_fit(path)
    assert not (tmp_path / "run").exists()
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("scale", ["false", "true", 0, 1, None])
def test_dry_run_rejects_non_boolean_coupling_scale(tmp_path, scale):
    path, cfg, _ = fixture(tmp_path)
    cfg["model"]["coupling_scale"] = scale
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="coupling_scale must be a boolean"):
        run_fit(path)


def test_cyclic_factory_leaves_constructor_and_legacy_loading_unchanged():
    cyclic = V7Config.cyclic()
    assert (cyclic.query_init, cyclic.query_source, cyclic.rounds, cyclic.share_rounds) == (
        "donor",
        True,
        4,
        True,
    )
    assert cyclic.unit_source_policy == "observed"
    assert V7Config.from_dict(cyclic.as_dict()) == cyclic
    variant = V7Config.cyclic(
        rounds=8, share_rounds=False, unit_source_policy="legacy_cell_sources"
    )
    assert (variant.rounds, variant.share_rounds, variant.unit_source_policy) == (
        8,
        False,
        "legacy_cell_sources",
    )
    for old in (V7Config(), V7Config.from_dict({})):
        assert (old.query_init, old.query_source, old.rounds, old.share_rounds) == (
            "seed",
            False,
            1,
            True,
        )


def test_unit_source_policy_is_versioned_independently_of_query_source(tmp_path):
    current = V7Config.cyclic(unit_layers=1)
    old_values = current.as_dict()
    del old_values["model_version"]
    del old_values["unit_source_policy"]
    old = V7Config.from_dict(old_values)
    assert old.unit_source_policy == "legacy_cell_sources"
    assert old.query_source and old.rounds == 4
    assert V7Config.from_dict(old.as_dict()) == old
    assert (
        V7Config.from_dict(old_values | {"model_version": "v7.3", "unit_source_policy": "observed"})
        == current
    )
    path, _, _ = fixture(tmp_path)
    assert run_fit(path)["manifest"]["model"]["unit_source_policy"] == "legacy_cell_sources"


@pytest.mark.parametrize("policy", ["cell_sources", "", None, True])
def test_invalid_unit_source_policy_rejected_before_run(tmp_path, policy):
    path, cfg, _ = fixture(tmp_path)
    cfg["model"]["unit_source_policy"] = policy
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="unit_source_policy"):
        run_fit(path)


def test_explicit_model_versions_and_historical_field_preservation():
    legacy, modern = V7Config.legacy(), V7Config.v73()
    assert legacy == V7Config.for_version("v7") == V7Config.from_dict({})
    assert modern == V7Config.for_version("v7.3") == V7Config.cyclic()
    assert modern == V7Config.from_dict({"model_version": "v7.3"})
    assert (
        legacy.model_version,
        legacy.coupling_bias,
        legacy.numeric_preprocessing,
        legacy.unit_source_policy,
        legacy.query_init,
        legacy.query_source,
        legacy.rounds,
    ) == ("v7", True, "legacy", "legacy_cell_sources", "seed", False, 1)
    assert (
        modern.model_version,
        modern.coupling_bias,
        modern.numeric_preprocessing,
        modern.unit_source_policy,
        modern.unit_layers,
        modern.query_init,
        modern.query_source,
        modern.rounds,
        modern.share_rounds,
    ) == ("v7.3", False, "standard_asinh_v1", "observed", 0, "donor", True, 4, True)
    # The low-level constructor and old explicit checkpoint fields keep their meaning.
    plain = V7Config()
    assert plain == legacy
    assert V7Config(model_version="v7.3") == modern
    old_values = V7Config.cyclic(rounds=8, unit_layers=1).as_dict()
    old_values.pop("model_version")
    restored = V7Config.from_dict(old_values)
    assert restored.model_version == "v7"
    assert restored.as_dict() == old_values | {"model_version": "v7"}
    for config in (legacy, modern, plain, restored):
        assert V7Config.from_dict(config.as_dict()) == config


@pytest.mark.parametrize("version", ["v7.2", "V7", "", None, True])
def test_invalid_model_version_fails_dry_run(tmp_path, version):
    path, cfg, _ = fixture(tmp_path)
    cfg["model"]["model_version"] = version
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="model_version"):
        run_fit(path)


def test_version_switch_resets_semantics_but_keeps_parent_tensor_dimensions(tmp_path):
    path, cfg, _ = fixture(tmp_path)
    parent = V7Config.legacy(**cfg["model"], round_loss_rho=0.7)
    parent_path = tmp_path / "parent.pt"
    torch.save({"config": parent.as_dict(), "model": {}}, parent_path)
    cfg["init_checkpoint"] = str(parent_path)
    cfg["model"] = {}
    path.write_text(yaml.safe_dump(cfg))
    assert run_fit(path)["manifest"]["model"] == parent.as_dict()

    cfg["model"] = {"model_version": "v7.3", "rounds": 8}
    path.write_text(yaml.safe_dump(cfg))
    plan = run_fit(path)["manifest"]
    model = V7Config.from_dict(plan["model"])
    assert model.model_version == "v7.3" and model.rounds == 8
    assert model.backbone == parent.backbone and model.coupling_hidden == parent.coupling_hidden
    assert not model.coupling_bias and model.numeric_preprocessing == "standard_asinh_v1"
    assert model.unit_source_policy == "observed" and model.unit_layers == parent.unit_layers
    assert model.query_init == "donor" and model.query_source and model.share_rounds
    assert model.round_loss_rho == 0.7
    assert plan["initialization"]["source_model_version"] == "v7"
    assert plan["initialization"]["target_model_version"] == "v7.3"
    assert plan["initialization"]["kind"].startswith("weights-only")


@pytest.mark.parametrize("share_rounds", [True, False])
def test_cross_version_weights_only_keeps_compatible_parent_layout(tmp_path, share_rounds):
    path, cfg, _ = fixture(tmp_path)
    parent_config = V7Config.legacy(**cfg["model"], coupling_bias=False, share_rounds=share_rounds)
    parent = V7Model(parent_config).double()
    parent_path = tmp_path / "parent.pt"
    torch.save({"config": parent_config.as_dict(), "model": parent.state_dict()}, parent_path)
    cfg.update(init_checkpoint=str(parent_path), steps=1, model={"model_version": "v7.3"})
    path.write_text(yaml.safe_dump(cfg))
    plan = run_fit(path)["manifest"]
    child_config = V7Config.from_dict(plan["model"])
    assert child_config.unit_layers == parent_config.unit_layers == 1
    assert child_config.share_rounds == share_rounds
    assert child_config.rounds == (4 if share_rounds else parent_config.rounds)
    assert child_config.numeric_preprocessing == "standard_asinh_v1"
    assert child_config.unit_source_policy == "observed"
    migration = plan["initialization"]
    assert migration["configuration_changes"]["numeric_preprocessing"] == {
        "from": "legacy", "to": "standard_asinh_v1",
    }
    assert migration["weight_transfer_plan"]["inherit_state_keys"] == sorted(parent.state_dict())
    assert migration["weight_transfer_plan"]["reinitialize_state_keys"] == []
    assert migration["weight_transfer_plan"]["optimizer_rng_sampler"] == "fresh"
    result = run_fit(path, execute=True)
    saved = torch.load(result["checkpoint"], weights_only=True)
    assert saved["step"] == 1 and saved["model_version"] == "v7.3"
    assert set(saved["model"]) == set(parent.state_dict())
    assert all(float(value["step"]) == 1 for value in saved["optimizer"]["state"].values())


def test_cross_version_biased_parent_requires_explicit_weight_migration(tmp_path):
    path, cfg, _ = fixture(tmp_path)
    parent_config = V7Config.legacy(**cfg["model"])
    parent_path = tmp_path / "biased-parent.pt"
    torch.save(
        {"config": parent_config.as_dict(), "model": V7Model(parent_config).double().state_dict()},
        parent_path,
    )
    cfg.update(init_checkpoint=str(parent_path), steps=1, model={"model_version": "v7.3"})
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        run_fit(path, execute=True)
    assert not (tmp_path / "run").exists()


def test_checkpoint_version_identity_and_unversioned_compatibility(tmp_path):
    config = V7Config.v73(
        backbone=dict(width=64, layers=1, heads=2, ff_width=64, slots=3),
        coupling_blocks=1,
        coupling_hidden=(16,),
    )
    model = V7Model(config).double()
    state = checkpoint_state(model, make_optimizer(model), step=0, manifest={})
    assert state["model_version"] == state["config"]["model_version"] == "v7.3"
    path = tmp_path / "modern.pt"
    save_checkpoint(path, state)
    assert load_checkpoint(path, V7Model(config).double(), manifest={}) == 0
    with pytest.raises(ValueError, match="ModelSpec"):
        load_checkpoint(path, V7Model(replace(config, model_version="v7")).double(), manifest={})

    state["model_version"] = "v7"
    mismatched = tmp_path / "mismatched.pt"
    save_checkpoint(mismatched, state)
    with pytest.raises(ValueError, match="model_version"):
        load_checkpoint(mismatched, model, manifest={})

    del state["model_version"]
    del state["config"]["model_version"]
    old = tmp_path / "unversioned.pt"
    save_checkpoint(old, state)
    restored_config = V7Config.from_dict(state["config"])
    assert restored_config.model_version == "v7"
    assert load_checkpoint(old, V7Model(restored_config).double(), manifest={}) == 0


def test_v73_example_resolves_without_running():
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "examples/v7-restoration/v73.yaml"
    plan = run_fit(path)
    config = V7Config.from_dict(plan["manifest"]["model"])
    assert plan["status"] == "validated-not-run" and plan["steps"] == 8
    assert config.model_version == "v7.3" and config.unit_layers == 0
    assert config.query_source and config.rounds == 4 and not config.coupling_bias
