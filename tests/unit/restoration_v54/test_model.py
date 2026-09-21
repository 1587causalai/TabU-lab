"""V5.4 presets, version boundaries, and shared-engine execution contracts."""

import copy
import json

import pytest
import torch
from torch import nn

from tabu_lab.models.restoration.end_to_end_checks import example_episode
from tabu_lab.models.restoration_v53 import V53Config, V53Model
from tabu_lab.models.restoration_v54 import (
    CODEC_VERSIONS,
    DEFAULT_CODEC_VERSION,
    BackboneConfig,
    RestorationInput,
    TruthSidecar,
    V54Config,
    V54LossConfig,
    V54Model,
    prepare_episode,
    score_episode,
    score_prepared_episode,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        yield
    torch.set_num_threads(threads)


def config(**overrides):
    return V54Config.from_dict({
        "backbone": {"layers": 1, "ff_width": 24, "slots": 4},
        "unit_layers": 1,
        "regression_width": 5,
        "center_chunk_size": 2,
        **overrides,
    })


@pytest.mark.parametrize(("size", "expected"), [
    ("nano", (2, 128, 4, 256, 0, 256)),
    ("small", (3, 128, 8, 256, 3, 256)),
    ("medium", (6, 192, 8, 384, 3, 256)),
    ("standard", (12, 256, 8, 512, 6, 256)),
    ("large", (24, 384, 12, 768, 6, 256)),
])
def test_named_sizes_resolve_consistently_and_roundtrip(size, expected):
    cfg = V54Config(size=size)
    backbone = cfg.backbone
    assert (backbone.layers, backbone.width, backbone.heads, backbone.ff_width,
            cfg.unit_layers, backbone.slots) == expected
    assert cfg == V54Config.from_size(size) == V54Config.from_dict({"size": size.upper()})
    assert cfg == V54Config.from_dict(json.loads(json.dumps(cfg.as_dict())))
    assert cfg.codec_version == DEFAULT_CODEC_VERSION == "constant_weight_composition_v1"
    assert cfg.center_chunk_size == 128
    assert cfg.numeric_scaling == "zscore" and cfg.slope_source == "shared_ll"
    assert cfg.subtokens == 1 and cfg.regression_width is None
    assert backbone.kind == "inducing" and backbone.tau_presence == 1
    assert cfg.as_dict()["size"] == size


def test_partial_overrides_are_explicit_and_do_not_relabel_small_as_nano():
    cfg = V54Config.from_dict({"unit_layers": 0, "backbone": {"slots": 7}})
    assert cfg.size == "small" and cfg.unit_layers == 0
    assert cfg.backbone == BackboneConfig(layers=3, heads=8, slots=7)
    assert cfg.as_dict()["backbone"]["slots"] == 7
    assert V54Config.from_dict(cfg.as_dict()) == cfg
    wider = V54Config.from_size("medium", backbone={"kind": "direct", "layers": 2})
    assert wider.backbone.width == 192 and wider.backbone.ff_width == 384
    assert wider.backbone.layers == 2 and wider.backbone.kind == "direct"
    explicit = BackboneConfig(layers=1, slots=3)
    assert V54Config(size="large", backbone=explicit, unit_layers=0).backbone is explicit


@pytest.mark.parametrize("subtokens", [0, 2, 3, 1.0, True])
def test_undefined_subtokens_are_rejected(subtokens):
    with pytest.raises(ValueError, match="supports only subtokens=1"):
        V54Config.from_dict({"subtokens": subtokens})


def test_old_defaults_are_unchanged_and_legacy_config_cannot_enter_v54():
    assert V53Config().codec_version == "constant_weight_v1"
    assert V53Config().unit_layers == 0
    assert V54Config() == V54Config.from_size("small")
    assert V54LossConfig().state_weights == (0.0, 1.0, 0.0, 0.0)
    with pytest.raises(TypeError, match="requires V54Config"):
        V54Model(V53Config())
    with pytest.raises(ValueError, match="composition codec identity"):
        V54Config.from_dict({"codec_version": "constant_weight_v1"})
    with pytest.raises(ValueError, match="size must be"):
        V54Config(size="tiny")


def test_default_model_uses_small_and_shared_trainable_projection():
    model = V54Model()
    assert len(model.backbone.layers) == 3 and len(model.unit_blocks) == 3
    assert isinstance(model.regression, nn.Identity)
    weight = model.encoder.projection.weight
    assert weight.shape == (128, 128) and weight.requires_grad
    assert model.encoder.projection.bias is None
    torch.testing.assert_close(weight.T @ weight, torch.eye(128) / 64, atol=1e-8, rtol=1e-5)
    assert model.state_dict()["_codec_signature"].tolist() == [4, 1]


@pytest.mark.parametrize("size", ["medium", "standard", "large"])
def test_wider_carriers_keep_128_dimensional_answers_and_trainable_lift(size):
    cfg = V54Config.from_size(size, backbone={"layers": 1, "ff_width": 24, "slots": 4},
                              unit_layers=0, regression_width=5)
    model = V54Model(cfg).double()
    score = score_episode(model, *example_episode())
    score.loss.backward()
    assert score.output.carriers.shape[-1] == cfg.backbone.width
    assert model.encoder.projection.weight.shape == (cfg.backbone.width, 128)
    assert all(column.result.encoding.shape[-1] == 128 for column in score.output.columns)
    assert torch.isfinite(model.encoder.projection.weight.grad).all()
    assert model.encoder.projection.weight.grad.norm() > 0


@pytest.mark.parametrize("codec_version", CODEC_VERSIONS)
@pytest.mark.parametrize("kind", ["direct", "inducing"])
@pytest.mark.parametrize("unit_layers", [0, 1])
def test_mixed_forward_backward_and_prepared_loss_agree(codec_version, kind, unit_layers):
    cfg = config(codec_version=codec_version, unit_layers=unit_layers,
                 backbone={"kind": kind, "layers": 1, "ff_width": 24, "slots": 4})
    model = V54Model(cfg).double()
    episode = example_episode(damage=True)
    score = score_episode(model, *episode)
    prepared = prepare_episode(model, *episode)
    cached = score_prepared_episode(model, prepared, decode=True)
    torch.testing.assert_close(cached.loss, score.loss, atol=0, rtol=0)
    torch.testing.assert_close(cached.per_target, score.per_target, atol=0, rtol=0)
    score.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gradients)
    assert model.encoder.projection.weight.grad.norm() > 0
    if unit_layers:
        assert any(p.grad is not None and p.grad.norm() > 0
                   for p in model.unit_blocks.parameters())
    else:
        assert torch.equal(score.output.units, score.output.carriers[:-1, -1])
    assert torch.count_nonzero(score.output.carriers[4, 0]) == 0
    feature_gradient = model.encoder.feature_seed.grad
    assert feature_gradient is None or torch.count_nonzero(feature_gradient) == 0
    column = score.output.columns[0]
    codec = score.output.facts[0].answers
    rows = episode[1].targets[column.target_indices, 0]
    expected = ((column.decoded - episode[2].values[0][rows]) / codec.scalar.scale).square()
    expected *= 4 if codec_version == "constant_weight_composition_v1" else 1
    torch.testing.assert_close(score.per_target[column.target_indices], expected)


@pytest.mark.parametrize("old_codec", [
    "legacy_v53", "unit_gaussian_v1", "unit_gaussian_v2", "constant_weight_v1",
])
def test_old_checkpoint_identity_cannot_load_into_v54_even_non_strict(old_codec):
    cfg = config()
    old = V53Model(V53Config(backbone=cfg.backbone, unit_layers=cfg.unit_layers,
                            regression_width=cfg.regression_width, codec_version=old_codec))
    model = V54Model(cfg)
    with pytest.raises(RuntimeError, match="codec identity does not match"):
        model.load_state_dict(old.state_dict(), strict=False)
    state = copy.deepcopy(model.state_dict())
    state.pop("_codec_signature")
    with pytest.raises(RuntimeError, match="lacks codec identity"):
        model.load_state_dict(state, strict=False)


def test_composition_modes_have_distinct_checkpoint_identity():
    sparse = V54Model(config())
    gaussian = V54Model(config(codec_version="unit_gaussian_composition_v1"))
    assert gaussian.state_dict()["_codec_signature"].tolist() == [5, 1]
    with pytest.raises(RuntimeError, match="codec identity does not match"):
        gaussian.load_state_dict(sparse.state_dict())


def test_prepared_replay_survives_updates_and_checkpoint_resume():
    model = V54Model(config()).double()
    episode = example_episode()
    prepared = prepare_episode(model, *episode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def update(net, opt, fixed):
        opt.zero_grad(set_to_none=True)
        score = score_prepared_episode(net, fixed)
        score.loss.backward()
        opt.step()
        return score.loss.detach()

    update(model, optimizer, prepared)
    clone = V54Model(V54Config.from_dict(json.loads(json.dumps(model.config.as_dict())))).double()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    other = torch.optim.AdamW(clone.parameters(), lr=1e-4)
    other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    cloned_episode = prepare_episode(clone, *episode)
    torch.testing.assert_close(update(model, optimizer, prepared),
                               update(clone, other, cloned_episode), atol=0, rtol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, clone.state_dict()[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="prepared model changed"):
        clone.forward_prepared(prepared.visible)
    prepared.visible.facts[0].answers.origin.add_(1)
    with pytest.raises(ValueError, match="mutated"):
        score_prepared_episode(model, prepared)


def test_empty_rows_remain_unit_receivers_and_unit_branch_never_writes_back():
    inputs, request, _ = example_episode()
    visible, query = inputs.visible.clone(), inputs.query.clone()
    visible[0] = False
    query[0] = True
    inputs = RestorationInput(inputs.schema, inputs.values, visible, query, inputs.code_seed)
    plain = V54Model(config(unit_layers=0)).double()
    refined = V54Model(config()).double()
    refined.encoder.load_state_dict(plain.encoder.state_dict())
    refined.backbone.load_state_dict(plain.backbone.state_dict())
    seen = []
    handle = refined.unit_blocks[0].register_forward_pre_hook(
        lambda _module, args: seen.append(args)
    )
    output = refined(inputs, request)
    handle.remove()
    receivers, sources, eligible = seen[0]
    assert len(receivers) == len(visible)
    assert torch.equal(eligible, visible.any(-1))
    assert not eligible[0] and not eligible[-1]
    without_ineligible_sources = refined.unit_blocks[0](
        receivers, sources[eligible], torch.ones(int(eligible.sum()), dtype=torch.bool)
    )
    torch.testing.assert_close(output.units, without_ineligible_sources)
    assert not torch.equal(output.units[0], receivers[0])
    torch.testing.assert_close(output.carriers, plain(inputs, request).carriers, atol=0, rtol=0)


def test_hidden_payload_isolated_and_inference_preparation_keeps_mutation_guard():
    inputs, request, truth = example_episode()
    model = V54Model(config()).double()
    expected = model(inputs, request)
    payload = tuple(v.clone() for v in inputs.values)
    payload[0][-1], payload[1][-1], payload[2][-1] = float("nan"), 999, 999
    changed = RestorationInput(
        inputs.schema, payload, inputs.visible, inputs.query, inputs.code_seed
    )
    with torch.inference_mode():
        prepared = model.prepare(changed, request)
        actual = model.forward_prepared(prepared)
        assert not torch.is_inference(prepared.facts[0].answers.encoded)
    for want, got in zip(expected.columns, actual.columns, strict=True):
        torch.testing.assert_close(want.result.encoding, got.result.encoding)
    values = (truth.values[0].clone().requires_grad_(), *truth.values[1:])
    score_episode(model, changed, request, TruthSidecar(values, truth.states)).loss.backward()
    assert values[0].grad is None
    prepared.facts[0].answers.origin.add_(1)
    with torch.inference_mode(), pytest.raises(ValueError, match="mutated"):
        model.forward_prepared(prepared)
