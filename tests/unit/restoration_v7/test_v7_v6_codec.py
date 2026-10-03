import pytest
import torch
from test_v7_c64 import table

from tabu_lab.models.restoration.backbone import BackboneConfig
from tabu_lab.models.restoration_v7 import V7Config, V7Model, prepare_episode
from tabu_lab.models.restoration_v7.codec import build_value_codec
from tabu_lab.models.restoration_v7.training import reference_values, score_rounds
from tabu_lab.models.restoration_v7.warm_start import from_v6_checkpoint
from tabu_lab.models.restoration_v53.encoding import AffineValueEncoder
from tabu_lab.models.restoration_v55 import V55Config, V55Model


def test_legacy_codec_exact_codes_and_decoder():
    inputs, _, _ = table()
    legacy = AffineValueEncoder(128, codec_version="constant_weight_composition_v2")
    facts = legacy.prepare(inputs)
    codec = build_value_codec(inputs, codec="constant_weight_composition_v2", dim=128)
    for a, (column, fact) in enumerate(zip(codec.columns, facts, strict=True)):
        values = inputs.values[a][fact.rows]
        torch.testing.assert_close(column.encode(values), fact.answers.encoded, rtol=0, atol=0)
        arbitrary = fact.answers.encoded + torch.randn_like(fact.answers.encoded) * 0.3
        torch.testing.assert_close(
            column.decode(arbitrary), fact.answers.decode(arbitrary), rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="128"):
        V7Config(codec="constant_weight_composition_v2")


@pytest.mark.parametrize("value_map", ["identity", "coupling"])
def test_inheritance_forward_gradients_and_reload(value_map):
    cfg = V55Config(
        backbone=BackboneConfig(
            width=128, layers=1, heads=4, ff_width=128, kind="inducing", slots=4
        ),
        unit_layers=1,
        codec_version="constant_weight_composition_v2",
    )
    parent_model = V55Model(cfg).double()
    parent = {"model_config": cfg.as_dict(), "model": parent_model.state_dict()}
    model = from_v6_checkpoint(parent, value_map=value_map)
    assert model.transfer_receipt["query_seed_exact_within_1e_5"]
    torch.testing.assert_close(
        model.rounds[0].lift.weight, parent_model.encoder.projection.weight, rtol=0, atol=0
    )
    inputs, _, truth = table()
    episode = prepare_episode(inputs, donor_seed=2, code_dim=128, codec=model.config.codec)
    output = model(episode)
    score = score_rounds(output, episode, reference_values(episode, truth), model.config)
    score.loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
    restored = V7Model(V7Config.from_dict(model.config.as_dict())).double()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(episode).states[0], output.states[0], rtol=0, atol=0)


@pytest.mark.parametrize("target", [1, 2, 3])
def test_original_codec_training_and_evaluation_all_kinds(target):
    from tabu_lab.models.restoration.contracts import make_episode
    from tabu_lab.models.restoration_v7 import V7Task, evaluate_task, make_optimizer, train_step

    inputs, _, truth = table()
    query = torch.zeros_like(inputs.query)
    query[-2:, target] = True
    inputs, _, truth = make_episode(
        inputs.schema, truth.values, torch.ones_like(query), query, code_seed=3
    )
    model = V7Model(
        V7Config(
            codec="constant_weight_composition_v2",
            code_dim=128,
            backbone=dict(width=128, layers=1, heads=4, ff_width=128, slots=4),
        )
    ).double()
    task = V7Task(inputs, truth, 2)
    train_step(model, make_optimizer(model), [task])
    report = evaluate_task(model, task)
    assert report.status == "ok"
    assert torch.isfinite(report.predictions[-1]).all()


def test_exact_preimage_does_not_claim_training_stability():
    cfg = V55Config(
        backbone=BackboneConfig(
            width=128, layers=1, heads=4, ff_width=128, kind="inducing", slots=4
        ),
        unit_layers=1,
        codec_version="constant_weight_composition_v2",
    )
    parent_model = V55Model(cfg).double()
    with torch.no_grad():
        parent_model.encoder.projection.weight.copy_(torch.eye(128))
        parent_model.encoder.projection.weight[0, 0] = 1e-6
        parent_model.encoder.cell_seed.zero_()
        parent_model.encoder.cell_seed[0] = 1
    parent = {"model_config": cfg.as_dict(), "model": parent_model.state_dict()}
    with pytest.warns(RuntimeWarning, match="strongly amplified"):
        model = from_v6_checkpoint(parent, value_map="identity")
    receipt = model.transfer_receipt
    assert receipt["query_seed_exact_within_1e_5"]
    assert receipt["query_seed_norm_amplification"] == pytest.approx(1e6)
    assert receipt["amplified_query_preimage"]
    assert receipt["training_stability_qualified"] is False
