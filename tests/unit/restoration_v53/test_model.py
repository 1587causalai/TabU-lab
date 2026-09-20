import copy
import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from tabu_lab.models.restoration.end_to_end_checks import example_episode
from tabu_lab.models.restoration_v53 import (
    AffineNumericAnswers,
    BackboneConfig,
    ColumnSchema,
    FeatureSlopeProvider,
    RestorationInput,
    RestorationRequest,
    TruthSidecar,
    V53Config,
    V53LossConfig,
    V53Model,
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


def config(kind="inducing", unit_layers=0):
    return V53Config(backbone=BackboneConfig(kind=kind, layers=1, ff_width=24, slots=4),
                     unit_layers=unit_layers, regression_width=5, center_chunk_size=2)


def test_affine_codec_is_fixed_local_random_state_and_roundtrips():
    values = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    state = torch.get_rng_state().clone()
    codec = AffineNumericAnswers.from_visible(values, epsilon=1e-6, seed=7, key="a")
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(codec.origin.norm(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(codec.direction.norm(), torch.tensor(1.0, dtype=torch.float64))
    assert abs(codec.origin @ codec.direction) > 1e-4  # No orthogonalization.
    torch.testing.assert_close(codec.decode(codec.encoded), values)
    repeated = AffineNumericAnswers.from_visible(values.flip(0), epsilon=1e-6, seed=7, key="a")
    assert torch.equal(codec.origin, repeated.origin)
    assert torch.equal(codec.direction, repeated.direction)
    changed = AffineNumericAnswers.from_visible(values, epsilon=1e-6, seed=8, key="a")
    assert not torch.equal(codec.direction, changed.direction)
    unseen = torch.tensor([-200.0, 0.0, 100.0], dtype=torch.float64)
    torch.testing.assert_close(codec.decode(codec.encode_targets(unseen)), unseen)


@pytest.mark.parametrize("kind", ["direct", "inducing"])
@pytest.mark.parametrize("unit_layers", [0, 1])
def test_mixed_forward_backward_and_numeric_loss_scale(kind, unit_layers):
    episode = example_episode(damage=True)
    model = V53Model(config(kind, unit_layers)).double()
    score = score_episode(model, *episode)
    score.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.square().sum()) for g in gradients) > 0
    assert torch.count_nonzero(score.output.carriers[4, 0]) == 0
    feature = model.encoder.feature_seed.grad
    assert feature is None or torch.count_nonzero(feature) == 0
    column = score.output.columns[0]
    codec = score.output.facts[0].answers
    rows = episode[1].targets[column.target_indices, 0]
    expected_loss = ((column.decoded - episode[2].values[0][rows]) / codec.scalar.scale).square()
    torch.testing.assert_close(score.per_target[column.target_indices], expected_loss)
    # LL closes on the numeric affine line without a post-hoc projection.
    restored_line = codec.encode_targets(column.decoded)
    torch.testing.assert_close(column.result.encoding, restored_line, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(
        score.loss, score.per_target[column.target_indices].mean()
        + score.per_target[episode[1].targets[:, 1] != 0].mean()
    )


def test_default_full_width_and_unit_identity():
    model = V53Model(replace(config(), regression_width=None)).double()
    inputs, request, _ = example_episode()
    output = model(inputs, request)
    assert output.slopes[0].shape == (128, 128)
    assert isinstance(model.regression, nn.Identity)
    assert len(model.unit_blocks) == 0
    assert torch.equal(output.units, output.carriers[:-1, -1])
    assert V53Config().backbone.slots == 256
    assert V53Config().regression_width is None


def test_compiler_uses_same_affine_answers_and_declared_ordinal_order():
    inputs, _, _ = example_episode()
    schema = (*inputs.schema[:2], ColumnSchema("rank", "ordinal", 3, order=(2, 0, 1)))
    inputs = RestorationInput(schema, inputs.values, inputs.visible, inputs.query, inputs.code_seed)
    model = V53Model(config()).double()
    facts = model.encoder.prepare(inputs)
    initial = model.encoder(inputs, facts)
    for a, fact in enumerate(facts):
        lift = fact.answers.encoded
        if a == 2:
            rank_by_label = torch.tensor([0.5, 1.0, 0.0], dtype=torch.float64)
            ranks = rank_by_label[inputs.values[a][fact.rows]]
            torch.testing.assert_close(fact.rank, ranks)
            torch.testing.assert_close(
                lift, fact.answers.origin + ranks[:, None] * fact.answers.direction
            )
        torch.testing.assert_close(initial[fact.rows, a], model.encoder.projection(lift))


@pytest.mark.parametrize("unit_layers", [0, 1])
def test_request_subset_reordering_and_row_column_equivariance(unit_layers):
    inputs, request, _ = example_episode()
    model = V53Model(config(unit_layers=unit_layers)).double()
    baseline = model(inputs, request)
    one = model(inputs, RestorationRequest(torch.tensor([[5, 0]])))
    torch.testing.assert_close(one.columns[0].result.encoding,
                               baseline.columns[0].result.encoding[-1:])
    torch.testing.assert_close(one.slopes[0], baseline.slopes[0])
    rows, columns = torch.tensor([3, 5, 2, 1, 0, 4]), torch.tensor([2, 0, 1])
    changed = RestorationInput(
        tuple(inputs.schema[i] for i in columns), tuple(inputs.values[i][rows] for i in columns),
        inputs.visible[rows][:, columns], inputs.query[rows][:, columns], inputs.code_seed,
    )
    output = model(changed, RestorationRequest((changed.visible | changed.query).nonzero()))
    for a, old_a in enumerate(columns):
        torch.testing.assert_close(output.columns[a].result.encoding,
                                   baseline.columns[old_a].result.encoding[rows])
    shuffled = model(inputs, RestorationRequest(request.targets.flip(0)))
    for old, new in zip(baseline.columns, shuffled.columns, strict=True):
        torch.testing.assert_close(old.result.encoding.flip(0), new.result.encoding)


def test_add_query_in_existing_empty_address_preserves_old_outputs():
    inputs, request, _ = example_episode(damage=True)
    model = V53Model(config(unit_layers=1)).double()
    baseline = model(inputs, request)
    query = inputs.query.clone()
    query[4, 0] = True
    changed = RestorationInput(
        inputs.schema, inputs.values, inputs.visible, query, inputs.code_seed
    )
    output = model(changed, request)
    for old, new in zip(baseline.columns, output.columns, strict=True):
        keep = request.targets[old.target_indices, 0] != 4 if old.column == 0 else slice(None)
        torch.testing.assert_close(old.result.encoding[keep], new.result.encoding[keep])
    torch.testing.assert_close(baseline.units, output.units)


def test_unit_branch_uses_visible_row_mask_without_writing_back_cells():
    inputs, request, _ = example_episode()
    visible, query = inputs.visible.clone(), inputs.query.clone()
    visible[0] = False
    query[0] = True
    inputs = RestorationInput(inputs.schema, inputs.values, visible, query, inputs.code_seed)
    plain, refined = V53Model(config()).double(), V53Model(config(unit_layers=1)).double()
    refined.encoder.load_state_dict(plain.encoder.state_dict())
    refined.backbone.load_state_dict(plain.backbone.state_dict())
    seen = []
    handle = refined.unit_blocks[0].register_forward_pre_hook(
        lambda _module, args: seen.append(args)
    )
    output = refined(inputs, request)
    handle.remove()
    assert torch.equal(seen[0][2], visible.any(-1))
    assert not seen[0][2][0] and not seen[0][2][-1]
    torch.testing.assert_close(output.carriers, plain(inputs, request).carriers)


def test_hidden_payload_and_truth_cannot_enter_forward_or_receive_gradient():
    inputs, request, truth = example_episode()
    payload = tuple(v.clone() for v in inputs.values)
    payload[0][-1], payload[1][-1], payload[2][-1] = float("nan"), 999, 999
    sanitized = RestorationInput(
        inputs.schema, payload, inputs.visible, inputs.query, inputs.code_seed
    )
    model = V53Model(config()).double()
    baseline = model(inputs, request)
    changed = model(sanitized, request)
    for old, new in zip(baseline.columns, changed.columns, strict=True):
        torch.testing.assert_close(old.result.encoding, new.result.encoding)
    values = (truth.values[0].clone().requires_grad_(), *truth.values[1:])
    score_episode(model, sanitized, request, TruthSidecar(values, truth.states)).loss.backward()
    assert values[0].grad is None


def test_preflight_rejects_missing_codes_and_partial_training_before_network():
    inputs, request, truth = example_episode()
    model = V53Model(config()).double()
    with torch.no_grad():
        model.encoder.projection.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="all original observations"):
        score_episode(model, inputs, RestorationRequest(request.targets[:2]), truth)
    schema = (inputs.schema[0], ColumnSchema("category", "nominal", 4), inputs.schema[2])
    inputs = RestorationInput(schema, inputs.values, inputs.visible, inputs.query, inputs.code_seed)
    values = tuple(v.clone() for v in truth.values)
    values[1][-1] = 3
    with pytest.raises(ValueError, match="no-answer-code"):
        score_episode(model, inputs, request, TruthSidecar(values, truth.states))


def test_no_support_and_single_support_are_explicit_inference_cases():
    schema = (ColumnSchema("x", "numeric"),)
    values = (torch.tensor([7.0, 999.0, -999.0]),)
    request = RestorationRequest(torch.tensor([[1, 0], [2, 0]]))
    model = V53Model(config()).double()
    for count in (0, 1):
        visible = torch.zeros(3, 1, dtype=torch.bool)
        visible[:count] = True
        inputs = RestorationInput(schema, values, visible, ~visible)
        output = model(inputs, request).columns[0]
        if count == 0:
            assert output.result.status == "no-support" and output.decoded is None
        else:
            torch.testing.assert_close(output.decoded, torch.full((2,), 7.0, dtype=torch.float64))


def test_prepared_snapshot_mutation_guard_and_optimizer_checkpoint_continuation():
    model = V53Model(config()).double()
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
    config_copy = V53Config.from_dict(json.loads(json.dumps(model.config.as_dict())))
    clone = V53Model(config_copy).double()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    other = torch.optim.AdamW(clone.parameters(), lr=1e-4)
    other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    cloned_episode = prepare_episode(clone, *episode)
    torch.testing.assert_close(
        update(model, optimizer, prepared), update(clone, other, cloned_episode)
    )
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, clone.state_dict()[name], atol=0, rtol=0)
    with torch.no_grad():
        prepared.visible.facts[0].answers.origin.add_(1)
    with pytest.raises(ValueError, match="mutated"):
        score_prepared_episode(model, prepared)


def test_public_inference_mode_preserves_predictions_and_prepared_mutation_guard():
    model = V53Model(config(unit_layers=1)).double().eval()
    inputs, request, _ = example_episode()
    with torch.no_grad():
        expected = model(inputs, request)
    with torch.inference_mode():
        # Inputs may themselves originate inside the inference context.
        inference_inputs = RestorationInput(
            inputs.schema, inputs.values, inputs.visible, inputs.query, inputs.code_seed
        )
        inference_request = RestorationRequest(request.targets.clone())
        actual = model(inference_inputs, inference_request)
        prepared = model.prepare(inference_inputs, inference_request)
        reused = model.forward_prepared(prepared)
        assert torch.is_inference(actual.carriers)
        assert not torch.is_inference(prepared.inputs.visible)
        assert not torch.is_inference(prepared.facts[0].answers.encoded)
    for want, got, cached in zip(expected.columns, actual.columns, reused.columns, strict=True):
        torch.testing.assert_close(got.result.encoding, want.result.encoding)
        torch.testing.assert_close(cached.result.encoding, want.result.encoding)
    # Disabling inference mode for preparation must not weaken snapshot guards.
    prepared.facts[0].answers.origin.add_(1)
    with torch.inference_mode(), pytest.raises(ValueError, match="mutated"):
        model.forward_prepared(prepared)


class LinearFeatureSlope(FeatureSlopeProvider):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(128, 128 * 5, bias=False)

    def forward(self, feature):
        return self.projection(feature).reshape(128, 5)


def test_feature_slope_is_explicit_extension_with_gradient_and_no_default_parameters():
    assert V53Model(config()).feature_slope is None
    feature_config = replace(config(), slope_source="feature")
    with pytest.raises(ValueError, match="explicit"):
        V53Model(feature_config)
    with pytest.raises(ValueError, match="explicit"):
        V53Model(config(), feature_slope=LinearFeatureSlope())
    model = V53Model(feature_config, feature_slope=LinearFeatureSlope()).double()
    score = score_episode(model, *example_episode())
    score.loss.backward()
    assert model.feature_slope.projection.weight.grad.norm() > 0
    assert model.encoder.feature_seed.grad.norm() > 0


def test_type_and_state_weighting_is_explicit():
    model = V53Model(config()).double()
    episode = example_episode()
    weighted = score_episode(model, *episode, V53LossConfig(16, (0.05, 0.95, 0, 0)))
    targets = episode[1].targets
    states = episode[2].states[targets[:, 0], targets[:, 1]]
    expected = 0
    for numeric, weight in ((True, 1), (False, 16)):
        for state, state_weight in ((0, 0.05), (1, 0.95)):
            mask = ((targets[:, 1] == 0) == numeric) & (states == state)
            expected += weight * state_weight * weighted.per_target[mask].mean()
    torch.testing.assert_close(weighted.loss, expected)
