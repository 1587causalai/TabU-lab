import copy
import json
import math

import pytest
import torch

from tabu_lab.models.restoration import (
    ColumnSchema,
    EncoderConfig,
    LossConfig,
    RestorationInput,
    RestorationModel,
    RestorationRequest,
    TruthSidecar,
    batch_loss,
    make_episode,
    score_episode,
)
from tabu_lab.models.restoration.backbone import OMAB, BackboneConfig
from tabu_lab.models.restoration.encoding import ValueEncoder
from tabu_lab.models.restoration.end_to_end_checks import (
    check_model_variants,
    check_optimizer_continuation,
    example_episode,
    small_config,
)
from tabu_lab.models.restoration.training import _masked_mean


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        yield
    torch.set_num_threads(threads)


def test_all_sixteen_variants_and_gradients():
    assert len(check_model_variants()) == 16


def test_checkpoint_optimizer_exact_continuation():
    assert check_optimizer_continuation()["model_optimizer_continuation"] == "exact"


@pytest.mark.parametrize("kind", ["direct", "inducing"])
def test_query_null_insertion_projectivity(kind):
    inputs, request, _ = example_episode()
    model = RestorationModel(small_config(kind)).double()
    baseline = model(inputs, request)
    for is_query in (False, True):
        visible = torch.cat((inputs.visible, torch.zeros(2, 3, dtype=torch.bool)))
        query = torch.cat((inputs.query, torch.full((2, 3), is_query)))
        values = tuple(torch.cat((v, v.new_zeros(2))) for v in inputs.values)
        extended = RestorationInput(inputs.schema, values, visible, query, inputs.code_seed)
        output = model(extended, request)
        torch.testing.assert_close(
            output.carriers[:6], baseline.carriers[:6], rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            output.carriers[-1], baseline.carriers[-1], rtol=1e-12, atol=1e-12
        )
        for old, new in zip(baseline.columns, output.columns, strict=True):
            torch.testing.assert_close(
                old.result.encoding, new.result.encoding, rtol=1e-9, atol=1e-10
            )


@pytest.mark.parametrize("kind", ["direct", "inducing"])
def test_row_column_permutation_and_request_independence(kind):
    inputs, request, _ = example_episode()
    model = RestorationModel(small_config(kind, "rotary32")).double()
    baseline = model(inputs, request)
    rows = torch.tensor([3, 5, 2, 1, 0, 4])
    cols = torch.tensor([2, 0, 1])
    changed = RestorationInput(
        tuple(inputs.schema[i] for i in cols),
        tuple(inputs.values[i][rows] for i in cols),
        inputs.visible[rows][:, cols],
        inputs.query[rows][:, cols],
        inputs.code_seed,
    )
    output = model(changed, RestorationRequest(torch.tensor([[0, 0]])))
    expected = baseline.carriers[torch.cat((rows, torch.tensor([6])))][
        :, torch.cat((cols, torch.tensor([3])))
    ]
    torch.testing.assert_close(output.carriers, expected, rtol=1e-11, atol=1e-11)
    one = model(inputs, RestorationRequest(request.targets[[2]]))
    torch.testing.assert_close(
        one.columns[0].result.encoding, baseline.columns[2].result.encoding[:1]
    )


def test_hidden_truth_never_retained_or_used_in_forward():
    inputs, request, truth = example_episode()
    payload = tuple(v.clone() for v in inputs.values)
    payload[0][-1] = float("nan")
    payload[1][-1] = 999
    sanitized = RestorationInput(
        inputs.schema, payload, inputs.visible, inputs.query, inputs.code_seed
    )
    assert all(torch.equal(x, y) for x, y in zip(inputs.values, sanitized.values, strict=True))
    model = RestorationModel(small_config()).double()
    other_truth = list(truth.values)
    other_truth[0] = other_truth[0].clone().requires_grad_()
    changed = score_episode(
        model, sanitized, request, TruthSidecar(tuple(other_truth), truth.states)
    )
    changed.loss.backward()
    assert other_truth[0].grad is None


def test_preflight_rejects_whole_episode_before_neural_forward():
    inputs, request, truth = example_episode()
    model = RestorationModel(small_config()).double()
    with torch.no_grad():
        model.encoder.projection.weight.fill_(float("nan"))
    hidden = tuple(v.clone() for v in truth.values)
    hidden[1][-1] = 3
    # Schema has a declared but invisible class; codebook must not learn it from truth.
    schema = (inputs.schema[0], ColumnSchema("category", "nominal", 4), inputs.schema[2])
    inputs = RestorationInput(schema, inputs.values, inputs.visible, inputs.query, inputs.code_seed)
    with pytest.raises(ValueError, match="no-answer-code"):
        score_episode(model, inputs, request, TruthSidecar(hidden, truth.states))


def test_training_cannot_drop_unscorable_query_but_inference_may_request_subset():
    episode = make_episode(
        (ColumnSchema("label", "nominal", 2),),
        (torch.tensor([0, 0, 1]),),
        torch.ones(3, 1, dtype=torch.bool),
        torch.tensor([[False], [False], [True]]),
        code_seed=1,
    )
    inputs, request, truth = episode
    model = RestorationModel(small_config()).double()
    with pytest.raises(ValueError, match="no-answer-code"):
        score_episode(model, *episode)
    subset = RestorationRequest(request.targets[:2])
    with pytest.raises(ValueError, match="all original observations"):
        score_episode(model, inputs, subset, truth)
    assert model(inputs, subset).columns[0].result.status == "ok"


def test_damage_uses_actual_corrupted_support_stats_and_original_truth():
    clean = example_episode()
    damage = example_episode(damage=True)
    model = RestorationModel(small_config()).double()
    score = score_episode(model, *damage)
    fact = score.output.facts[0]
    # Supports are the corrupted 9,1,2,3 (clean 0,1,2,3 would give median 1.5).
    assert fact.answers.median == 2.5
    assert fact.answers.scale == 1.375  # half-IQR of 1,2,3,9
    assert damage[2].values[0][0] == 0
    assert clean[0].values[0][0] == 0
    assert score.by_state["corrupted"]["count"] == 1
    assert score.by_state["query"]["count"] == 3
    assert score.by_state["null"]["count"] == 1
    states = damage[2].states[damage[1].targets[:, 0], damage[1].targets[:, 1]]
    numeric = damage[1].targets[:, 1] == 0
    expected = score.per_target[numeric].mean() + score.per_target[~numeric].mean()
    torch.testing.assert_close(score.loss, expected)
    weighted = score_episode(model, *damage, loss_config=LossConfig((0, 1, 2, 3)))
    expected = sum(
        weight * score.per_target[branch & (states == state)].mean()
        for branch in (numeric, ~numeric)
        for state, weight in enumerate((0, 1, 2, 3))
        if bool((branch & (states == state)).any())
    )
    torch.testing.assert_close(weighted.loss, expected)
    combined, _ = batch_loss(model, [damage, clean])
    torch.testing.assert_close(combined, (score.loss + score_episode(model, *clean).loss) / 2)


def test_zero_one_support_and_training_support_preflight():
    schema = (ColumnSchema("x", "numeric"),)
    visible = torch.tensor([[False], [False]])
    query = ~visible
    inputs = RestorationInput(schema, (torch.zeros(2),), visible, query)
    request = RestorationRequest(torch.tensor([[1, 0]]))
    model = RestorationModel(small_config()).double()
    assert model(inputs, request).columns[0].result.status == "no-support"
    inputs = RestorationInput(
        schema,
        (torch.tensor([7.0, 0.0]),),
        torch.tensor([[True], [False]]),
        torch.tensor([[False], [True]]),
    )
    assert model(inputs, request).columns[0].decoded.item() == 7
    truth = TruthSidecar((torch.tensor([7.0, 8.0]),), torch.tensor([[0], [1]]))
    with pytest.raises(ValueError, match="two visible supports"):
        score_episode(model, inputs, RestorationRequest(torch.tensor([[0, 0], [1, 0]])), truth)


def test_episode_construction_enforces_two_visible_supports():
    """n_a >= 2 is an episode well-formedness condition, checked at construction."""
    schema = (ColumnSchema("x", "numeric"),)
    values = (torch.tensor([1.0, 2.0, 3.0]),)
    observed = torch.ones(3, 1, dtype=torch.bool)
    # Query-masking two of three observations leaves n_a = 1: reject here.
    with pytest.raises(ValueError, match="no-valid-episode"):
        make_episode(schema, values, observed, torch.tensor([[True], [True], [False]]), code_seed=0)
    # Null damage has the same effect on the support count.
    with pytest.raises(ValueError, match="no-valid-episode"):
        make_episode(
            schema,
            values,
            observed,
            torch.tensor([[True], [False], [False]]),
            code_seed=0,
            null=torch.tensor([[False], [True], [False]]),
        )
    # A legal masking constructs fine; an unobserved untargeted column stays empty.
    inputs, _, _ = make_episode(
        schema, values, observed, torch.tensor([[True], [False], [False]]), code_seed=0
    )
    assert int(inputs.visible.sum()) == 2
    wider = (*schema, ColumnSchema("aux", "numeric"))
    empty_column = torch.zeros(3, 1, dtype=torch.bool)
    inputs, _, _ = make_episode(
        wider,
        (torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3)),
        torch.cat((observed, empty_column), dim=1),
        torch.tensor([[True, False], [False, False], [False, False]]),
        code_seed=0,
    )
    assert not bool(inputs.visible[:, 1].any())


def test_numeric_input_coordinate_is_the_answer_encoding():
    """v2: one shared robust coordinate for input, answer, score, and decode."""
    inputs, _, _ = example_episode()
    model = RestorationModel(small_config()).double()
    fact = model.encoder.prepare(inputs)[0]
    assert float(fact.answers.median) == 2.0 and float(fact.answers.scale) == 1.0
    torch.testing.assert_close(fact.input_coordinates, fact.answers.encoded, rtol=0, atol=0)


def test_visible_codes_stable_under_global_rng_and_row_order():
    inputs, _, _ = example_episode()
    encoder = ValueEncoder(EncoderConfig(category_map="rotary32"))
    before = torch.get_rng_state().clone()
    a = encoder.prepare(inputs)
    assert torch.equal(before, torch.get_rng_state())
    torch.manual_seed(999)
    b = encoder.prepare(inputs)
    assert torch.equal(a[1].answers.codebook, b[1].answers.codebook)
    changed = RestorationInput(inputs.schema, inputs.values, inputs.visible, inputs.query, 18)
    assert not torch.equal(a[1].answers.codebook, encoder.prepare(changed)[1].answers.codebook)
    assert a[1].answers.encoded.shape[-1] == 32
    assert a[2].answers.encoded.shape[-1] == 128  # ordinal default retained explicitly


def test_ordinal_rank_follows_declared_order_not_label_index():
    """rank_a(x) counts categories below x under the DECLARED order, not labels."""
    schema = (ColumnSchema("rank", "ordinal", 3, order=(2, 0, 1)),)
    values = (torch.tensor([0, 1, 2], dtype=torch.long),)
    visible = torch.ones(3, 1, dtype=torch.bool)
    query = torch.zeros(3, 1, dtype=torch.bool)
    fact = ValueEncoder(EncoderConfig()).prepare(
        RestorationInput(schema, values, visible, query, 0)
    )[0]
    # order[0]=2 declares label 2 lowest: label 2 -> rank 0, 0 -> 1/2, 1 -> 1.
    torch.testing.assert_close(
        fact.rank, torch.tensor([0.5, 1.0, 0.0], dtype=torch.float64), rtol=0, atol=0
    )
    identity = RestorationInput((ColumnSchema("rank", "ordinal", 3),), values, visible, query, 0)
    torch.testing.assert_close(
        ValueEncoder(EncoderConfig()).prepare(identity)[0].rank,
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_ordinal_order_schema_validation():
    with pytest.raises(ValueError, match="permute"):
        ColumnSchema("bad", "ordinal", 3, order=(0, 0, 1))
    with pytest.raises(ValueError, match="order"):
        ColumnSchema("bad", "nominal", 3, order=(0, 1, 2))
    with pytest.raises(ValueError, match="order"):
        ColumnSchema("bad", "numeric", order=(0,))
    with pytest.raises(ValueError, match="only int"):
        ColumnSchema("bad", "ordinal", 3, order=(0.0, 1, 2))


def test_restoration_input_owns_schema_container():
    inputs, _, _ = example_episode()
    schema = list(inputs.schema)
    owned = RestorationInput(schema, inputs.values, inputs.visible, inputs.query, inputs.code_seed)
    schema[0] = ColumnSchema("changed", "numeric")
    assert owned.schema[0] == inputs.schema[0]


def test_masked_mean_scales_before_sum():
    values = torch.tensor([1.21e308, 1.21e308], dtype=torch.float64)
    mean = _masked_mean(values, torch.ones(2, dtype=torch.bool))
    assert torch.isfinite(mean)
    torch.testing.assert_close(mean, torch.tensor(1.21e308, dtype=torch.float64))


def test_rotary_is_four_isometries_and_input_projection_init():
    encoder = ValueEncoder(EncoderConfig(width=160, category_map="rotary32")).double()
    t = encoder.category_features(torch.eye(32, dtype=torch.float64)).T
    torch.testing.assert_close(
        t.T @ t, 4 * torch.eye(32, dtype=torch.float64), atol=1e-12, rtol=1e-12
    )
    w = encoder.projection.weight
    torch.testing.assert_close(w.T @ w, torch.eye(128, dtype=w.dtype) / 64, atol=1e-8, rtol=1e-5)


def test_omab_reference_mass_ineligible_nan_and_local_empty_update():
    config = BackboneConfig(width=4, heads=1, ff_width=4, reference_mass=2)
    op = OMAB(config).double()
    with torch.no_grad():
        op.q.weight.zero_()
        op.k.weight.zero_()
        op.v.weight.copy_(torch.eye(4))
        op.out.weight.copy_(torch.eye(4))
        op.ff[0].weight.zero_()
    receiver = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)
    sources = torch.tensor([[1.0, 0, 0, 0], [float("nan"), 0, 0, 0]], dtype=torch.float64)
    output = op(receiver, sources, torch.tensor([True, False]))
    torch.testing.assert_close(output, receiver * 1.1)  # pR=.5,pS=.5 => .5*.5/(2+.5)
    torch.testing.assert_close(
        op(torch.zeros_like(receiver), sources, torch.tensor([True, False])),
        torch.zeros_like(receiver),
    )
    op = OMAB(config).double()
    assert not torch.equal(op(receiver, sources, torch.tensor([False, False])), receiver)


def test_tiny_nonzero_presence_survives_content_logits_and_matches_fp64():
    op = OMAB(BackboneConfig(width=128, heads=1)).float()
    with torch.no_grad():
        op.attention_presence.weight[0, 0] = 1e-23
        op.q.weight.zero_()
        op.k.weight.zero_()
        op.q.weight[0, 0] = math.sqrt(120 * math.sqrt(128))
        op.k.weight[0, 0] = math.sqrt(120 * math.sqrt(128))
        op.v.weight.copy_(torch.eye(128))
        op.out.weight.copy_(torch.eye(128))
        op.ff[0].weight.zero_()
    source = torch.zeros(1, 128)
    source[0, 0] = 1
    receiver = source.clone()
    receiver[0, 1] = 1
    eligible = torch.ones(1, dtype=torch.bool)
    actual = op(receiver, source, eligible)
    reference = copy.deepcopy(op).double()(receiver.double(), source.double(), eligible)
    assert actual[0, 0] > 1.49
    torch.testing.assert_close(actual.double(), reference, atol=1e-6, rtol=1e-6)
    actual.sum().backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in op.parameters() if p.grad is not None)
    assert torch.isneginf(op.log_presence(torch.zeros_like(source))).all()
    # FP64 square underflow must also preserve exact nonzero evidence identity.
    with torch.no_grad():
        op = op.double()
        op.attention_presence.weight[0, 0] = 1e-200
    assert torch.isfinite(op.log_presence(source.double())).all()


def test_collect_projected_zero_mass_does_not_leak_seed():
    inputs, request, _ = example_episode()
    model = RestorationModel(small_config()).double()
    with torch.no_grad():
        model.backbone.layers[0].collect.attention_presence.weight.zero_()
    baseline = model(inputs, request)
    with torch.no_grad():
        model.backbone.layers[0].slot_seed.add_(100)
    changed = model(inputs, request)
    torch.testing.assert_close(baseline.carriers, changed.carriers, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["direct", "inducing"])
def test_null_target_keeps_ll_and_differentiable_geometry(kind):
    episode = example_episode(damage=True)
    model = RestorationModel(small_config(kind)).double()
    score = score_episode(model, *episode)
    col = score.output.columns[0]
    null_position = 4  # target row 4 of numeric column
    fact = score.output.facts[0]
    nw = col.result.log_weights[null_position].exp() @ fact.answers.encoded
    assert not torch.allclose(col.result.encoding[null_position], nw)
    score.loss.backward()
    for parameter in (
        model.encoder.unit_seed,
        model.encoder.cell_seed,
        model.encoder.projection.weight,
        model.encoder.frequencies,
    ):
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
    if kind == "inducing":
        layer = model.backbone.layers[0]
        for parameter in (layer.slot_seed, layer.collect.k.weight, layer.collect.v.weight):
            assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_configuration_tiny_cpu_backward(dtype):
    # This is a structural smoke of the actual defaults, not a resource benchmark.
    model = RestorationModel().to(dtype=dtype)
    score = score_episode(model, *example_episode())
    score.loss.backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    assert model.backbone.layers[0].slot_seed.shape == (256, 128)


@pytest.mark.parametrize("mapping", ["identity128", "rotary32", "mlp32", "mlp256"])
def test_empty_categorical_support_and_empty_prediction_request(mapping):
    schema = (ColumnSchema("label", "nominal", 3),)
    visible = torch.zeros(2, 1, dtype=torch.bool)
    inputs = RestorationInput(schema, (torch.zeros(2, dtype=torch.long),), visible, ~visible)
    model = RestorationModel(small_config(category_map=mapping)).double()
    empty = model(inputs, RestorationRequest(torch.empty(0, 2, dtype=torch.long)))
    assert empty.columns == ()
    output = model(inputs, RestorationRequest(torch.tensor([[0, 0]])))
    assert output.columns[0].result.status == "no-support"


def test_invalid_requests_masks_config_and_no_support_batch():
    inputs, request, truth = example_episode()
    model = RestorationModel(small_config())
    for targets in (
        torch.tensor([[0, 0], [0, 0]]),
        torch.tensor([[-1, 0]]),
        torch.tensor([[0, 3]]),
    ):
        with pytest.raises(ValueError):
            model(inputs, RestorationRequest(targets))
    with pytest.raises(ValueError, match="disjoint"):
        RestorationInput(inputs.schema, inputs.values, inputs.visible, inputs.visible)
    with pytest.raises(ValueError, match="at least 128"):
        EncoderConfig(width=64)
    with pytest.raises(ValueError, match="four finite"):
        LossConfig((1, float("nan"), 1, 1))
    with pytest.raises(ValueError, match="at least one"):
        batch_loss(model, [])
    bad_states = copy.deepcopy(truth.states)
    bad_states[0, 0] = 1
    with pytest.raises(ValueError, match="roles disagree"):
        score_episode(model, inputs, request, TruthSidecar(truth.values, bad_states))
    with pytest.raises(ValueError, match="nonempty Q"):
        make_episode(
            inputs.schema,
            truth.values,
            torch.ones_like(inputs.visible),
            torch.zeros_like(inputs.query),
            code_seed=1,
        )


def test_cli_records_checks_and_never_overwrites(tmp_path, capsys):
    from tabu_lab.cli import main

    path = tmp_path / "check.json"
    assert main(["restoration", "verify", "--components-only", "--output", str(path)]) == 0
    before = path.read_bytes()
    assert json.loads(before)["status"] == "local_unissued"
    with pytest.raises(SystemExit):
        main(["restoration", "verify", "--components-only", "--output", str(path)])
    assert path.read_bytes() == before
    capsys.readouterr()
