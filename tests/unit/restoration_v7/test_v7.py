"""V7 acceptance checks from the end-to-end design's implementation contract."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tabu_lab.models.restoration.contracts import ColumnSchema, RestorationInput, make_episode
from tabu_lab.models.restoration_v7 import (
    V7Config,
    V7Model,
    V7ProtocolError,
    build_g64_codec,
    column_shared_ll,
    compose_state,
    prepare_episode,
    reference_values,
    round_weights,
    score_rounds,
)
from tabu_lab.models.restoration_v7.readout import support_weights

SCHEMA = (
    ColumnSchema("x", "numeric"),
    ColumnSchema("c", "nominal", domain_size=4),
    ColumnSchema("o", "ordinal", domain_size=4, order=(2, 0, 3, 1)),
    ColumnSchema("y", "numeric"),
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield


def small_config(**kwargs) -> V7Config:
    values = dict(
        backbone=dict(width=64, layers=1, heads=2, ff_width=64, slots=4),
        rounds=3,
        coupling_hidden=(32, 32),
        center_chunk_size=5,
    )
    values.update(kwargs)
    return V7Config(**values)


def table(n: int = 12, query_rows: slice = slice(8, None), target: int = 3, **overrides):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(n, generator=generator, dtype=torch.float64)
    c = torch.randint(3, (n,), generator=generator)
    o = torch.randint(4, (n,), generator=generator)
    y = 2 * x + 0.1 * torch.randn(n, generator=generator, dtype=torch.float64)
    values = dict(x=x, c=c, o=o, y=y) | overrides
    observed = torch.ones(n, len(SCHEMA), dtype=torch.bool)
    query = torch.zeros_like(observed)
    query[query_rows, target] = True
    return make_episode(SCHEMA, tuple(values[s.key] for s in SCHEMA), observed, query, code_seed=3)


# --- G64 codec -------------------------------------------------------------


def test_g64_base_directions_are_unit_and_composed_codes_are_not_renormalised():
    inputs, _, _ = table()
    codec = build_g64_codec(inputs)
    numeric, nominal, ordinal = codec.columns[0], codec.columns[1], codec.columns[2]
    for vector in (numeric.base, numeric.direction, nominal.base, ordinal.direction):
        assert float(vector.norm()) == pytest.approx(1.0)
    assert torch.allclose(
        (nominal.codes - nominal.base).norm(dim=-1),
        torch.ones(len(nominal.codes), dtype=nominal.codes.dtype),
    )
    assert not torch.allclose(
        nominal.codes.norm(dim=-1), torch.ones(len(nominal.codes), dtype=nominal.codes.dtype)
    )


def test_numeric_round_trip_subtracts_the_column_base():
    inputs, _, _ = table()
    column = build_g64_codec(inputs).columns[0]
    values = torch.linspace(-3, 3, 7, dtype=torch.float64)
    assert abs(float(column.base @ column.direction)) > 1e-4
    assert torch.allclose(column.decode(column.encode(values)), values, atol=1e-10)


def test_numeric_statistics_use_population_std_of_visible_values_only():
    inputs, _, _ = table()
    column = build_g64_codec(inputs).columns[3]
    visible = inputs.values[3][inputs.visible[:, 3]]
    assert column.mean == pytest.approx(float(visible.mean()))
    assert column.scale == pytest.approx(float(visible.std(unbiased=False)))


@pytest.mark.parametrize(
    ("values", "expected_mean", "expected_scale", "epsilon"),
    (
        ([1e200, 2e200], 1.5e200, 5e199, 1e-6),
        ([1e308, 1e308], 1e308, 1e-6, 1e-6),
        ([-1.7e308, 1.7e308, 1.7e308], 1.7e308 / 3, 1.7e308 * (8 / 9) ** 0.5, 1e-6),
        ([1e-200, 3e-200], 2e-200, 1e-200, 1e-250),
    ),
)
def test_numeric_statistics_and_roundtrip_survive_representable_extremes(
    values, expected_mean, expected_scale, epsilon
):
    values = torch.tensor(values, dtype=torch.float64)
    visible = torch.ones(len(values), 1, dtype=torch.bool)
    inputs = RestorationInput(
        (ColumnSchema("y", "numeric"),), (values,), visible, torch.zeros_like(visible), 3
    )
    column = build_g64_codec(inputs, epsilon=epsilon).columns[0]
    assert column.mean == pytest.approx(expected_mean, rel=1e-14, abs=0)
    assert column.scale == pytest.approx(expected_scale, rel=1e-14, abs=0)
    codes = column.encode(values)
    decoded = column.decode(codes)
    assert codes.dtype == decoded.dtype == torch.float64
    assert bool(torch.isfinite(codes).all()) and bool(torch.isfinite(decoded).all())
    # Relative comparisons avoid overflowing a test's squared absolute error.
    assert torch.allclose(decoded / values.abs().max(), values / values.abs().max(), atol=1e-14)


def test_numeric_codec_rejects_unrepresentable_codes_and_decoded_answers():
    inputs, _, _ = table()
    column = build_g64_codec(inputs).columns[3]
    tiny_scale = replace(column, mean=0.0, scale=1e-6)
    with pytest.raises(FloatingPointError, match="G64 numeric codes"):
        tiny_scale.encode(torch.tensor([1e308], dtype=torch.float64))
    huge_scale = replace(column, mean=0.0, scale=1e200)
    finite_code = huge_scale.base[None] + 1e200 * huge_scale.direction[None]
    with pytest.raises(FloatingPointError, match="G64 numeric decoded values"):
        huge_scale.decode(finite_code)


@pytest.mark.parametrize(
    "statistics", ({"mean": float("inf")}, {"scale": float("inf")}, {"scale": 0})
)
def test_numeric_codec_rejects_invalid_statistics(statistics):
    inputs, _, _ = table()
    column = build_g64_codec(inputs).columns[0]
    with pytest.raises(FloatingPointError, match="G64 numeric statistics"):
        replace(column, **statistics)


def test_nominal_codebook_holds_only_visible_categories_and_flags_unseen_ones():
    inputs, _, _ = table()
    column = build_g64_codec(inputs).columns[1]
    visible = inputs.values[1][inputs.visible[:, 1]].unique()
    assert torch.equal(column.categories, visible)
    assert 3 not in visible.tolist()
    with pytest.raises(V7ProtocolError, match="no-answer-code"):
        column.encode(torch.tensor([3]))
    assert torch.equal(column.decode(column.encode(visible)), visible)


def test_ordinal_uses_full_declared_domain_with_rank_term():
    inputs, _, _ = table()
    column = build_g64_codec(inputs).columns[2]
    assert column.categories.tolist() == [0, 1, 2, 3]
    ranks = torch.tensor(SCHEMA[2].rank_positions(), dtype=torch.float64) / 3
    identity = column.codes - column.base - ranks[:, None] * column.direction
    assert torch.allclose(identity.norm(dim=-1), torch.ones(4, dtype=torch.float64))
    assert torch.equal(column.decode(column.codes), column.categories)


# --- round weights ---------------------------------------------------------


def test_round_weights_are_normalised_and_last_round_first():
    assert round_weights(1, 0.5) == (1.0,)
    assert round_weights(4, 0.5) == pytest.approx(tuple(w / 15 for w in (1, 2, 4, 8)))
    assert sum(round_weights(8, 0.5)) == pytest.approx(1.0)
    assert round_weights(8, 0.5)[0] == pytest.approx(1 / 255)
    assert round_weights(3, 1.0) == pytest.approx((1 / 3,) * 3)
    assert round_weights(3, 0.0) == (0.0, 0.0, 1.0)


# --- column-shared LL ------------------------------------------------------


def test_ll_matches_the_direct_weighted_least_squares_solution():
    generator = torch.Generator().manual_seed(1)
    n_rows, d, p = 9, 3, 5
    units = torch.randn(n_rows, 2, generator=generator, dtype=torch.float64)
    features = torch.randn(n_rows, d, generator=generator, dtype=torch.float64)
    support = torch.tensor([0, 2, 3, 5, 6, 8])
    responses = torch.randn(len(support), p, generator=generator, dtype=torch.float64)
    eval_rows = torch.tensor([1, 4, 7])
    ridge = 0.1
    result = column_shared_ll(
        units,
        support,
        responses,
        features,
        eval_rows,
        ridge=ridge,
        bandwidth=0.8,
        center_chunk_size=4,
    )
    weights = support_weights(units, units[support], 0.8)
    g = features[support]
    s_gg, s_eg = torch.zeros(d, d, dtype=torch.float64), torch.zeros(p, d, dtype=torch.float64)
    for r in range(n_rows):
        gc = g - weights[r] @ g
        ec = responses - weights[r] @ responses
        s_gg += (weights[r, :, None] * gc).T @ gc / n_rows
        s_eg += (weights[r, :, None] * ec).T @ gc / n_rows
    slope = s_eg @ torch.linalg.inv(s_gg + ridge * torch.eye(d, dtype=torch.float64))
    w = weights[eval_rows]
    expected = w @ responses + (features[eval_rows] - w @ g) @ slope.T
    assert torch.allclose(result.slope, slope, atol=1e-10)
    assert torch.allclose(result.recovered, expected, atol=1e-10)


def test_ll_single_support_returns_that_support_and_responses_get_gradients():
    units = torch.randn(4, 2, dtype=torch.float64)
    features = torch.randn(4, 3, dtype=torch.float64)
    one = torch.randn(1, 5, dtype=torch.float64)
    single = column_shared_ll(
        units, torch.tensor([0]), one, features, torch.tensor([2, 3]), ridge=1e-3, bandwidth=1.0
    )
    assert torch.allclose(single.recovered, one.expand(2, -1))

    responses = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    result = column_shared_ll(
        units,
        torch.tensor([0, 1, 2]),
        responses,
        features,
        torch.tensor([3]),
        ridge=1e-3,
        bandwidth=1.0,
    )
    result.recovered.square().sum().backward()
    assert responses.grad is not None and responses.grad.abs().sum() > 0


# --- episode preparation ---------------------------------------------------


def test_single_target_column_and_training_admission():
    inputs, _, _ = table()
    two_columns = inputs.query.clone()
    two_columns[0, 0] = True
    from tabu_lab.models.restoration.contracts import RestorationInput

    with pytest.raises(ValueError, match="exactly one target column"):
        prepare_episode(
            RestorationInput(
                inputs.schema,
                inputs.values,
                inputs.visible & ~two_columns,
                two_columns,
                inputs.code_seed,
            ),
            donor_seed=0,
        )
    constant, _, _ = table(y=torch.ones(12, dtype=torch.float64))
    with pytest.raises(V7ProtocolError, match="no-valid-episode"):
        prepare_episode(constant, donor_seed=0)


def test_inference_accepts_a_single_support_but_training_does_not():
    observed = torch.ones(4, len(SCHEMA), dtype=torch.bool)
    observed[1:3, 3] = False
    query = torch.zeros_like(observed)
    query[3, 3] = True
    values = (
        torch.randn(4, dtype=torch.float64),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 1, 2, 3]),
        torch.randn(4, dtype=torch.float64),
    )
    from tabu_lab.models.restoration.contracts import RestorationInput

    inputs = RestorationInput(SCHEMA, values, observed & ~query, query, 3)
    with pytest.raises(V7ProtocolError, match="no-valid-episode"):
        prepare_episode(inputs, donor_seed=0)
    episode = prepare_episode(inputs, donor_seed=0, admission="inference")
    assert episode.support_rows.tolist() == [0]


def test_donors_come_from_target_supports_once_and_follow_the_seed():
    inputs, _, _ = table()
    episode = prepare_episode(inputs, donor_seed=11)
    again = prepare_episode(inputs, donor_seed=11)
    assert torch.equal(episode.donor_rows, again.donor_rows)
    assert set(episode.donor_rows.tolist()) <= set(episode.support_rows.tolist())
    assert not set(episode.support_rows.tolist()) & set(episode.query_rows.tolist())

    model = V7Model(small_config()).double()
    first = model(episode)
    other = model(prepare_episode(inputs, donor_seed=99))
    seed = model.query_seed_numeric.detach()
    assert torch.equal(first.initial, seed.expand_as(first.initial))
    assert torch.equal(first.initial, other.initial)
    donor_codes = episode.observed[episode.donor_rows, episode.target]
    assert not torch.equal(first.initial, donor_codes)


@pytest.mark.parametrize(("target", "name"), ((1, "nominal"), (2, "ordinal")))
def test_query_seed_follows_the_target_kind(target, name):
    inputs, _, _ = table(target=target)
    model = V7Model(small_config()).double()
    output = model(prepare_episode(inputs, donor_seed=5))
    seed = model.query_seed(name).detach()
    assert torch.equal(output.initial, seed.expand_as(output.initial))
    others = {
        "numeric": model.query_seed_numeric,
        "nominal": model.query_seed_nominal,
        "ordinal": model.query_seed_ordinal,
    }
    for kind, parameter in others.items():
        if kind != name:
            assert not torch.equal(parameter.detach(), seed)


def test_donor_query_init_copies_the_donor_code():
    inputs, _, _ = table()
    model = V7Model(small_config(query_init="donor")).double()
    episode = prepare_episode(inputs, donor_seed=11)
    other = prepare_episode(inputs, donor_seed=99)
    first = model(episode)
    second = model(other)
    donor_codes = episode.observed[episode.donor_rows, episode.target].to(first.initial)
    assert torch.equal(first.initial, donor_codes)
    assert not torch.equal(first.initial, second.initial)


def test_compose_state_changes_only_query_target_cells():
    inputs, _, _ = table()
    episode = prepare_episode(inputs, donor_seed=0)
    state = torch.randn(len(episode.query_rows), 64, dtype=episode.observed.dtype)
    composed = compose_state(episode.observed, episode.query_rows, episode.target, state)
    changed = (composed != episode.observed).any(-1)
    assert changed.nonzero().tolist() == [[int(r), episode.target] for r in episode.query_rows]


# --- model -----------------------------------------------------------------


def test_initialisation_follows_the_example_spec():
    model = V7Model(small_config())
    block = model.rounds[0]
    x = torch.randn(10, 64)
    assert torch.equal(block.phi(x), x)
    gram = block.lift.weight.T @ block.lift.weight
    assert torch.allclose(gram, torch.eye(64), atol=1e-5)
    assert float(block.unit_seed.norm()) == pytest.approx(1.0)
    assert float(block.feature_seed.norm()) == pytest.approx(1.0)
    seeds = (model.query_seed_numeric, model.query_seed_nominal, model.query_seed_ordinal)
    assert len({parameter.data_ptr() for parameter in seeds}) == 3
    for parameter in seeds:
        assert parameter.shape == (64,)
    slots = block.backbone.layers[0].slot_seed
    assert torch.allclose(slots.norm(dim=-1), torch.ones(len(slots)))


def test_default_config_is_the_example_spec():
    config = V7Config()
    assert (config.code_dim, config.backbone.width, config.backbone.layers, config.rounds) == (
        64,
        128,
        4,
        1,
    )
    assert config.codec == "C64/8"
    assert config.query_init == "seed"
    with pytest.raises(ValueError, match="query_init"):
        small_config(query_init="broadcast")
    assert config.chi_numeric == 128
    assert (config.backbone.ff_width, config.backbone.slots, config.backbone.heads) == (512, 256, 4)
    assert V7Config.from_dict(config.as_dict()) == config
    with pytest.raises(ValueError, match="token width"):
        small_config(backbone=dict(width=32, layers=1, heads=2, ff_width=64, slots=4))


def test_round_sharing_switch():
    assert len(V7Model(small_config()).rounds) == 1
    assert len(V7Model(small_config(share_rounds=False)).rounds) == 3


def test_main_default_executes_one_recovery_and_scores_only_its_output():
    inputs, _, truth = table()
    config = V7Config()
    model = V7Model(config).double()
    episode = prepare_episode(inputs, donor_seed=2)
    calls = []
    handle = model.rounds[0].backbone.register_forward_hook(
        lambda module, args, output: calls.append(True)
    )
    output = model(episode, decode=False)
    handle.remove()
    assert calls == [True]
    assert len(output.states) == len(output.slopes) == 1
    assert torch.equal(output.initial, model.query_seed_numeric.expand_as(output.initial))
    reference = reference_values(episode, truth)
    codes = episode.codec.columns[episode.target].encode(reference).to(output.initial)
    expected = (128 / 64) * (output.states[0] - codes).square().sum(-1).mean()
    for rho in (0.0, 0.5, 1.0):
        score = score_rounds(output, episode, reference, replace(config, round_loss_rho=rho))
        assert score.weights == (1.0,)
        assert torch.allclose(score.loss, expected)
    expected.backward()
    assert model.query_seed_numeric.grad.abs().sum() > 0
    with torch.no_grad():
        other = model(prepare_episode(inputs, donor_seed=999), decode=False)
    assert torch.equal(output.states[0], other.states[0])


def test_hidden_query_truth_does_not_change_the_forward():
    inputs, _, _ = table()
    changed_y = inputs.values[3].clone()
    changed_y[8:] = 1e6
    other, _, _ = table(y=torch.where(torch.arange(12) >= 8, 1e6, changed_y))
    model = V7Model(small_config()).double()
    left = model(prepare_episode(inputs, donor_seed=4))
    right = model(prepare_episode(other, donor_seed=4))
    for a, b in zip(left.states, right.states, strict=True):
        assert torch.equal(a, b)


def test_forward_keeps_observed_codes_and_decodes_the_final_state():
    inputs, _, _ = table()
    episode = prepare_episode(inputs, donor_seed=2)
    before = episode.observed.clone()
    output = V7Model(small_config()).double()(episode)
    assert torch.equal(episode.observed, before)
    assert len(output.states) == 3
    column = episode.codec.columns[episode.target]
    assert torch.equal(output.decoded, column.decode(output.states[-1].detach()))


def test_loss_scores_each_written_state_and_reaches_the_parameters():
    inputs, _, truth = table()
    config = small_config()
    episode = prepare_episode(inputs, donor_seed=3)
    model = V7Model(config).double()
    output = model(episode, decode=False)
    reference = reference_values(episode, truth)
    score = score_rounds(output, episode, reference, config)

    codes = episode.codec.columns[episode.target].encode(reference)
    manual = torch.stack([(128 / 64) * (s - codes).square().sum(-1).mean() for s in output.states])
    assert torch.allclose(score.round_losses, manual)
    assert score.loss.item() == pytest.approx(
        float((manual.detach() * manual.new_tensor(round_weights(3, 0.5))).sum())
    )

    score.loss.backward()
    block = model.rounds[0]
    for name, parameter in (
        ("lift", block.lift.weight),
        ("unit seed", block.unit_seed),
        ("slot seed", block.backbone.layers[0].slot_seed),
        ("phi output layer", block.phi.blocks[0].net[-1].weight),
        ("numeric query seed", model.query_seed_numeric),
    ):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert block.feature_seed.grad is None or block.feature_seed.grad.abs().sum() == 0
    assert model.query_seed_nominal.grad is None or model.query_seed_nominal.grad.abs().sum() == 0
    assert model.query_seed_ordinal.grad is None or model.query_seed_ordinal.grad.abs().sum() == 0


def test_discrete_target_uses_the_discrete_coefficient():
    inputs, _, truth = table(target=1)
    config = small_config(rounds=1)
    episode = prepare_episode(inputs, donor_seed=0)
    output = V7Model(config).double()(episode, decode=False)
    codes = episode.codec.columns[1].encode(reference_values(episode, truth))
    score = score_rounds(output, episode, reference_values(episode, truth), config)
    expected = (1 / 64) * (output.states[0] - codes).square().sum(-1).mean()
    assert score.loss.item() == pytest.approx(float(expected))
