"""Default C64/8 contract and retained explicit G64 compatibility."""

from __future__ import annotations

import pytest
import torch

from tabu_lab.models.restoration.contracts import ColumnSchema, RestorationInput, make_episode
from tabu_lab.models.restoration_v7 import V7Config, V7Model, prepare_episode
from tabu_lab.models.restoration_v7.codec import (
    build_c64_codec,
    build_g64_codec,
    build_value_codec,
    constant_weight_bank,
)
from tabu_lab.models.restoration_v7.training import state_loss

SCHEMA = (
    ColumnSchema("x", "numeric"),
    ColumnSchema("c", "nominal", domain_size=4),
    ColumnSchema("o", "ordinal", domain_size=4, order=(2, 0, 3, 1)),
    ColumnSchema("y", "numeric"),
)


def table():
    generator = torch.Generator().manual_seed(0)
    n = 12
    x = torch.randn(n, generator=generator, dtype=torch.float64)
    c = torch.randint(3, (n,), generator=generator)
    o = torch.randint(4, (n,), generator=generator)
    y = 2 * x + 0.1 * torch.randn(n, generator=generator, dtype=torch.float64)
    observed = torch.ones(n, len(SCHEMA), dtype=torch.bool)
    query = torch.zeros_like(observed)
    query[8:, 3] = True
    return make_episode(SCHEMA, (x, c, o, y), observed, query, code_seed=3)


def assert_raw_weight(vector: torch.Tensor, weight: int = 8) -> None:
    assert vector.dtype == torch.float64
    assert set(vector.unique().tolist()) <= {0.0, 1.0}
    assert int(vector.sum()) == weight
    assert int(vector.square().sum()) == weight


def test_constant_weight_bank_is_uniform_raw_and_distinct():
    generator = torch.Generator().manual_seed(1)
    bank = constant_weight_bank(5, 6, 2, generator)
    assert bank.shape == (5, 6)
    seen = set()
    for row in bank:
        assert_raw_weight(row, 2)
        seen.add(tuple(row.tolist()))
    assert len(seen) == 5


def test_constant_weight_bank_rejects_a_request_past_capacity_and_redraws_collisions():
    generator = torch.Generator().manual_seed(2)
    with pytest.raises(ValueError, match="capacity"):
        constant_weight_bank(7, 4, 2, generator)
    filled = constant_weight_bank(3, 3, 1, generator)
    assert set(tuple(row.tolist()) for row in filled) == {
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    }


def test_c64_bases_are_raw_weight_8_and_composed_codes_stay_unnormalised():
    inputs, _, _ = table()
    codec = build_c64_codec(inputs)
    assert codec.family == "C64/8" and codec.dim == 64
    numeric, nominal, ordinal = codec.columns[0], codec.columns[1], codec.columns[2]
    for vector in (numeric.base, numeric.direction, nominal.base, ordinal.base, ordinal.direction):
        assert_raw_weight(vector)
    nominal_identity = nominal.codes - nominal.base
    for row in nominal_identity:
        assert_raw_weight(row)
    assert bool((nominal.codes.square().sum(-1) >= 16).all())


def test_c64_numeric_decoder_uses_denominator_8():
    inputs, _, _ = table()
    column = build_c64_codec(inputs).columns[0]
    targets = torch.tensor([-2.0, 0.0, 1.5], dtype=torch.float64)
    values = column.mean + column.scale * targets
    codes = column.encode(values)
    z = (codes - column.base) @ column.direction / 8
    assert torch.allclose(z, targets)
    assert torch.allclose(column.decode(codes), values, atol=1e-10)
    on_line = (column.base + column.direction) - column.encode(torch.tensor([column.mean]))
    assert float(on_line.square().sum()) == pytest.approx(8.0)
    assert float(
        state_loss(on_line[None], torch.zeros_like(on_line)[None], 128.0)
    ) == pytest.approx(16.0)


def test_c64_nominal_distances_match_overlap_and_argmax():
    inputs, _, _ = table()
    column = build_c64_codec(inputs).columns[1]
    identity = column.codes - column.base
    for i in range(len(identity)):
        for j in range(i + 1, len(identity)):
            overlap = int((identity[i] * identity[j]).sum())
            distance = float((identity[i] - identity[j]).square().sum())
            assert distance == 16 - 2 * overlap
            assert distance in set(range(2, 17, 2))
    truth = column.codes[0]
    delta = column.codes[1] - truth
    guess = truth + 0.1 * delta / delta.norm()
    assert int(column.decode(guess[None])) == int(column.categories[0])
    scores = identity @ (guess - column.base)
    assert int(scores.argmax()) == 0
    assert torch.equal(column.codes.sum(-1), torch.full((len(column.codes),), 16.0))


def test_c64_ordinal_codes_stay_distinct_across_the_declared_domain():
    inputs, _, _ = table()
    column = build_c64_codec(inputs).columns[2]
    assert column.categories.tolist() == [0, 1, 2, 3]
    assert len(column.codes.unique(dim=0)) == 4
    ranks = torch.tensor(SCHEMA[2].rank_positions(), dtype=torch.float64) / 3
    identity = column.codes - column.base - ranks[:, None] * column.direction
    assert torch.allclose(identity.sum(-1), identity.new_full((len(identity),), 8.0), atol=1e-12)
    assert bool(((identity - identity.round()).abs() < 1e-12).all())
    assert torch.equal(column.decode(column.codes), column.categories)


def test_hidden_nominal_labels_do_not_enter_the_codebook_and_the_seed_replays():
    values = torch.tensor([0, 1, 0, 1, 3, 3])
    visible = torch.tensor([True, True, True, True, False, False])[:, None]
    query = ~visible
    schema = (ColumnSchema("c", "nominal", domain_size=4),)
    hidden = RestorationInput(schema, (values,), visible, query, code_seed=11)
    other = RestorationInput(
        schema, (torch.tensor([0, 1, 0, 1, 2, 9]),), visible, query, code_seed=11
    )
    column = build_c64_codec(hidden).columns[0]
    again = build_c64_codec(other).columns[0]
    assert column.categories.tolist() == [0, 1]
    assert torch.equal(column.codes, again.codes)
    shifted = RestorationInput(schema, (values,), visible, query, code_seed=12)
    assert not torch.equal(column.base, build_c64_codec(shifted).columns[0].base)


def test_c64_is_default_and_keeps_the_loss_coefficient():
    config = V7Config()
    assert config.codec == "C64/8"
    assert config.code_dim == 64 and config.chi_numeric == 128 and config.chi_discrete == 1
    with pytest.raises(ValueError, match="code dimension 64"):
        build_c64_codec(*table()[:1], dim=32)
    with pytest.raises(ValueError, match="C64/8 requires code_dim 64"):
        V7Config(
            codec="C64/8",
            code_dim=32,
            backbone=dict(width=64, layers=1, heads=2, ff_width=64, slots=4),
        )
    with pytest.raises(ValueError, match="codec must be G64 or C64/8"):
        V7Config(codec="H4")
    inputs, _, _ = table()
    assert build_value_codec(inputs).family == config.codec
    assert prepare_episode(inputs, donor_seed=2).codec.family == config.codec
    assert build_g64_codec(inputs).family == "G64"
    assert float(build_g64_codec(inputs).columns[0].direction.square().sum()) == pytest.approx(1.0)


def test_model_uses_the_selected_family_and_rejects_a_mismatch():
    inputs, _, _ = table()
    config = V7Config(
        codec="C64/8",
        backbone=dict(width=64, layers=1, heads=2, ff_width=64, slots=4),
        rounds=1,
        coupling_hidden=(32, 32),
    )
    episode = prepare_episode(inputs, donor_seed=2, codec="C64/8")
    assert episode.codec.family == "C64/8"
    assert_raw_weight(episode.codec.columns[3].direction)
    output = V7Model(config).double()(episode)
    assert output.decoded.shape == (4,)
    assert bool(torch.isfinite(output.decoded).all())
    g64 = prepare_episode(inputs, donor_seed=2, codec="G64")
    with pytest.raises(ValueError, match="codec family"):
        V7Model(config).double()(g64)


def test_new_configuration_defaults_and_saved_codec_choices_are_unambiguous():
    current = V7Config()
    assert V7Config.from_dict(current.as_dict()) == current
    assert V7Config.from_dict({}).codec == "C64/8"
    for family in ("G64", "C64/8", "constant_weight_composition_v2"):
        config = V7Config(
            codec=family, code_dim=128 if family == "constant_weight_composition_v2" else 64
        )
        assert V7Config.from_dict(config.as_dict()) == config
