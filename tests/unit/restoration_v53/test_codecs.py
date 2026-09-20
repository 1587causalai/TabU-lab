import copy
import json
from dataclasses import replace

import pytest
import torch

from tabu_lab.models.restoration.answers import NumericAnswers
from tabu_lab.models.restoration.encoding import visible_codes
from tabu_lab.models.restoration.end_to_end_checks import example_episode
from tabu_lab.models.restoration_v53 import (
    AffineNumericAnswers,
    AffineOrdinalAnswers,
    ColumnSchema,
    GaussianNominalAnswers,
    RestorationInput,
    RestorationRequest,
    TruthSidecar,
    V53Config,
    V53Model,
    ZScoreAnswers,
    prepare_episode,
    score_episode,
    score_prepared_episode,
)

from .test_model import config, deterministic_cpu  # noqa: F401


def test_population_zscore_uses_visible_statistics_and_truth_cannot_update_them():
    values = torch.tensor([0.0, 1.0, 2.0, 9.0], dtype=torch.float64, requires_grad=True)
    codec = ZScoreAnswers.from_visible(values, epsilon=1e-6)
    torch.testing.assert_close(codec.mean, values.mean())
    torch.testing.assert_close(codec.scale, values.std(correction=0))
    torch.testing.assert_close(codec.encoded.mean(), torch.zeros((), dtype=torch.float64))
    torch.testing.assert_close(codec.encoded.square().mean(), torch.ones((), dtype=torch.float64))
    assert not codec.encoded.requires_grad
    mean, scale = codec.mean.clone(), codec.scale.clone()
    targets = torch.tensor([-100.0, 0.0, 200.0], dtype=torch.float64, requires_grad=True)
    encoded = codec.encode_targets(targets)
    assert not encoded.requires_grad
    torch.testing.assert_close(codec.decode(encoded), targets)
    assert torch.equal(codec.mean, mean) and torch.equal(codec.scale, scale)
    robust = NumericAnswers.from_visible(values, epsilon=1e-6)
    assert not torch.allclose(robust.encoded, codec.encoded)


@pytest.mark.parametrize("values", [[7.0], [7.0, 7.0, 7.0]])
def test_zscore_constant_and_singleton_use_scale_floor(values):
    values = torch.tensor(values, dtype=torch.float64)
    codec = ZScoreAnswers.from_visible(values, epsilon=0.01)
    assert codec.scale == 0.01
    assert torch.count_nonzero(codec.encoded) == 0
    torch.testing.assert_close(codec.decode(codec.encoded), values)


def test_zscore_large_offsets_extremes_and_empty_are_explicit():
    shifted = torch.tensor([1e12, 1e12 + 1, 1e12 + 2], dtype=torch.float64)
    codec = ZScoreAnswers.from_visible(shifted, epsilon=1e-6)
    torch.testing.assert_close(codec.scale, shifted.std(correction=0), rtol=1e-12, atol=0)
    extreme = torch.tensor([-1e308, 1e308], dtype=torch.float64)
    codec = ZScoreAnswers.from_visible(extreme, epsilon=1e-6)
    torch.testing.assert_close(codec.encoded[:, 0], torch.tensor([-1.0, 1.0]).double())
    empty = ZScoreAnswers.from_visible(torch.empty(0), epsilon=1e-6)
    assert empty.mean is None and empty.scale is None
    with pytest.raises(ValueError, match="no-support"):
        empty.encode_targets(torch.tensor([0.0]))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        ZScoreAnswers.from_visible(torch.tensor([float("inf")]), epsilon=1e-6)


def test_nominal_unit_codes_reproducible_per_class_and_do_not_fill_hidden_classes():
    schema = ColumnSchema("color", "nominal", 5)
    labels = torch.tensor([3, 1, 3, 1])
    state = torch.get_rng_state().clone()
    codec = GaussianNominalAnswers.from_visible(labels, schema=schema, seed=12)
    assert torch.equal(torch.get_rng_state(), state)
    assert codec.classes.tolist() == [1, 3]
    torch.testing.assert_close(codec.codebook.norm(dim=1), torch.ones(2).double())
    assert bool((codec.codebook < 0).any()) and bool((codec.codebook > 0).any())
    torch.testing.assert_close(codec.decode(codec.encoded), labels)
    expanded = GaussianNominalAnswers.from_visible(
        torch.tensor([4, 3, 1, 0]), schema=schema, seed=12
    )
    assert torch.equal(codec.encode_targets(labels), expanded.encode_targets(labels))
    changed = GaussianNominalAnswers.from_visible(labels, schema=schema, seed=13)
    assert not torch.equal(codec.codebook, changed.codebook)
    with pytest.raises(ValueError, match="no-answer-code"):
        codec.encode_targets(torch.tensor([0]))
    assert codec.decode(torch.zeros(1, 128)).item() == 1


def test_nominal_decoder_matches_nearest_euclidean_code_for_non_tied_predictions():
    codec = GaussianNominalAnswers.from_visible(
        torch.arange(7), schema=ColumnSchema("c", "nominal", 7), seed=34
    )
    prediction = torch.randn(40, 128, dtype=torch.float64)
    expected = ((prediction[:, None] - codec.codebook).square().sum(-1)).argmin(-1)
    torch.testing.assert_close(codec.decode(prediction), codec.classes[expected])
    with pytest.raises(FloatingPointError, match="nonfinite"):
        codec.decode(torch.full((1, 128), float("nan")))


def test_ordinal_schema_rank_geometry_includes_unseen_declared_categories():
    schema = ColumnSchema("severity", "ordinal", 4, order=(2, 0, 3, 1))
    codec = AffineOrdinalAnswers.from_visible(torch.tensor([2, 1]), schema=schema, seed=7)
    labels = torch.arange(4)
    encoded = codec.encode_targets(labels)
    torch.testing.assert_close(codec.decode(encoded), labels)
    torch.testing.assert_close(codec.rank_by_label, torch.tensor([1/3, 1, 0, 2/3]).double())
    torch.testing.assert_close(
        torch.cdist(encoded, encoded),
        (codec.rank_by_label[:, None] - codec.rank_by_label[None, :]).abs(),
    )
    assert abs(codec.origin @ codec.direction) > 1e-4
    with pytest.raises(ValueError, match="declared domain"):
        codec.encode_targets(torch.tensor([4]))


def test_ordinal_projection_decodes_lower_rank_ties_endpoints_and_singleton():
    schema = ColumnSchema("rank", "ordinal", 3, order=(2, 0, 1))
    codec = AffineOrdinalAnswers.from_visible(torch.tensor([0, 1]), schema=schema, seed=5)
    # Exact axes make the midpoint tie representable without projection roundoff.
    origin, direction = torch.zeros(128).double(), torch.zeros(128).double()
    origin[1], direction[0] = 1, 1
    codec = replace(codec, origin=origin, direction=direction)
    coordinates = torch.tensor([-10, 0.25, 0.75, 10]).double()
    prediction = origin + coordinates[:, None] * direction
    prediction[:, 2] = 4  # Orthogonal residual does not change the projected rank.
    assert codec.decode(prediction).tolist() == [2, 2, 0, 1]
    singleton = AffineOrdinalAnswers.from_visible(
        torch.tensor([0]), schema=ColumnSchema("only", "ordinal", 1), seed=5
    )
    assert singleton.rank_by_label.item() == 0
    assert torch.equal(singleton.encoded[0], singleton.origin)
    assert singleton.decode(torch.randn(3, 128)).tolist() == [0, 0, 0]


def test_explicit_legacy_candidate_preserves_old_scalar_codes_and_ordinal_lift():
    model = V53Model(replace(config(), codec_version="legacy_v53",
                             numeric_scaling="median_half_iqr")).double()
    inputs, _, _ = example_episode()
    facts = model.encoder.prepare(inputs)
    scalar = NumericAnswers.from_visible(inputs.values[0][facts[0].rows], epsilon=1e-6)
    assert torch.equal(facts[0].answers.scalar.encoded, scalar.encoded)
    initial = model.encoder(inputs, facts)
    for a in (1, 2):
        classes, codes = visible_codes(inputs.values[a][facts[a].rows], width=128,
                                       seed=inputs.code_seed, key=inputs.schema[a].key)
        assert torch.equal(facts[a].answers.classes, classes)
        assert torch.equal(facts[a].answers.codebook, codes)
        lift = facts[a].answers.encoded
        if a == 2:
            lift = lift + facts[a].rank[:, None]
        torch.testing.assert_close(initial[facts[a].rows, a], model.encoder.projection(lift))
    # Robust scaling remains independently selectable with the new discrete codec.
    robust = AffineNumericAnswers.from_visible(
        inputs.values[0][facts[0].rows], epsilon=1e-6, seed=inputs.code_seed,
        key=inputs.schema[0].key, scaling="median_half_iqr",
    )
    assert torch.equal(robust.encoded, facts[0].answers.encoded)


@pytest.mark.parametrize("codec_version", ["unit_gaussian_v1", "legacy_v53"])
@pytest.mark.parametrize("numeric_scaling", ["zscore", "median_half_iqr"])
def test_config_and_checkpoint_codec_identity_roundtrip(codec_version, numeric_scaling):
    selected = replace(config(), codec_version=codec_version, numeric_scaling=numeric_scaling)
    recovered = V53Config.from_dict(json.loads(json.dumps(selected.as_dict())))
    assert recovered == selected
    model, clone = V53Model(selected).double(), V53Model(recovered).double()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    baseline, actual = model(*example_episode()[:2]), clone(*example_episode()[:2])
    for left, right in zip(baseline.columns, actual.columns, strict=True):
        torch.testing.assert_close(left.result.encoding, right.result.encoding, atol=0, rtol=0)


def test_unversioned_and_mismatched_checkpoints_cannot_silently_become_new_defaults():
    model = V53Model(config()).double()
    unversioned = model.config.as_dict()
    del unversioned["codec_version"]
    with pytest.raises(ValueError, match="lacks codec identity"):
        V53Config.from_dict(unversioned)
    state = copy.deepcopy(model.state_dict())
    del state["_codec_signature"]
    with pytest.raises(RuntimeError, match="lacks codec identity"):
        model.load_state_dict(state, strict=False)
    legacy = V53Model(replace(config(), codec_version="legacy_v53",
                              numeric_scaling="median_half_iqr")).double()
    legacy.load_state_dict(state)  # Explicit historical interpretation only.
    with pytest.raises(RuntimeError, match="codec identity does not match"):
        model.load_state_dict(legacy.state_dict(), strict=False)


def test_unseen_ordinal_truth_is_scorable_but_inference_remains_truth_free():
    schema = (ColumnSchema("rank", "ordinal", 4, order=(2, 0, 3, 1)),)
    visible = torch.tensor([[True], [True], [False]])
    inputs = RestorationInput(schema, (torch.tensor([2, 1, 999]),), visible, ~visible, 14)
    request = RestorationRequest(torch.tensor([[0, 0], [1, 0], [2, 0]]))
    truth = TruthSidecar((torch.tensor([2, 1, 0]),), torch.tensor([[0], [0], [1]]))
    other_truth = TruthSidecar((torch.tensor([2, 1, 3]),), truth.states)
    model = V53Model(config()).double()
    first, second = prepare_episode(model, inputs, request, truth), prepare_episode(
        model, inputs, request, other_truth
    )
    assert torch.equal(first.visible.facts[0].answers.encoded,
                       second.visible.facts[0].answers.encoded)
    assert not torch.equal(first.encoded_truth[0], second.encoded_truth[0])
    a, b = score_prepared_episode(model, first), score_prepared_episode(model, second)
    torch.testing.assert_close(a.output.carriers, b.output.carriers, atol=0, rtol=0)
    # Ordinal keeps chi=1: continuous rank error / 128, before hard decoding.
    codec = first.visible.facts[0].answers
    ranks = (a.output.columns[0].result.encoding - codec.origin) @ codec.direction
    expected = (ranks - codec.rank_by_label[truth.values[0]]).square() / 128
    torch.testing.assert_close(a.per_target, expected)
    codec.rank_by_label.add_(1)
    with pytest.raises(ValueError, match="mutated"):
        score_prepared_episode(model, first)


@pytest.mark.parametrize("kind", ["nominal", "ordinal"])
def test_new_discrete_no_support_is_explicit_and_nominal_book_is_guarded(kind):
    schema = (ColumnSchema("c", kind, 2),)
    visible = torch.zeros(2, 1, dtype=torch.bool)
    inputs = RestorationInput(schema, (torch.tensor([999, 999]),), visible, ~visible, 3)
    request = RestorationRequest(torch.tensor([[0, 0], [1, 0]]))
    model = V53Model(config()).double()
    assert model(inputs, request).columns[0].result.status == "no-support"
    if kind == "nominal":
        inputs, request, truth = example_episode()
        prepared = prepare_episode(model, inputs, request, truth)
        prepared.visible.facts[1].answers.codebook.add_(1)
        with pytest.raises(ValueError, match="mutated"):
            score_prepared_episode(model, prepared)


def test_numeric_diversity_rejected_before_forward_but_constant_inference_remains_valid():
    schema = (ColumnSchema("x", "numeric"),)
    visible = torch.tensor([[True], [True], [False]])
    values = (torch.tensor([7.0, 7.0, 100.0]),)
    inputs = RestorationInput(schema, values, visible, ~visible)
    request = RestorationRequest(torch.tensor([[0, 0], [1, 0], [2, 0]]))
    truth = TruthSidecar(values, torch.tensor([[0], [0], [1]]))
    model = V53Model(config()).double()
    torch.testing.assert_close(model(inputs, request).columns[0].decoded,
                               torch.full((3,), 7.0).double())
    with torch.no_grad():
        model.encoder.projection.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="two distinct"):
        score_episode(model, inputs, request, truth)
