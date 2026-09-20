"""Identity/rank geometry, raw sparse scale, and versioned training boundaries."""

import copy
from dataclasses import replace

import pytest
import torch

from tabu_lab.models.restoration.end_to_end_checks import example_episode
from tabu_lab.models.restoration_v53 import (
    AffineNumericAnswers,
    AffineOrdinalAnswers,
    ColumnSchema,
    ConstantWeightNominalAnswers,
    IdentityOrdinalAnswers,
    RestorationInput,
    RestorationRequest,
    TruthSidecar,
    V53LossConfig,
    V53Model,
    prepare_episode,
    score_episode,
)
from tabu_lab.models.restoration_v53.answers import constant_weight_vectors

from .test_model import config, deterministic_cpu  # noqa: F401

CURRENT_CODECS = ["unit_gaussian_v2", "constant_weight_v1"]


@pytest.mark.parametrize("version", CURRENT_CODECS)
def test_empty_query_rejected_before_training_but_visible_inference_is_valid(version):
    schema = (ColumnSchema("x", "numeric"),)
    values = (torch.arange(4, dtype=torch.float64),)
    observed = torch.ones(4, 1, dtype=torch.bool)
    inputs = RestorationInput(schema, values, observed, torch.zeros_like(observed), code_seed=7)
    request = RestorationRequest(observed.nonzero())
    truth = TruthSidecar(values, torch.zeros_like(observed, dtype=torch.long))
    episode = inputs, request, truth
    model = V53Model(replace(config(), codec_version=version)).double()
    inputs, request, _ = episode
    assert model(inputs, request).columns[0].result.status == "ok"
    with pytest.raises(ValueError, match="nonempty Query"):
        prepare_episode(model, *episode)


@pytest.mark.parametrize("version", CURRENT_CODECS)
def test_full_ordinal_identity_book_is_visible_independent_and_not_a_rank_line(version):
    schema = ColumnSchema("severity", "ordinal", 4, order=(2, 0, 3, 1))
    state = torch.get_rng_state().clone()
    codec = IdentityOrdinalAnswers.from_visible(
        torch.tensor([2, 1, 2]), schema=schema, seed=7, codec_version=version,
    )
    other = IdentityOrdinalAnswers.from_visible(
        torch.tensor([0, 3]), schema=schema, seed=7, codec_version=version,
    )
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(codec.codebook, other.codebook)
    assert torch.equal(codec.encoded[0], codec.encoded[2])
    assert codec.identities.unique(dim=0).shape[0] == 4
    assert torch.linalg.matrix_rank(codec.codebook - codec.codebook[0]) > 1
    torch.testing.assert_close(codec.rank_by_label, torch.tensor([1/3, 1, 0, 2/3]).double())
    torch.testing.assert_close(codec.decode(codec.encode_targets(torch.arange(4))), torch.arange(4))
    torch.testing.assert_close(codec.codebook,
                               codec.identities + codec.rank_by_label[:, None] * codec.direction)
    norm = 4 if version == "constant_weight_v1" else 1
    torch.testing.assert_close(codec.identities.square().sum(-1), torch.full((4,), norm).double())
    torch.testing.assert_close(codec.direction.square().sum(), torch.tensor(norm).double())
    if version == "constant_weight_v1":
        assert bool(((codec.identities == 0) | (codec.identities == 1)).all())
        torch.testing.assert_close(codec.codebook.sum(-1), 4 * (1 + codec.rank_by_label))


@pytest.mark.parametrize("version", CURRENT_CODECS)
def test_ordinal_decoder_uses_nearest_vector_including_candidate_norms_and_rank_ties(version):
    codec = IdentityOrdinalAnswers.from_visible(
        torch.tensor([0, 1]), schema=ColumnSchema("c", "ordinal", 7), seed=34,
        codec_version=version,
    )
    predictions = torch.randn(40, 128, dtype=torch.float64)
    ordered = codec.codebook[codec.classes]
    expected = (predictions[:, None] - ordered).square().sum(-1).argmin(-1)
    torch.testing.assert_close(codec.decode(predictions), codec.classes[expected])
    with pytest.raises(FloatingPointError, match="nonfinite"):
        codec.decode(torch.full((1, 128), float("nan")))
    # An exact, legal Gaussian geometry exposes both dot-only and rank-only errors.
    two = IdentityOrdinalAnswers.from_visible(
        torch.tensor([0, 1]), schema=ColumnSchema("tie", "ordinal", 2, order=(1, 0)), seed=1,
    )
    identities = torch.zeros(2, 128).double()
    identities[0, 0], identities[1, 1] = 1, 1
    direction = identities[0].clone()
    book = identities + two.rank_by_label[:, None] * direction
    two = replace(two, identities=identities, direction=direction, codebook=book, encoded=book)
    prediction = torch.zeros(2, 128).double()
    prediction[0, 0] = 0.7
    prediction[1, :2] = torch.tensor([1.0, 0.5])  # Exact equidistant midpoint.
    assert (prediction[0] @ book.T).argmax().item() == 0
    assert two.decode(prediction).tolist() == [1, 1]  # Lower declared rank wins the tie.


@pytest.mark.parametrize("version", CURRENT_CODECS)
def test_ordinal_singleton_and_empty_support(version):
    schema = ColumnSchema("single", "ordinal", 1)
    codec = IdentityOrdinalAnswers.from_visible(torch.tensor([0]), schema=schema, seed=5,
                                               codec_version=version)
    assert codec.rank_by_label.item() == 0
    assert torch.equal(codec.codebook, codec.identities)
    assert codec.decode(torch.randn(3, 128)).tolist() == [0, 0, 0]
    empty = IdentityOrdinalAnswers.from_visible(torch.empty(0, dtype=torch.long), schema=schema,
                                               seed=5, codec_version=version)
    assert torch.equal(empty.codebook, codec.codebook)
    with pytest.raises(ValueError, match="no-support"):
        empty.encode_targets(torch.tensor([0]))
    with pytest.raises(ValueError, match="no-support"):
        empty.decode(torch.zeros(1, 128))


def test_raw_sparse_numeric_inverse_includes_direction_norm_and_nominal_has_eight_ones():
    values = torch.tensor([-9.0, 1.0, 4.0, 12.0]).double()
    codec = AffineNumericAnswers.from_visible(values, epsilon=1e-6, seed=7, key="x",
                                             codec_version="constant_weight_v1")
    for vector in (codec.origin, codec.direction):
        assert bool(((vector == 0) | (vector == 1)).all())
        assert vector.sum() == 4
    assert not torch.equal(codec.origin, codec.direction)
    targets = torch.tensor([-100.0, 0.0, 200.0]).double()
    torch.testing.assert_close(codec.decode(codec.encode_targets(targets)), targets)
    schema = ColumnSchema("nom", "nominal", 6)
    nominal = ConstantWeightNominalAnswers.from_visible(torch.tensor([5, 1, 5, 3]),
                                                       schema=schema, seed=3)
    assert nominal.classes.tolist() == [1, 3, 5]
    assert bool((nominal.codebook.sum(-1) == 8).all())
    assert nominal.codebook.unique(dim=0).shape[0] == 3
    torch.testing.assert_close(nominal.decode(nominal.encoded), torch.tensor([5, 1, 5, 3]))
    with pytest.raises(ValueError, match="no-answer-code"):
        nominal.encode_targets(torch.tensor([0]))
    state = torch.get_rng_state().clone()
    repeated = ConstantWeightNominalAnswers.from_visible(torch.tensor([3, 1, 5]),
                                                        schema=schema, seed=3)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(repeated.codebook, nominal.codebook)
    with pytest.raises(ValueError, match="capacity"):
        constant_weight_vectors(["capacity"], 129, 1, torch.device("cpu"))


@pytest.mark.parametrize("version", CURRENT_CODECS)
def test_query_default_retained_switch_and_legacy_mean_are_distinct(version):
    model = V53Model(replace(config(), codec_version=version)).double()
    inputs, request, truth = example_episode()
    default = score_episode(model, inputs, request, truth)
    retained = score_episode(model, inputs, request, truth,
                             V53LossConfig(state_weights=(1, 0, 0, 0)))
    combined = score_episode(model, inputs, request, truth,
                             V53LossConfig(state_weights=(0.3, 1, 0, 0)))
    historical = score_episode(model, inputs, request, truth, V53LossConfig(state_weights=None))
    torch.testing.assert_close(combined.loss, default.loss + 0.3 * retained.loss)
    states = truth.states[request.targets[:, 0], request.targets[:, 1]]
    masks = [request.targets[:, 1] == 0, request.targets[:, 1] != 0]
    expected = sum(default.per_target[(states == 1) & mask].mean() for mask in masks)
    torch.testing.assert_close(default.loss, expected)
    torch.testing.assert_close(historical.loss,
                               sum(default.per_target[mask].mean() for mask in masks))


def test_old_gaussian_checkpoint_remains_explicit_and_cannot_load_as_new_codec():
    old = V53Model(replace(config(), codec_version="unit_gaussian_v1")).double()
    facts = old.encoder.prepare(example_episode()[0])
    assert isinstance(facts[2].answers, AffineOrdinalAnswers)
    state = copy.deepcopy(old.state_dict())
    assert state["_codec_signature"].tolist() == [1, 1]
    for version in CURRENT_CODECS:
        model = V53Model(replace(config(), codec_version=version)).double()
        with pytest.raises(RuntimeError, match="codec identity does not match"):
            model.load_state_dict(state, strict=False)
