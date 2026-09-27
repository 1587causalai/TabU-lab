"""V5.5 ordinal composition, explicit older candidate, and codec identity."""

import copy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from tabu_lab.models.restoration.end_to_end_checks import example_episode
from tabu_lab.models.restoration_v53.answers import (
    AffineOrdinalAnswers,
    CompositionOrdinalAnswers,
)
from tabu_lab.models.restoration_v54 import V54Config, V54Model
from tabu_lab.models.restoration_v55 import (
    CODEC_VERSIONS,
    DEFAULT_CODEC_VERSION,
    BackboneConfig,
    ColumnSchema,
    V55Config,
    V55Model,
    score_episode,
)

NEW_CODECS = ("constant_weight_composition_v2", "unit_gaussian_composition_v2")
OLD_CODECS = ("constant_weight_composition_v1", "unit_gaussian_composition_v1")


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        yield
    torch.set_num_threads(threads)


def small_model(version=DEFAULT_CODEC_VERSION):
    return V55Model(V55Config(
        backbone=BackboneConfig(layers=1, ff_width=24, slots=4),
        unit_layers=0, regression_width=5, center_chunk_size=2,
        codec_version=version,
    )).double()


@pytest.mark.parametrize("version", NEW_CODECS)
def test_ordinal_uses_full_category_plus_rank_composition(version):
    schema = ColumnSchema("severity", "ordinal", 4, order=(2, 0, 3, 1))
    codec = CompositionOrdinalAnswers.from_visible(
        torch.tensor([2, 1, 2]), schema=schema, seed=7, codec_version=version,
    )
    norm = 4 if version.startswith("constant_weight") else 1
    assert codec.codebook.shape == (4, 128)
    assert codec.classes.tolist() == [2, 0, 3, 1]
    torch.testing.assert_close(
        codec.rank_by_label, torch.tensor([1 / 3, 1, 0, 2 / 3]).double(),
    )
    torch.testing.assert_close(
        codec.codebook,
        codec.origin + codec.category_vectors + codec.rank_by_label[:, None] * codec.direction,
    )
    torch.testing.assert_close(codec.encoded, codec.codebook[torch.tensor([2, 1, 2])])
    bases = torch.cat((codec.origin[None], codec.direction[None], codec.category_vectors))
    torch.testing.assert_close(bases.square().sum(-1), torch.full((6,), norm).double())
    assert codec.category_vectors.unique(dim=0).shape[0] == 4
    assert codec.codebook.unique(dim=0).shape[0] == 4
    if norm == 4:
        assert bool(((bases == 0) | (bases == 1)).all())
        torch.testing.assert_close(codec.codebook.sum(-1), 4 * (2 + codec.rank_by_label))
    torch.testing.assert_close(codec.decode(codec.codebook), torch.arange(4))


@pytest.mark.parametrize("version", NEW_CODECS)
def test_ordinal_full_declared_domain_is_independent_of_visible_labels_and_process_rng(version):
    schema = ColumnSchema("severity", "ordinal", 5, order=(4, 2, 0, 3, 1))
    rng = torch.get_rng_state().clone()
    sparse = CompositionOrdinalAnswers.from_visible(
        torch.tensor([4, 1]), schema=schema, seed=11, codec_version=version,
    )
    assert torch.equal(rng, torch.get_rng_state())
    other = CompositionOrdinalAnswers.from_visible(
        torch.tensor([0, 2, 3]), schema=schema, seed=11, codec_version=version,
    )
    torch.testing.assert_close(sparse.codebook, other.codebook, rtol=0, atol=0)
    torch.testing.assert_close(sparse.category_vectors, other.category_vectors, rtol=0, atol=0)
    torch.testing.assert_close(
        sparse.encode_targets(torch.arange(5)), sparse.codebook, rtol=0, atol=0,
    )
    torch.testing.assert_close(sparse.decode(sparse.codebook), torch.arange(5))
    different = CompositionOrdinalAnswers.from_visible(
        torch.tensor([4, 1]), schema=schema, seed=12, codec_version=version,
    )
    assert not torch.equal(sparse.codebook, different.codebook)


@pytest.mark.parametrize("version", NEW_CODECS)
def test_ordinal_decoder_is_euclidean_over_schema_order_not_rank_projection(version):
    schema = ColumnSchema("order", "ordinal", 4, order=(2, 0, 3, 1))
    codec = CompositionOrdinalAnswers.from_visible(
        torch.tensor([2, 1]), schema=schema, seed=17, codec_version=version,
    )
    predictions = torch.randn(40, 128, dtype=torch.float64)
    candidates = codec.codebook[codec.classes]
    expected = (predictions[:, None] - candidates).square().sum(-1).argmin(-1)
    torch.testing.assert_close(codec.decode(predictions), codec.classes[expected])

    # The lower declared rank wins an exact Euclidean tie, while a dot-only
    # comparison picks the wrong answer for a candidate with larger norm.
    two = CompositionOrdinalAnswers.from_visible(
        torch.tensor([0, 1]), schema=ColumnSchema("tie", "ordinal", 2, order=(1, 0)),
        seed=1, codec_version=version,
    )
    book = torch.zeros(2, 128, dtype=torch.float64)
    book[0, 0] = 2
    book[1, 1] = 1
    two = replace(two, codebook=book)
    prediction = torch.zeros(2, 128, dtype=torch.float64)
    prediction[0, 0], prediction[0, 1] = 0.8, 0.2
    prediction[1, 0], prediction[1, 1] = 1, 0.5
    assert (prediction[0] @ book.T).argmax().item() == 0
    assert two.decode(prediction).tolist() == [1, 1]


@pytest.mark.parametrize("version", NEW_CODECS)
def test_ordinal_singleton_empty_and_invalid_values(version):
    schema = ColumnSchema("only", "ordinal", 1)
    codec = CompositionOrdinalAnswers.from_visible(
        torch.tensor([0]), schema=schema, seed=4, codec_version=version,
    )
    assert codec.rank_by_label.item() == 0
    torch.testing.assert_close(codec.codebook[0], codec.origin + codec.category_vectors[0])
    assert codec.decode(torch.randn(3, 128)).tolist() == [0, 0, 0]
    for invalid in (-1, 1):
        with pytest.raises(ValueError, match="outside the declared domain"):
            codec.encode_targets(torch.tensor([invalid]))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        codec.decode(torch.full((1, 128), float("nan")))
    empty = CompositionOrdinalAnswers.from_visible(
        torch.empty(0, dtype=torch.long), schema=schema, seed=4, codec_version=version,
    )
    torch.testing.assert_close(empty.codebook, codec.codebook, rtol=0, atol=0)
    with pytest.raises(ValueError, match="no-support"):
        empty.encode_targets(torch.tensor([0]))
    with pytest.raises(ValueError, match="no-support"):
        empty.decode(torch.zeros(1, 128))


@pytest.mark.parametrize("version", NEW_CODECS)
def test_model_input_lift_uses_same_ordinal_answers(version):
    model = small_model(version)
    inputs, request, _ = example_episode()
    prepared = model.prepare(inputs, request)
    fact = prepared.facts[2]
    assert isinstance(fact.answers, CompositionOrdinalAnswers)
    torch.testing.assert_close(fact.input_coordinates, fact.answers.encoded, rtol=0, atol=0)
    carriers = model.encoder.forward_prepared(prepared.inputs, prepared.features)
    expected = F.linear(fact.answers.encoded, model.encoder.projection.weight)
    torch.testing.assert_close(carriers[fact.rows, 2], expected)


@pytest.mark.parametrize("version", NEW_CODECS)
def test_new_ordinal_code_reaches_full_vector_query_loss_and_backward(version):
    model = small_model(version)
    inputs, request, truth = example_episode()
    score = score_episode(model, inputs, request, truth)
    ordinal = next(column for column in score.output.columns if column.column == 2)
    rows = request.targets[ordinal.target_indices, 0]
    truth_codes = score.output.facts[2].answers.encode_targets(truth.values[2][rows])
    expected = (ordinal.result.encoding - truth_codes).square().mean(-1)
    torch.testing.assert_close(score.per_target[ordinal.target_indices], expected)
    score.loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(bool(grad.isfinite().all()) for grad in grads)
    assert sum(float(grad.square().sum()) for grad in grads) > 0


@pytest.mark.parametrize(("old_version", "new_version"), zip(OLD_CODECS, NEW_CODECS, strict=True))
def test_v2_changes_only_ordinal_realization(old_version, new_version):
    inputs, _, _ = example_episode()
    old = small_model(old_version).encoder.prepare(inputs)
    new = small_model(new_version).encoder.prepare(inputs)
    for column in (0, 1):
        torch.testing.assert_close(old[column].answers.encoded,
                                   new[column].answers.encoded, rtol=0, atol=0)
    for field in ("origin", "direction"):
        torch.testing.assert_close(getattr(old[0].answers, field),
                                   getattr(new[0].answers, field), rtol=0, atol=0)
    for field in ("origin", "category_vectors", "codebook"):
        torch.testing.assert_close(getattr(old[1].answers, field),
                                   getattr(new[1].answers, field), rtol=0, atol=0)
    for field in ("origin", "direction"):
        torch.testing.assert_close(getattr(old[2].answers, field),
                                   getattr(new[2].answers, field), rtol=0, atol=0)
    assert not torch.equal(old[2].answers.encoded, new[2].answers.encoded)


def test_v55_default_and_old_candidate_have_distinct_checkpoint_identities():
    assert DEFAULT_CODEC_VERSION == "constant_weight_composition_v2"
    assert set((*NEW_CODECS, *OLD_CODECS)) <= set(CODEC_VERSIONS)
    assert V55Config(size="small", unit_layers=None).unit_layers == 0
    default = small_model()
    assert default.config.codec_version == DEFAULT_CODEC_VERSION
    with pytest.raises(TypeError, match="requires V54Config"):
        V54Model(V55Config())
    default_state = copy.deepcopy(default.state_dict())
    for version in (*OLD_CODECS, "unit_gaussian_composition_v2"):
        other = small_model(version)
        assert not torch.equal(default._codec_signature, other._codec_signature)
        with pytest.raises(RuntimeError, match="codec identity does not match"):
            other.load_state_dict(copy.deepcopy(default_state), strict=False)


@pytest.mark.parametrize("version", OLD_CODECS)
def test_old_ordinal_line_remains_an_explicit_replay_candidate(version):
    schema = ColumnSchema("severity", "ordinal", 4, order=(2, 0, 3, 1))
    inputs, _, _ = example_episode()
    old = AffineOrdinalAnswers.from_visible(
        torch.tensor([2, 1]), schema=schema, seed=7, codec_version=version,
    )
    model = small_model(version)
    selected = model.encoder.prepare(inputs)[2].answers
    assert type(selected) is AffineOrdinalAnswers
    historical = V54Model(V54Config(
        backbone=BackboneConfig(layers=1, ff_width=24, slots=4),
        unit_layers=0, codec_version=version,
    )).encoder.prepare(inputs)[2].answers
    torch.testing.assert_close(selected.origin, historical.origin, rtol=0, atol=0)
    torch.testing.assert_close(selected.direction, historical.direction, rtol=0, atol=0)
    torch.testing.assert_close(selected.encoded, historical.encoded, rtol=0, atol=0)
    torch.testing.assert_close(old.encode_targets(torch.arange(4)),
                               old.origin + old.rank_by_label[:, None] * old.direction)
