"""Nominal column direction candidate leaves the V5.5 v2 line intact."""

import copy

import pytest
import torch
from torch.nn import functional as F

from tabu_lab.models.restoration.encoding import prepare_features
from tabu_lab.models.restoration.end_to_end_checks import example_episode
from tabu_lab.models.restoration_v53.answers import CompositionNominalAnswers
from tabu_lab.models.restoration_v53.codec_versions import CODEC_IDS
from tabu_lab.models.restoration_v55 import (
    CODEC_VERSIONS,
    DEFAULT_CODEC_VERSION,
    BackboneConfig,
    ColumnSchema,
    RestorationInput,
    V55Config,
    V55Model,
    prepare_episode,
    score_prepared_episode,
)

V2 = "constant_weight_composition_v2"
V3 = "constant_weight_composition_v3"


def small_model(version):
    return V55Model(V55Config(
        backbone=BackboneConfig(layers=1, ff_width=24, slots=4),
        unit_layers=0, regression_width=5, center_chunk_size=2,
        codec_version=version,
    )).double()


def test_v3_is_explicit_and_reuses_every_trainable_parameter_shape():
    assert DEFAULT_CODEC_VERSION == V2
    assert V3 in CODEC_VERSIONS
    assert CODEC_IDS[V3] == 8
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        old = small_model(V2)
        torch.manual_seed(29)
        new = small_model(V3)
    assert old._codec_signature.tolist() == [6, 1]
    assert new._codec_signature.tolist() == [8, 1]
    old_parameters = dict(old.named_parameters())
    new_parameters = dict(new.named_parameters())
    assert old_parameters.keys() == new_parameters.keys()
    for name in old_parameters:
        torch.testing.assert_close(old_parameters[name], new_parameters[name], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="codec identity does not match"):
        new.load_state_dict(copy.deepcopy(old.state_dict()), strict=False)


def test_v3_only_changes_observed_nominal_code_and_keeps_query_carrier():
    inputs, _, _ = example_episode()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        old_model = small_model(V2)
        torch.manual_seed(31)
        new_model = small_model(V3)
    old = old_model.encoder.prepare(inputs)
    new = new_model.encoder.prepare(inputs)
    for column in (0, 2):
        torch.testing.assert_close(old[column].answers.encoded,
                                   new[column].answers.encoded, rtol=0, atol=0)
        torch.testing.assert_close(old[column].input_coordinates,
                                   new[column].input_coordinates, rtol=0, atol=0)
    old_nominal, new_nominal = old[1].answers, new[1].answers
    assert isinstance(old_nominal, CompositionNominalAnswers)
    assert isinstance(new_nominal, CompositionNominalAnswers)
    assert old_nominal.direction is None
    assert new_nominal.direction is not None
    for field in ("origin", "category_vectors"):
        torch.testing.assert_close(getattr(old_nominal, field),
                                   getattr(new_nominal, field), rtol=0, atol=0)
    torch.testing.assert_close(new_nominal.codebook,
                               old_nominal.codebook + new_nominal.direction,
                               rtol=0, atol=0)
    torch.testing.assert_close(new_nominal.encoded,
                               old_nominal.encoded + new_nominal.direction,
                               rtol=0, atol=0)
    assert bool(((new_nominal.direction == 0) | (new_nominal.direction == 1)).all())
    assert new_nominal.direction.square().sum().item() == 4

    old_carriers = old_model.encoder.forward_prepared(inputs, prepare_features(inputs, old))
    new_carriers = new_model.encoder.forward_prepared(inputs, prepare_features(inputs, new))
    n, m = inputs.visible.shape
    torch.testing.assert_close(old_carriers[:n, :m][inputs.query],
                               new_carriers[:n, :m][inputs.query], rtol=0, atol=0)
    torch.testing.assert_close(old_carriers[old[1].rows, 1]
                               + F.linear(new_nominal.direction[None],
                                          old_model.encoder.projection.weight),
                               new_carriers[new[1].rows, 1], rtol=0, atol=0)


def test_v3_nominal_direction_is_stable_across_visible_coverage_and_decode():
    schema = ColumnSchema("category", "nominal", 4)
    labels = torch.tensor([2, 0, 2, 1])
    rng = torch.get_rng_state().clone()
    codec = CompositionNominalAnswers.from_visible(
        labels, schema=schema, seed=17, codec_version=V3,
    )
    assert torch.equal(torch.get_rng_state(), rng)
    assert codec.classes.tolist() == [0, 1, 2]
    torch.testing.assert_close(codec.codebook,
                               codec.origin + codec.direction + codec.category_vectors,
                               rtol=0, atol=0)
    torch.testing.assert_close(codec.decode(codec.encoded), labels)
    torch.testing.assert_close(codec.encode_targets(labels), codec.encoded)
    reordered = CompositionNominalAnswers.from_visible(
        labels.flip(0), schema=schema, seed=17, codec_version=V3,
    )
    expanded = CompositionNominalAnswers.from_visible(
        torch.tensor([0, 1, 2, 3]), schema=schema, seed=17, codec_version=V3,
    )
    torch.testing.assert_close(codec.codebook, reordered.codebook, rtol=0, atol=0)
    torch.testing.assert_close(codec.direction, expanded.direction, rtol=0, atol=0)
    torch.testing.assert_close(codec.origin, expanded.origin, rtol=0, atol=0)
    with pytest.raises(ValueError, match="no-answer-code"):
        codec.encode_targets(torch.tensor([3]))

    empty = CompositionNominalAnswers.from_visible(
        torch.empty(0, dtype=torch.long), schema=schema, seed=17, codec_version=V3,
    )
    torch.testing.assert_close(empty.direction, codec.direction, rtol=0, atol=0)
    assert empty.encoded.shape == empty.codebook.shape == (0, 128)
    with pytest.raises(ValueError, match="no-support"):
        empty.decode(torch.zeros(1, 128))
    with pytest.raises(ValueError, match="no-support"):
        empty.encode_targets(torch.tensor([0]))


def test_v3_realizations_follow_column_keys_after_permutation():
    inputs, _, _ = example_episode()
    order = (2, 0, 1)
    permuted = RestorationInput(
        tuple(inputs.schema[index] for index in order),
        tuple(inputs.values[index] for index in order),
        inputs.visible[:, order], inputs.query[:, order], inputs.code_seed,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        encoder = small_model(V3).encoder
    original_facts = encoder.prepare(inputs)
    permuted_facts = encoder.prepare(permuted)
    for position, original_position in enumerate(order):
        original = original_facts[original_position].answers
        moved = permuted_facts[position].answers
        torch.testing.assert_close(original.encoded, moved.encoded, rtol=0, atol=0)
        if original_position == 1:
            torch.testing.assert_close(original.direction, moved.direction, rtol=0, atol=0)


def test_v3_direction_is_guarded_in_prepared_snapshot_and_backward_is_finite():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model = small_model(V3)
    prepared = prepare_episode(model, *example_episode())
    score = score_prepared_episode(model, prepared)
    assert torch.isfinite(score.loss)
    score.loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(bool(grad.isfinite().all()) for grad in grads)
    assert sum(float(grad.square().sum()) for grad in grads) > 0
    prepared.visible.facts[1].answers.direction.add_(1)
    with pytest.raises(ValueError, match="mutated"):
        score_prepared_episode(model, prepared)
