"""V5.4 composition geometry, decoder correctness and historical identities."""

import copy
from dataclasses import replace

import pytest
import torch

from tabu_lab.models.restoration.end_to_end_checks import example_episode
from tabu_lab.models.restoration_v53 import (
    BackboneConfig,
    ColumnSchema,
    RestorationInput,
    RestorationRequest,
    TruthSidecar,
    V53Config,
    V53Model,
    prepare_episode,
    score_prepared_episode,
)
from tabu_lab.models.restoration_v53.answers import (
    AffineOrdinalAnswers,
    CompositionNominalAnswers,
    ConstantWeightNominalAnswers,
    GaussianNominalAnswers,
    constant_weight_vectors,
    unit_gaussians,
)
from tabu_lab.models.restoration_v53.codec_versions import (
    CODEC_IDS,
    COMPOSITION_CODEC_VERSIONS,
    DEFAULT_CODEC_VERSION,
    DEFAULT_COMPOSITION_CODEC_VERSION,
)
from tabu_lab.models.restoration_v53.encoding import AffineNumericAnswers, AffineValueEncoder


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        yield
    torch.set_num_threads(threads)


def small_model(version):
    # Exercise the shared implementation independently of the V5.4 entry point.
    return V53Model(V53Config(
        backbone=BackboneConfig(layers=1, ff_width=24, slots=4),
        regression_width=5, center_chunk_size=2, codec_version=version,
    )).double()


@pytest.mark.parametrize("version", COMPOSITION_CODEC_VERSIONS)
def test_nominal_column_origin_replay_and_visible_category_geometry(version):
    schema = ColumnSchema("color", "nominal", 8)
    labels = torch.tensor([6, 1, 6, 3])
    rng = torch.get_rng_state().clone()
    codec = CompositionNominalAnswers.from_visible(
        labels, schema=schema, seed=19, codec_version=version,
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert codec.classes.tolist() == [1, 3, 6]
    assert codec.category_vectors.unique(dim=0).shape[0] == 3
    torch.testing.assert_close(codec.codebook, codec.origin + codec.category_vectors)
    torch.testing.assert_close(codec.decode(codec.encoded), labels)
    repeated = CompositionNominalAnswers.from_visible(
        labels.flip(0), schema=schema, seed=19, codec_version=version,
    )
    torch.testing.assert_close(codec.codebook, repeated.codebook, rtol=0, atol=0)
    expanded = CompositionNominalAnswers.from_visible(
        torch.tensor([0, 1, 3, 6]), schema=schema, seed=19, codec_version=version,
    )
    # Column identity is independent of visible class coverage.
    torch.testing.assert_close(codec.origin, expanded.origin, rtol=0, atol=0)
    other = CompositionNominalAnswers.from_visible(
        labels, schema=replace(schema, key="other"), seed=19, codec_version=version,
    )
    assert not torch.equal(codec.origin, other.origin)
    bases = torch.cat((codec.origin[None], codec.category_vectors))
    norm = 4 if version == "constant_weight_composition_v1" else 1
    torch.testing.assert_close(bases.square().sum(-1), torch.full((4,), norm).double())
    if norm == 4:
        assert bool(((bases == 0) | (bases == 1)).all())
        overlap = (codec.category_vectors * codec.origin).sum(-1)
        torch.testing.assert_close(codec.codebook.square().sum(-1), 8 + 2 * overlap)
        torch.testing.assert_close(codec.codebook.sum(-1), torch.full((3,), 8.0).double())
        pairwise = torch.cdist(codec.codebook, codec.codebook).square()
        assert float(pairwise.max()) <= 8.0 + 1e-12
    with pytest.raises(ValueError, match="no-answer-code"):
        codec.encode_targets(torch.tensor([0]))
    predictions = torch.randn(40, 128, dtype=torch.float64)
    nearest = (predictions[:, None] - codec.codebook).square().sum(-1).argmin(-1)
    torch.testing.assert_close(codec.decode(predictions), codec.classes[nearest])


@pytest.mark.parametrize("version", COMPOSITION_CODEC_VERSIONS)
def test_nominal_decoder_subtracts_origin_for_unequal_norms_and_breaks_ties(version):
    codec = CompositionNominalAnswers.from_visible(
        torch.tensor([0, 1]), schema=ColumnSchema("unequal", "nominal", 2),
        seed=1, codec_version=version,
    )
    # Legal basis banks: q=b_0 is allowed, b_1 is disjoint/orthogonal.
    width = 4 if version == "constant_weight_composition_v1" else 1
    vectors = torch.zeros(2, 128, dtype=torch.float64)
    vectors[0, :width], vectors[1, width:2 * width] = 1, 1
    origin = vectors[0].clone()
    book = origin + vectors
    codec = replace(codec, origin=origin, category_vectors=vectors, codebook=book, encoded=book)
    prediction = 0.25 * origin[None]
    assert book[0].square().sum() > book[1].square().sum()
    assert (prediction @ book.T).argmax().item() == 0  # Incorrect old shortcut.
    assert codec.decode(prediction).item() == 1
    midpoint = origin + vectors.mean(0)
    assert codec.decode(midpoint[None]).item() == 0  # Declared nominal order.
    with pytest.raises(FloatingPointError, match="nonfinite"):
        codec.decode(torch.full((1, 128), float("nan")))


@pytest.mark.parametrize("version", COMPOSITION_CODEC_VERSIONS)
def test_composition_empty_and_singleton_nominal_boundaries(version):
    schema = ColumnSchema("single", "nominal", 3)
    codec = CompositionNominalAnswers.from_visible(
        torch.tensor([2, 2]), schema=schema, seed=2, codec_version=version,
    )
    assert codec.decode(torch.randn(3, 128)).tolist() == [2, 2, 2]
    empty = CompositionNominalAnswers.from_visible(
        torch.empty(0, dtype=torch.long), schema=schema, seed=2, codec_version=version,
    )
    assert empty.encoded.shape == empty.codebook.shape == (0, 128)
    for operation in (
        lambda: empty.encode_targets(torch.tensor([2])),
        lambda: empty.decode(torch.zeros(1, 128)),
    ):
        with pytest.raises(ValueError, match="no-support"):
            operation()


@pytest.mark.parametrize("version", COMPOSITION_CODEC_VERSIONS)
def test_numeric_and_full_schema_ordinal_keep_the_shared_affine_line(version):
    numeric = AffineNumericAnswers.from_visible(
        torch.tensor([-3.0, 1.0, 7.0]), epsilon=1e-6, seed=7, key="x",
        codec_version=version,
    )
    targets = torch.tensor([-100.0, 0.0, 200.0]).double()
    torch.testing.assert_close(numeric.decode(numeric.encode_targets(targets)), targets)
    norm = 4 if version == "constant_weight_composition_v1" else 1
    assert numeric.direction_norm_squared == pytest.approx(norm)
    assert not torch.equal(numeric.origin, numeric.direction)
    schema = ColumnSchema("severity", "ordinal", 4, order=(2, 0, 3, 1))
    ordinal = AffineOrdinalAnswers.from_visible(
        torch.tensor([2, 1]), schema=schema, seed=7, codec_version=version,
    )
    all_codes = ordinal.encode_targets(torch.arange(4))
    torch.testing.assert_close(all_codes, ordinal.origin + ordinal.rank_by_label[:, None]
                               * ordinal.direction)
    torch.testing.assert_close(ordinal.decode(all_codes), torch.arange(4))
    torch.testing.assert_close(torch.cdist(all_codes, all_codes).square(),
                               norm * (ordinal.rank_by_label[:, None]
                                       - ordinal.rank_by_label[None]).square())
    other = AffineOrdinalAnswers.from_visible(
        torch.tensor([0, 3]), schema=schema, seed=7, codec_version=version,
    )
    torch.testing.assert_close(ordinal.origin, other.origin, rtol=0, atol=0)
    torch.testing.assert_close(ordinal.direction, other.direction, rtol=0, atol=0)
    empty = AffineOrdinalAnswers.from_visible(
        torch.empty(0, dtype=torch.long), schema=schema, seed=7, codec_version=version,
    )
    with pytest.raises(ValueError, match="no-support"):
        empty.decode(all_codes)


@pytest.mark.parametrize("version", COMPOSITION_CODEC_VERSIONS)
def test_composition_truth_isolation_loss_scales_and_affine_ll_closure(version):
    model = small_model(version)
    inputs, request, truth = example_episode()
    altered_values = list(truth.values)
    altered_values[0] = altered_values[0].clone()
    altered_values[0][-1] = 500
    altered = TruthSidecar(tuple(altered_values), truth.states)
    prepared = prepare_episode(model, inputs, request, truth)
    other = prepare_episode(model, inputs, request, altered)
    score, changed = (score_prepared_episode(model, p, decode=True) for p in (prepared, other))
    torch.testing.assert_close(score.output.carriers, changed.output.carriers, rtol=0, atol=0)
    assert not torch.equal(score.per_target, changed.per_target)
    norm = 4 if version == "constant_weight_composition_v1" else 1
    for column in score.output.columns:
        codec = score.output.facts[column.column].answers
        rows = request.targets[column.target_indices, 0]
        error = column.result.encoding - codec.encode_targets(truth.values[column.column][rows])
        expected = error.square().sum(-1)
        if column.column == 0:
            torch.testing.assert_close(expected, norm * (
                (column.decoded - truth.values[0][rows]) / codec.scalar.scale
            ).square())
            torch.testing.assert_close(column.result.encoding, codec.encode_targets(column.decoded))
        else:
            expected = expected / 128
        torch.testing.assert_close(score.per_target[column.target_indices], expected)
    score.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(g.isfinite().all()) for g in gradients)
    assert sum(float(g.square().sum()) for g in gradients) > 0


@pytest.mark.parametrize("version", COMPOSITION_CODEC_VERSIONS)
@pytest.mark.parametrize("field", ["origin", "category_vectors", "codebook"])
def test_all_composition_decoder_state_is_guarded_by_prepared_snapshots(version, field):
    model = small_model(version)
    prepared = prepare_episode(model, *example_episode())
    getattr(prepared.visible.facts[1].answers, field).add_(1)
    with pytest.raises(ValueError, match="mutated"):
        score_prepared_episode(model, prepared)


@pytest.mark.parametrize("version", COMPOSITION_CODEC_VERSIONS)
def test_empty_columns_are_no_support_without_reading_hidden_placeholders(version):
    model = small_model(version)
    schema = (ColumnSchema("x", "numeric"), ColumnSchema("n", "nominal", 2),
              ColumnSchema("o", "ordinal", 2))
    visible = torch.zeros(2, 3, dtype=torch.bool)
    inputs = RestorationInput(schema, (torch.full((2,), float("nan")),
                                      torch.full((2,), 999), torch.full((2,), 999)),
                              visible, ~visible, code_seed=3)
    output = model(inputs, RestorationRequest((~visible).nonzero()))
    assert [c.result.status for c in output.columns] == ["no-support"] * 3


def test_historical_id_defaults_and_realization_namespaces_are_unchanged():
    assert DEFAULT_CODEC_VERSION == V53Config().codec_version == "constant_weight_v1"
    assert DEFAULT_COMPOSITION_CODEC_VERSION == "constant_weight_composition_v1"
    assert CODEC_IDS == {
        "legacy_v53": 0, "unit_gaussian_v1": 1, "unit_gaussian_v2": 2,
        "constant_weight_v1": 3, "constant_weight_composition_v1": 4,
        "unit_gaussian_composition_v1": 5,
    }
    inputs, _, _ = example_episode()
    for version in ("legacy_v53", "unit_gaussian_v1", "unit_gaussian_v2", "constant_weight_v1"):
        facts = AffineValueEncoder(codec_version=version).prepare(inputs)
        key, seed, device = inputs.schema[0].key, inputs.code_seed, inputs.visible.device
        if version == "constant_weight_v1":
            expected = constant_weight_vectors(["v53-affine-constant", seed, key], 2, 4, device)
            assert isinstance(facts[1].answers, ConstantWeightNominalAnswers)
            assert bool((facts[1].answers.codebook.square().sum(-1) == 8).all())
        else:
            expected = unit_gaussians(["v53-affine", seed, key], 2, device)
            if version != "legacy_v53":
                assert type(facts[1].answers) is GaussianNominalAnswers
        torch.testing.assert_close(facts[0].answers.origin, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(facts[0].answers.direction, expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("version", COMPOSITION_CODEC_VERSIONS)
def test_new_codec_checkpoint_signatures_roundtrip_and_reject_other_identities(version):
    model, clone = small_model(version), small_model(version)
    state = copy.deepcopy(model.state_dict())
    assert state["_codec_signature"].tolist() == [CODEC_IDS[version], 1]
    clone.load_state_dict(state)
    inputs, request, _ = example_episode()
    for left, right in zip(model(inputs, request).columns, clone(inputs, request).columns,
                           strict=True):
        torch.testing.assert_close(left.result.encoding, right.result.encoding, rtol=0, atol=0)
    for other in CODEC_IDS:
        if other != version:
            with pytest.raises(RuntimeError, match="codec identity does not match"):
                small_model(other).load_state_dict(copy.deepcopy(state), strict=False)
