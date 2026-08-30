from __future__ import annotations

import pytest
import torch

from tabu_lab.compiler import (
    CategoricalCodebook,
    FeatureSelectionManifest,
    FitPartitionBindingError,
    Imputer,
    NumericNormalizer,
    bind_split_view,
    split_dataset,
)
from tabu_lab.contracts import FeatureKind, FeatureSpec, OriginState, RawDataset


def _views(*, held_out_numeric: float, held_out_category: float):  # type: ignore[no-untyped-def]
    dataset = RawDataset(
        dataset_id="fit-isolation",
        values=torch.tensor(
            [
                [1.0, 10.0],
                [3.0, 0.0],
                [held_out_numeric, 10.0],
                [held_out_numeric + 1.0, held_out_category],
            ]
        ),
        origin_states=(
            (OriginState.OBSERVED, OriginState.OBSERVED),
            (OriginState.OBSERVED, OriginState.NATURAL_MISSING),
            (OriginState.OBSERVED, OriginState.OBSERVED),
            (OriginState.OBSERVED, OriginState.OBSERVED),
        ),
        row_ids=("fit-0", "fit-1", "test-0", "test-1"),
        feature_specs=(
            FeatureSpec(name="numeric"),
            FeatureSpec(
                name="category",
                kind=FeatureKind.CATEGORICAL,
                domain=("ten", "twenty", "other"),
                codebook_id="statistics-test-category-v1",
            ),
        ),
    )
    manifest = split_dataset(dataset, {"train": (0, 1), "test": (2, 3)})
    return (
        bind_split_view(dataset, manifest, "train"),
        bind_split_view(dataset, manifest, "test"),
    )


def test_nonfit_mutation_cannot_change_fitted_artifacts() -> None:
    fit_a, _ = _views(held_out_numeric=100.0, held_out_category=20.0)
    fit_b, _ = _views(held_out_numeric=-9_999.0, held_out_category=999.0)

    normalizer_a = NumericNormalizer.fit(fit_a)
    normalizer_b = NumericNormalizer.fit(fit_b)
    codebook_a = CategoricalCodebook.fit(fit_a, "category")
    codebook_b = CategoricalCodebook.fit(fit_b, "category")
    imputer_a = Imputer.fit(fit_a)
    imputer_b = Imputer.fit(fit_b)
    selection_a = FeatureSelectionManifest.fit(fit_a, ("category", "numeric"))
    selection_b = FeatureSelectionManifest.fit(fit_b, ("category", "numeric"))

    assert fit_a.view_hash == fit_b.view_hash
    assert normalizer_a.artifact_hash == normalizer_b.artifact_hash
    assert codebook_a.artifact_hash == codebook_b.artifact_hash
    assert imputer_a.artifact_hash == imputer_b.artifact_hash
    assert selection_a.artifact_hash == selection_b.artifact_hash
    assert normalizer_a.statistics.counts.tolist() == [2, 0]
    assert normalizer_a.statistics.means.tolist() == [2.0, 0.0]
    assert codebook_a.vocabulary == (10.0,)
    assert codebook_a.oov_code == 1
    assert imputer_a.fitted.counts.tolist() == [2, 1]
    assert imputer_a.fitted.fill_values.tolist() == [2.0, 10.0]


def test_fit_on_nonfit_partition_is_rejected() -> None:
    _, test_view = _views(held_out_numeric=100.0, held_out_category=20.0)

    with pytest.raises(FitPartitionBindingError, match="fit partition"):
        NumericNormalizer.fit(test_view)
    with pytest.raises(FitPartitionBindingError, match="fit partition"):
        CategoricalCodebook.fit(test_view, "category")
    with pytest.raises(FitPartitionBindingError, match="fit partition"):
        Imputer.fit(test_view)
    with pytest.raises(FitPartitionBindingError, match="fit partition"):
        FeatureSelectionManifest.fit(test_view, ("numeric",))


def test_transform_uses_fit_statistics_and_explicit_oov() -> None:
    fit_view, test_view = _views(held_out_numeric=100.0, held_out_category=20.0)
    normalizer = NumericNormalizer.fit(fit_view)
    codebook = CategoricalCodebook.fit(fit_view, "category")
    imputer = Imputer.fit(fit_view)
    selection = FeatureSelectionManifest.fit(fit_view, ("category", "numeric"))

    transformed = normalizer.transform(test_view)
    encoded = codebook.transform(test_view)
    imputed = imputer.transform(fit_view)
    selected = selection.apply(test_view)

    assert transformed[:, 0].tolist() == [98.0, 99.0]
    assert transformed[:, 1].tolist() == [10.0, 20.0]
    assert encoded.tolist() == [0, codebook.oov_code]
    assert imputed.tolist() == [[1.0, 10.0], [3.0, 10.0]]
    assert selected.feature_names == ("category", "numeric")
    assert selected.values.tolist() == [[10.0, 100.0], [20.0, 101.0]]
    assert selected.origin_states.shape == selected.values.shape


def test_transform_rejects_another_split_definition() -> None:
    fit_view, _ = _views(held_out_numeric=100.0, held_out_category=20.0)
    normalizer = NumericNormalizer.fit(fit_view)
    dataset = fit_view.dataset
    other_manifest = split_dataset(
        dataset,
        {"train": (0, 2), "test": (1, 3)},
        split_id="other",
    )
    other_view = bind_split_view(dataset, other_manifest, "test")

    with pytest.raises(FitPartitionBindingError, match="split definition"):
        normalizer.transform(other_view)


def test_numeric_statistics_exclude_precommitted_episode_targets() -> None:
    fit_view, _ = _views(held_out_numeric=100.0, held_out_category=20.0)
    excluded = torch.zeros(fit_view.shape, dtype=torch.bool)
    excluded[1, 0] = True

    normalizer = NumericNormalizer.fit(fit_view, excluded_mask=excluded)
    transformed = normalizer.transform(fit_view)

    assert normalizer.statistics.counts.tolist() == [1, 0]
    assert normalizer.statistics.means.tolist() == [1.0, 0.0]
    assert transformed[:, 0].tolist() == [0.0, 2.0]


def test_numeric_statistics_bind_the_excluded_target_mask() -> None:
    fit_view, _ = _views(held_out_numeric=100.0, held_out_category=20.0)
    first = torch.zeros(fit_view.shape, dtype=torch.bool)
    second = first.clone()
    second[0, 0] = True

    normalizer_a = NumericNormalizer.fit(fit_view, excluded_mask=first)
    normalizer_b = NumericNormalizer.fit(fit_view, excluded_mask=second)

    assert normalizer_a.statistics.fit_view_hash == normalizer_b.statistics.fit_view_hash
    assert normalizer_a.statistics.fit_value_mask_hash != (
        normalizer_b.statistics.fit_value_mask_hash
    )
    assert normalizer_a.artifact_hash != normalizer_b.artifact_hash


def test_response_family_can_share_one_numeric_scale_across_columns() -> None:
    dataset = RawDataset.from_values(
        dataset_id="shared-response-family",
        values=torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
        row_ids=("u0", "u1"),
        feature_names=("i0", "i1"),
    )
    manifest = split_dataset(dataset, {"train": (0, 1)})
    fit_view = bind_split_view(dataset, manifest, "train")

    normalizer = NumericNormalizer.fit(
        fit_view,
        shared_numeric_groups=(("i0", "i1"),),
    )
    transformed = normalizer.transform(fit_view)

    assert normalizer.statistics.counts.tolist() == [4, 4]
    assert normalizer.statistics.means.tolist() == [4.0, 4.0]
    assert normalizer.statistics.scales.tolist() == [5**0.5, 5**0.5]
    assert torch.allclose(
        transformed,
        (dataset.values.to(torch.float64) - 4.0) / (5**0.5),
    )
