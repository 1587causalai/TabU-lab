from __future__ import annotations

import pytest
import torch

from tabu_lab.compiler import (
    EpisodeCompiler,
    FitPartitionBindingError,
    NumericNormalizer,
    SplitBeforeCompileError,
    bind_split_view,
    compile_episode,
    split_dataset,
)
from tabu_lab.contracts import EpisodeRecipe, ForwardRole, RawDataset

SOURCE = ForwardRole.RECEIVER | ForwardRole.SOURCE
TARGET = ForwardRole.RECEIVER | ForwardRole.TARGET


def _bound_views():  # type: ignore[no-untyped-def]
    dataset = RawDataset.from_values(
        dataset_id="binding",
        values=torch.arange(12, dtype=torch.float32).reshape(6, 2),
        row_ids=tuple(f"row-{index}" for index in range(6)),
    )
    manifest = split_dataset(
        dataset,
        {"train": (0, 1, 2), "validation": (3,), "test": (4, 5)},
        split_id="fixed",
        fit_partition="train",
    )
    return (
        dataset,
        manifest,
        bind_split_view(dataset, manifest, "train"),
        bind_split_view(dataset, manifest, "validation"),
        bind_split_view(dataset, manifest, "test"),
    )


def test_raw_dataset_cannot_cross_compile_boundary() -> None:
    dataset, _, fit_view, _, _ = _bound_views()
    recipe = EpisodeRecipe.create(
        fit_view,
        fit_view,
        (
            (SOURCE, TARGET),
            (SOURCE, ForwardRole.RECEIVER),
            (SOURCE, ForwardRole.RECEIVER),
        ),
    )

    with pytest.raises(SplitBeforeCompileError, match="SplitView"):
        EpisodeCompiler().compile(dataset, recipe, fit_view=fit_view)  # type: ignore[arg-type]


def test_recipe_is_bound_to_declared_fit_partition() -> None:
    _, _, fit_view, validation_view, test_view = _bound_views()
    roles = (
        (SOURCE, TARGET),
        (SOURCE, TARGET),
    )
    recipe = EpisodeRecipe.create(test_view, fit_view, roles)

    result = compile_episode(test_view, recipe, fit_view=fit_view)
    assert result.evidence.fit_partition == "train"
    assert result.provenance.fit_view_hash == fit_view.view_hash

    with pytest.raises(FitPartitionBindingError, match="fit partition"):
        compile_episode(test_view, recipe, fit_view=validation_view)


def test_recipe_from_another_source_view_is_rejected() -> None:
    _, _, fit_view, validation_view, test_view = _bound_views()
    validation_recipe = EpisodeRecipe.create(
        validation_view,
        fit_view,
        ((SOURCE, TARGET),),
    )

    with pytest.raises(FitPartitionBindingError, match="not bound"):
        compile_episode(test_view, validation_recipe, fit_view=fit_view)


def test_post_binding_dataset_mutation_fails_closed() -> None:
    dataset, _, fit_view, _, test_view = _bound_views()
    recipe = EpisodeRecipe.create(
        test_view,
        fit_view,
        (
            (SOURCE, TARGET),
            (SOURCE, TARGET),
        ),
    )
    dataset.values[0, 0] = 999.0

    with pytest.raises(SplitBeforeCompileError, match="not bound"):
        compile_episode(test_view, recipe, fit_view=fit_view)


def test_split_manifest_is_disjoint_and_complete() -> None:
    dataset = RawDataset.from_values(
        dataset_id="invalid-split",
        values=torch.ones((3, 2)),
        row_ids=("a", "b", "c"),
    )

    with pytest.raises(ValueError, match="pairwise disjoint"):
        split_dataset(dataset, {"train": ("a", "b"), "test": ("b", "c")})
    with pytest.raises(ValueError, match="omits"):
        split_dataset(dataset, {"train": ("a",), "test": ("b",)})


def test_compiler_applies_target_excluded_fit_normalizer_to_evidence_and_sidecar() -> None:
    _, _, fit_view, _, _ = _bound_views()
    roles = (
        (SOURCE, TARGET),
        (SOURCE, SOURCE),
        (SOURCE, SOURCE),
    )
    recipe = EpisodeRecipe.create(fit_view, fit_view, roles)
    excluded = torch.tensor(
        [[False, True], [False, False], [False, False]],
        dtype=torch.bool,
    )
    normalizer = NumericNormalizer.fit(fit_view, excluded_mask=excluded)

    result = compile_episode(
        fit_view,
        recipe,
        fit_view=fit_view,
        numeric_normalizer=normalizer,
    )

    expected = normalizer.transform(fit_view).to(torch.float32)
    assert torch.allclose(
        result.evidence.forward_values[result.evidence.source_mask],
        expected[result.evidence.source_mask],
    )
    assert torch.allclose(
        result.truth.target_values[result.truth.target_mask],
        expected[result.truth.target_mask],
    )
    assert result.provenance.numeric_normalizer_hash == normalizer.artifact_hash
    assert result.evidence.metadata["numeric_normalized"] is True
    assert "numeric_normalizer_hash" not in result.evidence.metadata


def test_compiler_rejects_fit_normalizer_that_includes_or_misexcludes_target_truth() -> None:
    _, _, fit_view, _, _ = _bound_views()
    roles = (
        (SOURCE, TARGET),
        (SOURCE, SOURCE),
        (SOURCE, SOURCE),
    )
    recipe = EpisodeRecipe.create(fit_view, fit_view, roles)
    leaky = NumericNormalizer.fit(fit_view)
    wrong_exclusion = torch.zeros(fit_view.shape, dtype=torch.bool)
    wrong_exclusion[1, 1] = True
    mismatched = NumericNormalizer.fit(fit_view, excluded_mask=wrong_exclusion)

    for normalizer in (leaky, mismatched):
        with pytest.raises(FitPartitionBindingError, match="target exclusion"):
            compile_episode(
                fit_view,
                recipe,
                fit_view=fit_view,
                numeric_normalizer=normalizer,
            )
