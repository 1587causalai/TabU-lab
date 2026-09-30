"""Split construction and binding helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from tabu_lab.contracts.dataset import RawDataset, SplitManifest, SplitView


def split_dataset(
    dataset: RawDataset,
    partitions: Mapping[str, Iterable[str | int]],
    *,
    split_id: str = "default",
    fit_partition: str = "train",
    strategy: str = "explicit",
    seed: int | None = None,
    require_complete: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> SplitManifest:
    """Create a content-bound manifest without compiling any episode."""

    if not isinstance(dataset, RawDataset):
        raise TypeError("split_dataset requires a RawDataset")
    return SplitManifest.create(
        dataset,
        partitions,
        split_id=split_id,
        fit_partition=fit_partition,
        strategy=strategy,
        seed=seed,
        require_complete=require_complete,
        metadata=metadata,
    )


def bind_split_view(dataset: RawDataset, manifest: SplitManifest, partition: str) -> SplitView:
    """Validate and expose exactly one declared partition."""

    if not isinstance(dataset, RawDataset) or not isinstance(manifest, SplitManifest):
        raise TypeError("bind_split_view requires RawDataset and SplitManifest")
    return SplitView(dataset=dataset, manifest=manifest, partition=partition)


def bind_split_views(
    dataset: RawDataset, manifest: SplitManifest
) -> Mapping[str, SplitView]:
    return {
        name: bind_split_view(dataset, manifest, name)
        for name in manifest.partitions
    }


__all__ = ["bind_split_view", "bind_split_views", "split_dataset"]
