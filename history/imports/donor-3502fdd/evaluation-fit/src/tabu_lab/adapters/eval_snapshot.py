"""Offline registration boundary for prepared Evaluation Foundry snapshots.

This module deliberately performs no fetching and does not know how to turn a
provider-specific file into examples.  It closes the smaller but important
boundary between a truth-isolated :class:`PreparedScenario` and the public
``DatasetSnapshotSpec`` indexed by the Git-native catalog.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from tabu_lab.catalog import DatasetAdapter, DatasetAuthorityStatus, DatasetSnapshotSpec
from tabu_lab.contracts import to_canonical_data
from tabu_lab.evaluation.foundry import EvalSuiteSpec, PreparedScenario, dry_run_suite

ADAPTER_ID = "offline-eval-prepared-snapshot"
ADAPTER_VERSION = "1.0.0"


class EvalSnapshotRegistrationError(ValueError):
    """A prepared scenario cannot be registered under its frozen suite."""


def dataset_snapshot_from_prepared(
    *,
    suite: EvalSuiteSpec,
    scenario_id: str,
    prepared: PreparedScenario,
    request_sha256: str,
    authority_sha256: str,
    adapter_id: str = ADAPTER_ID,
    adapter_version: str = ADAPTER_VERSION,
) -> DatasetSnapshotSpec:
    """Validate and bind one offline prepared scenario to a catalog snapshot.

    The returned manifest binds the exact retained source bytes, prepared
    content, split, and preparation recipe.  It does not imply that the data is
    redistributable and it does not make an evaluation result publication
    eligible.
    """

    matches = tuple(item for item in suite.scenarios if item.scenario_id == scenario_id)
    if len(matches) != 1:
        raise EvalSnapshotRegistrationError(
            f"suite must contain exactly one scenario named {scenario_id!r}"
        )
    scenario = matches[0]
    try:
        prepared = PreparedScenario.model_validate(prepared.model_dump(mode="python"))
    except ValueError as error:
        raise EvalSnapshotRegistrationError("prepared scenario failed self-verification") from error

    # Reuse the foundry's public readiness gate without requiring unrelated
    # scenarios in the same suite to be materialized.
    single_scenario_suite = suite.model_copy(update={"scenarios": (scenario,)})
    availability = dry_run_suite(
        single_scenario_suite,
        prepared={scenario_id: prepared},
    ).scenarios[0]
    if not availability.ready:
        raise EvalSnapshotRegistrationError(
            "prepared scenario violates the frozen suite: " + ", ".join(availability.blockers)
        )

    content_sha256 = prepared.content_hash
    snapshot_id = f"{scenario.dataset.dataset_id}-{scenario_id}-{content_sha256[:16]}"
    mask_boundary = (
        "post-split artificial masking; target truth remains evaluator-side"
        if scenario.mask is not None
        else "test target truth remains evaluator-side and absent from adapter payloads"
    )
    return DatasetSnapshotSpec(
        schema_version="tabu.dataset-snapshot.v3",
        dataset_snapshot_id=snapshot_id,
        dataset_id=scenario.dataset.dataset_id,
        source_uri=scenario.dataset.source_uri,
        source_sha256=prepared.source_material.raw_sha256,
        content_sha256=content_sha256,
        license_id=scenario.dataset.license_id,
        split_manifest_sha256=prepared.binding.split_sha256,
        fit_partition=scenario.preprocessing_fit_partition,
        adapter=DatasetAdapter(
            adapter_id=adapter_id,
            adapter_version=adapter_version,
        ),
        episode_recipe_hashes=(prepared.binding.recipe_sha256,),
        evaluation_scenario_id=scenario_id,
        truth_sidecar_sha256=prepared.binding.truth_sidecar_sha256,
        request_sha256=request_sha256,
        authority_sha256=authority_sha256,
        authority_status=DatasetAuthorityStatus.SELF_CONSISTENT_UNREVIEWED,
        review_ids=(),
        mask_boundary=mask_boundary,
        contamination_boundary=(
            "split-before-prepare; selection, preprocessing, normalizers, and codebooks "
            "are fitted on train only"
        ),
    )


def write_dataset_snapshot_manifest(
    snapshot: DatasetSnapshotSpec,
    destination: str | os.PathLike[str],
) -> Path:
    """Write one deterministic create-once catalog manifest.

    Repeating the operation with byte-identical content is idempotent.  A
    different payload at the same destination is rejected.
    """

    target = Path(destination)
    canonical = (
        json.dumps(
            to_canonical_data(snapshot),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_text(encoding="utf-8") != canonical:
            raise FileExistsError(
                f"dataset snapshot manifest already exists with different content: {target}"
            )
        return target

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_text(encoding="utf-8") != canonical:
                raise
        return target
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "EvalSnapshotRegistrationError",
    "dataset_snapshot_from_prepared",
    "write_dataset_snapshot_manifest",
]
