from __future__ import annotations

import numpy as np

from tabu_lab.experiments.scm_missingness import (
    SCM_MISSINGNESS_FAMILIES,
    SCMMissingnessManifest,
    apply_scm_missingness,
    sample_scm_missingness_manifest,
)


def test_all_scm_missingness_families_emit_real_nan_without_mutating_complete_table() -> None:
    complete = np.random.default_rng(1729).normal(size=(128, 7))
    baseline = complete.copy()
    for index, family in enumerate(SCM_MISSINGNESS_FAMILIES):
        manifest = SCMMissingnessManifest(
            family=family,
            rate=0.2,
            mechanism_seed=1000 + index,
        )
        result = apply_scm_missingness(
            complete,
            manifest,
            eligible_columns=tuple(range(6)),
            driver_columns=tuple(range(6)),
        )
        replay = apply_scm_missingness(
            complete,
            manifest,
            eligible_columns=tuple(range(6)),
            driver_columns=tuple(range(6)),
        )
        assert np.array_equal(complete, baseline)
        assert result.missing_count > 0
        assert np.array_equal(np.isnan(result.raw_values), result.missing_mask)
        assert np.array_equal(result.missing_mask, replay.missing_mask)
        assert not bool(result.missing_mask[:, -1].any())
        assert abs(float(result.missing_mask[:, :6].mean()) - 0.2) < 0.08
        if family.value == "mar":
            assert int((~result.missing_mask[:, :6].any(axis=0)).sum()) >= 2


def test_scm_missingness_manifest_sampling_is_world_deterministic_and_identified() -> None:
    manifest = sample_scm_missingness_manifest(
        root_seed=2718, world_id="world-a", partition="train"
    )
    replay = sample_scm_missingness_manifest(
        root_seed=2718, world_id="world-a", partition="train"
    )
    other = sample_scm_missingness_manifest(
        root_seed=2718, world_id="world-b", partition="train"
    )
    assert manifest == replay
    assert manifest.manifest_hash == replay.manifest_hash
    assert manifest.manifest_hash != other.manifest_hash
    assert manifest.as_dict()["component_id"] == "tabur.scm-missingness.v1"
