from __future__ import annotations

import pytest

from tabu_lab.experiments.tabubase_classification_r0 import select_global_r0_schedule
from tabu_lab.experiments.tabubase_scale import ROOT_SEEDS


def _payload() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for dataset_id, preferred_lr in (("easy", 1.0e-4), ("hard", 3.0e-4)):
        for seed in ROOT_SEEDS:
            for arm in ("pretrained", "scratch"):
                for learning_rate in (1.0e-4, 3.0e-4):
                    for updates in (400, 1_200):
                        penalty = 0.0 if learning_rate == preferred_lr else 0.2
                        update_penalty = 0.0 if updates == 400 else 0.1
                        records.append(
                            {
                                "dataset_id": dataset_id,
                                "seed": seed,
                                "arm": arm,
                                "learning_rate": learning_rate,
                                "updates": updates,
                                "validation_metrics": {"log_loss": 0.3 + penalty + update_penalty},
                            }
                        )
    return {
        "selection_partition": "validation",
        "test_evaluations": 0,
        "source_tree_sha256": "fixture-source",
        "records": records,
    }


def test_r0_selection_is_dataset_macro_paired_and_validation_only() -> None:
    result = select_global_r0_schedule([_payload()])
    assert result["selection_partition"] == "validation"
    assert result["test_evaluations"] == 0
    # The learning-rate candidates tie after dataset macro-averaging, so the
    # deterministic final key selects the lower learning rate and 400 updates.
    assert result["selected"]["learning_rate"] == 1.0e-4
    assert result["selected"]["updates"] == 400


def test_r0_selection_rejects_test_access_and_source_drift() -> None:
    contaminated = _payload()
    contaminated["test_evaluations"] = 1
    with pytest.raises(ValueError, match="evaluated test rows"):
        select_global_r0_schedule([contaminated])

    first = _payload()
    second = _payload()
    second["source_tree_sha256"] = "different-source"
    with pytest.raises(ValueError, match="one exact source tree"):
        select_global_r0_schedule([first, second])
