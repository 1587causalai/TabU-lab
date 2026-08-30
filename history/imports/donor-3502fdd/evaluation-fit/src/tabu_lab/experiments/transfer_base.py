"""Versioned TabUBase transfer schemas and fail-closed loaders.

The historical ``transfer-v1`` files remain TabUL-only.  This module owns the
Base 0.2.0 profile line and refuses to load a mismatched contract/profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tabu_lab.models import TabUCellBaseProfile


@dataclass(frozen=True, slots=True)
class BaseTransferRef:
    contract_id: str = "tabu.cell.base"
    contract_version: str = "0.2.0"
    profile_id: str = TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1.value

    def validate(self) -> BaseTransferRef:
        if self.contract_id != "tabu.cell.base" or self.contract_version != "0.2.0":
            raise ValueError("transfer-base-v1 requires tabu.cell.base@0.2.0")
        if self.profile_id != TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1.value:
            raise ValueError("transfer-base-v1 is restricted to supervised.label_broadcast.v1")
        return self


@dataclass(frozen=True, slots=True)
class BasePretrainSpec:
    reference: BaseTransferRef
    worlds: int
    updates: int
    seeds: tuple[int, ...]
    heldout_families: tuple[str, ...]
    checkpoints: tuple[int, ...]
    pilot_worlds: int = 2_048
    pilot_updates: int = 2_000
    pilot_seeds: tuple[int, ...] = (1729,)
    pilot_gates: tuple[str, ...] = (
        "finite",
        "exact_resume",
        "validation_improvement",
    )
    predictor_kinds: tuple[str, ...] = ("numeric", "ordinal", "nominal")
    world_split_before_masking: bool = True
    world_split_before_statistics: bool = True
    category_threshold_source: str = "generator_or_train_context_only"

    def validate(self) -> BasePretrainSpec:
        self.reference.validate()
        if self.worlds <= 0 or self.updates <= 0 or not self.seeds:
            raise ValueError("pretraining worlds, updates and seeds must be positive")
        if (self.pilot_worlds, self.pilot_updates) != (2_048, 2_000):
            raise ValueError("PT-S0 is frozen to 2,048 worlds and 2,000 updates")
        if len(self.pilot_seeds) != 1:
            raise ValueError("PT-S0 uses exactly one seed")
        if self.pilot_gates != ("finite", "exact_resume", "validation_improvement"):
            raise ValueError("PT-S0 gates must be finite, exact-resume, and validation-improvement")
        if self.predictor_kinds != ("numeric", "ordinal", "nominal"):
            raise ValueError("Base pretraining must cover numeric, ordinal, and nominal predictors")
        if not self.world_split_before_masking or not self.world_split_before_statistics:
            raise ValueError("world split must precede masking and statistic estimation")
        if self.category_threshold_source != "generator_or_train_context_only":
            raise ValueError("category thresholds must use generator or train/context evidence")
        if 20_000 in self.checkpoints and self.updates < 20_000:
            raise ValueError("20k checkpoint requires the 20k update budget")
        return self


@dataclass(frozen=True, slots=True)
class BaseIclSpec:
    reference: BaseTransferRef
    context_sizes: tuple[int, ...]
    heldout_worlds: int
    arms: tuple[str, ...]

    def validate(self) -> BaseIclSpec:
        self.reference.validate()
        if self.context_sizes != (0, 1, 2, 4, 8, 16, 32):
            raise ValueError("ICL context sizes must be the frozen K grid")
        required = {"pretrained_frozen", "random_init_frozen", "pretrained_shuffled", "scratch_finetune"}
        if not required.issubset(self.arms):
            raise ValueError("ICL arms are incomplete")
        return self


@dataclass(frozen=True, slots=True)
class BaseFineTuneTask:
    task_id: str
    dataset_id: str
    label_budgets: tuple[int, ...]
    primary_budget: int

    def validate(self) -> BaseFineTuneTask:
        if not self.task_id or not self.dataset_id or not self.label_budgets:
            raise ValueError("fine-tune task identity and budgets are required")
        if any(budget <= 0 for budget in self.label_budgets):
            raise ValueError("fine-tune label budgets must be positive")
        if self.primary_budget not in self.label_budgets:
            raise ValueError("fine-tune primary budget must be one of label_budgets")
        return self


@dataclass(frozen=True, slots=True)
class BaseFineTuneSpec:
    reference: BaseTransferRef
    initialization_arms: tuple[str, ...]
    tasks: tuple[BaseFineTuneTask, ...]
    seeds: tuple[int, ...]
    learning_rates: tuple[float, ...]
    update_budgets: tuple[int, ...]

    def validate(self) -> BaseFineTuneSpec:
        self.reference.validate()
        if self.initialization_arms != ("pretrained_s1", "scratch"):
            raise ValueError("Base fine-tune requires pretrained_s1 and scratch arms")
        if {task.task_id for task in self.tasks} != {"adult_classification", "diabetes_regression"}:
            raise ValueError("Base fine-tune requires Adult classification and Diabetes regression")
        for task in self.tasks:
            task.validate()
        if len(self.seeds) != 3 or not all(seed >= 0 for seed in self.seeds):
            raise ValueError("Base fine-tune requires three non-negative seeds")
        if self.learning_rates != (1.0e-4, 3.0e-4, 1.0e-3):
            raise ValueError("R0 learning-rate candidates are frozen")
        if self.update_budgets != (400, 1_200):
            raise ValueError("R0 update candidates are frozen")
        return self


def _ref(payload: dict[str, Any]) -> BaseTransferRef:
    value = BaseTransferRef(
        contract_id=str(payload.get("contract_id", "")),
        contract_version=str(payload.get("contract_version", "")),
        profile_id=str(payload.get("profile_id", "")),
    )
    return value.validate()


def load_pretrain_spec(path: str | Path) -> BasePretrainSpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spec = BasePretrainSpec(
        reference=_ref(payload["model"]),
        worlds=int(payload["worlds"]),
        updates=int(payload["updates"]),
        seeds=tuple(int(seed) for seed in payload["seeds"]),
        heldout_families=tuple(str(item) for item in payload["heldout_families"]),
        checkpoints=tuple(int(item) for item in payload["checkpoints"]),
        pilot_worlds=int(payload.get("pilot", {}).get("worlds", 2_048)),
        pilot_updates=int(payload.get("pilot", {}).get("updates", 2_000)),
        pilot_seeds=tuple(int(seed) for seed in payload.get("pilot", {}).get("seeds", (1729,))),
        pilot_gates=tuple(str(item) for item in payload.get("pilot", {}).get("gates", (
            "finite", "exact_resume", "validation_improvement"
        ))),
        predictor_kinds=tuple(str(item) for item in payload.get(
            "predictor_kinds", ("numeric", "ordinal", "nominal")
        )),
        world_split_before_masking=bool(payload.get("world_split_before_masking", True)),
        world_split_before_statistics=bool(payload.get("world_split_before_statistics", True)),
        category_threshold_source=str(payload.get(
            "category_threshold_source", payload.get("label_threshold_source", "")
        )),
    )
    return spec.validate()


def load_icl_spec(path: str | Path) -> BaseIclSpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spec = BaseIclSpec(
        reference=_ref(payload["model"]),
        context_sizes=tuple(int(item) for item in payload["context_sizes"]),
        heldout_worlds=int(payload["heldout_worlds"]),
        arms=tuple(str(item) for item in payload["arms"]),
    )
    return spec.validate()


def load_finetune_spec(path: str | Path) -> BaseFineTuneSpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    schedule = payload["schedule_selection"]
    spec = BaseFineTuneSpec(
        reference=_ref(payload["model"]),
        initialization_arms=tuple(str(item) for item in payload["initialization_arms"]),
        tasks=tuple(
            BaseFineTuneTask(
                task_id=str(item["task_id"]),
                dataset_id=str(item["dataset_id"]),
                label_budgets=tuple(int(budget) for budget in item["label_budgets"]),
                primary_budget=int(item["primary_budget"]),
            )
            for item in payload["tasks"]
        ),
        seeds=tuple(int(seed) for seed in payload["seeds"]),
        learning_rates=tuple(float(rate) for rate in schedule["learning_rates"]),
        update_budgets=tuple(int(updates) for updates in schedule["updates"]),
    )
    return spec.validate()


__all__ = [
    "BaseFineTuneSpec",
    "BaseFineTuneTask",
    "BaseIclSpec",
    "BasePretrainSpec",
    "BaseTransferRef",
    "load_icl_spec",
    "load_finetune_spec",
    "load_pretrain_spec",
]
