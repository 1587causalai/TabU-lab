#!/usr/bin/env python3
"""Regenerate immutable F0/S1 preregistrations and fit-first JSON Schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from tabu_lab.experiments import (
    AugmentedReadoutGeometry,
    FitExperimentSpec,
    FitFeasibilityReport,
    LabelAddressPlan,
    ReferenceBackendConfig,
    RunAttempt,
)
from tabu_lab.experiments.fixtures import BUILDABLE_CONTRACTS
from tabu_lab.experiments.preregistration import build_f0_preregistration
from tabu_lab.experiments.s1_preregistration import build_s1_preregistration
from tabu_lab.experiments.s1_registry import list_s1_registrations


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_preregistration_once(directory: Path, spec: FitExperimentSpec) -> None:
    """Create a preregistration once; never silently rewrite experiment identity."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "preregistration.yaml"
    text = yaml.safe_dump(
        spec.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    if path.exists():
        try:
            existing = FitExperimentSpec.model_validate(
                yaml.safe_load(path.read_text(encoding="utf-8"))
            )
        except (ValueError, yaml.YAMLError) as exc:
            raise RuntimeError(f"existing preregistration is invalid: {path}") from exc
        if existing != spec:
            raise RuntimeError(
                f"refusing to rewrite immutable preregistration {path}; "
                "register a new experiment/version instead"
            )
        return
    path.write_text(text, encoding="utf-8")


def build_s1_assets(repository: Path) -> None:
    """Create the canonical S1 preregistrations once."""

    experiment_root = repository / "experiments" / "fit-first" / "S1"
    for registration in list_s1_registrations():
        spec = build_s1_preregistration(registration.experiment_id)
        _write_preregistration_once(experiment_root / spec.experiment_id, spec)


def build_assets(
    repository: Path,
    *,
    device: str,
    device_index: int | None,
    reference: ReferenceBackendConfig,
    augmented_readout_geometry: AugmentedReadoutGeometry,
) -> None:
    experiment_root = repository / "experiments" / "fit-first" / "F0"
    for contract_id in BUILDABLE_CONTRACTS:
        geometry = (
            augmented_readout_geometry
            if contract_id in {"tabuf", "tabul", "tabufl", "tabu4rec"}
            else None
        )
        spec = build_f0_preregistration(
            contract_id,
            device=device,
            device_index=device_index,
            reference=reference,
            augmented_readout_geometry=geometry,
        )
        directory = experiment_root / spec.experiment_id
        _write_preregistration_once(directory, spec)

    # The original seven v1 preregistrations remain immutable historical
    # inputs.  Completion v2 is a separately named, representation-
    # identifiable repair; it is generated in addition to, never over, v1.
    for contract_id in ("tabuf", "tabu.unit_row", "tabu.unit_pair"):
        geometry = augmented_readout_geometry if contract_id == "tabuf" else None
        v2_reference = (
            reference.model_copy(
                update={
                    "geometry_normalization": "rms_unit",
                    "routing_bandwidth": 2.5,
                }
            )
            if contract_id == "tabu.unit_row"
            else reference
        )
        spec = build_f0_preregistration(
            contract_id,
            device=device,
            device_index=device_index,
            reference=v2_reference,
            augmented_readout_geometry=geometry,
            fixture_version="v2",
        )
        directory = experiment_root / spec.experiment_id
        _write_preregistration_once(directory, spec)

    graph_v2 = build_f0_preregistration(
        "tabu4graph",
        device=device,
        device_index=device_index,
        reference=reference,
        fixture_version="v2",
    )
    graph_v2_directory = experiment_root / graph_v2.experiment_id
    _write_preregistration_once(graph_v2_directory, graph_v2)

    for contract_id in ("tabul", "tabufl"):
        supervised_v2 = build_f0_preregistration(
            contract_id,
            device=device,
            device_index=device_index,
            reference=reference,
            augmented_readout_geometry=augmented_readout_geometry,
            fixture_version="v2",
        )
        supervised_v2_directory = experiment_root / supervised_v2.experiment_id
        _write_preregistration_once(supervised_v2_directory, supervised_v2)

        supervised_v3 = build_f0_preregistration(
            contract_id,
            device=device,
            device_index=device_index,
            reference=reference,
            augmented_readout_geometry=augmented_readout_geometry,
            supervised_label_address_plan=(LabelAddressPlan.PREDICTOR_UNIT_LINKED_PER_LABEL_V2),
            fixture_version="v2",
        )
        supervised_v3_directory = experiment_root / supervised_v3.experiment_id
        _write_preregistration_once(supervised_v3_directory, supervised_v3)

    # F0-016 established that the v3 TabUFL architecture could fit its L
    # ledger but exposed an asymmetric, unstable F target schedule.  v4 keeps
    # that architecture fixed and registers a separately named, balanced
    # support-realizable 3 Features x 4 latent-level F ledger.
    tabufl_v4 = build_f0_preregistration(
        "tabufl",
        device=device,
        device_index=device_index,
        reference=reference,
        augmented_readout_geometry=augmented_readout_geometry,
        fixture_version="v4",
    )
    tabufl_v4_directory = experiment_root / tabufl_v4.experiment_id
    _write_preregistration_once(tabufl_v4_directory, tabufl_v4)

    # F0-017 was a useful three-seed diagnostic but registered only 12 F
    # targets.  F0-018 restores the frozen 16-F/32-L contract under a new,
    # immutable experiment and generator identity.
    tabufl_v5 = build_f0_preregistration(
        "tabufl",
        device=device,
        device_index=device_index,
        reference=reference,
        augmented_readout_geometry=augmented_readout_geometry,
        fixture_version="v5",
    )
    _write_preregistration_once(experiment_root / tabufl_v5.experiment_id, tabufl_v5)

    # TabUBase 0.2.0 owns three independent F0 assets: completion, single-
    # response regression, and single-response classification.
    for base_version in ("supervised_regression", "supervised_classification"):
        base_spec = build_f0_preregistration(
            "tabu.cell.base",
            device=device,
            device_index=device_index,
            reference=reference,
            fixture_version=base_version,
        )
        _write_preregistration_once(experiment_root / base_spec.experiment_id, base_spec)

    rec_v2 = build_f0_preregistration(
        "tabu4rec",
        device=device,
        device_index=device_index,
        reference=reference,
        augmented_readout_geometry=augmented_readout_geometry,
        fixture_version="v2",
    )
    rec_v2_directory = experiment_root / rec_v2.experiment_id
    _write_preregistration_once(rec_v2_directory, rec_v2)

    # S1 execution is frozen independently of command-line F0 diagnostics:
    # canonical preregistrations always use deterministic CUDA:0.
    build_s1_assets(repository)

    schema_types = {
        "fit-experiment.schema.json": FitExperimentSpec,
        "fit-feasibility.schema.json": FitFeasibilityReport,
        "run-attempt.schema.json": RunAttempt,
    }
    for filename, schema_type in schema_types.items():
        _write_json(repository / "schemas" / filename, schema_type.model_json_schema())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--inducing-slots", type=int, default=4)
    parser.add_argument("--matched-slots", type=int, default=4)
    parser.add_argument(
        "--geometry-normalization",
        choices=("none", "rms_unit"),
        default="none",
    )
    parser.add_argument("--routing-bandwidth", type=float, default=1.0)
    parser.add_argument(
        "--augmented-readout-geometry",
        choices=tuple(AugmentedReadoutGeometry),
        default=AugmentedReadoutGeometry.MATCHED_UF,
    )
    args = parser.parse_args()
    device_index = args.device_index if args.device == "cuda" else None
    reference = ReferenceBackendConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_blocks=args.n_blocks,
        inducing_slots=args.inducing_slots,
        matched_slots=args.matched_slots,
        geometry_normalization=args.geometry_normalization,
        routing_bandwidth=args.routing_bandwidth,
    )
    build_assets(
        args.repository.resolve(),
        device=args.device,
        device_index=device_index,
        reference=reference,
        augmented_readout_geometry=AugmentedReadoutGeometry(args.augmented_readout_geometry),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
