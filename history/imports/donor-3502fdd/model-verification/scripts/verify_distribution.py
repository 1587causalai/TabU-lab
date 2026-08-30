#!/usr/bin/env python3
"""Fail closed when release archives omit public audit material or unsafe artifacts."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

MODEL_IDS = (
    "tabu.unit_pair",
    "tabu.unit_row",
    "tabu4do",
    "tabu4graph",
    "tabu4rec",
    "tabuf",
    "tabufl",
    "tabul",
    "tabu.cell.base",
    "tabu.cell.column",
    "tabu.cell.rec",
    "tabu.cell.row",
    "tabu.cell.row_column",
)
MODEL_HISTORY = (("tabu4rec", "0.2.0"),)
UNSAFE_SUFFIXES = (".bin", ".ckpt", ".dill", ".joblib", ".pickle", ".pkl", ".pt", ".pth")


def _exactly_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one {pattern!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {member.rstrip("/") for member in archive.namelist() if member.rstrip("/")}


def _sdist_members(path: Path) -> set[str]:
    with tarfile.open(path, mode="r:gz") as archive:
        raw = {member.name.rstrip("/") for member in archive.getmembers() if member.name}
    roots = {member.split("/", maxsplit=1)[0] for member in raw}
    if len(roots) != 1:
        raise SystemExit(f"sdist must have exactly one archive root, found {sorted(roots)}")
    root = roots.pop()
    return {
        member.removeprefix(f"{root}/")
        for member in raw
        if member != root and member.startswith(f"{root}/")
    }


def _require(members: set[str], required: set[str], *, label: str) -> None:
    missing = sorted(required - members)
    if missing:
        rendered = "\n".join(f"- {member}" for member in missing)
        raise SystemExit(f"{label} is missing required public material:\n{rendered}")


def _reject_unsafe(members: set[str], *, label: str) -> None:
    unsafe = sorted(member for member in members if member.lower().endswith(UNSAFE_SUFFIXES))
    if unsafe:
        rendered = "\n".join(f"- {member}" for member in unsafe)
        raise SystemExit(f"{label} contains unsafe serialized artifacts:\n{rendered}")


def _wheel_required_members() -> set[str]:
    package_source = Path(__file__).resolve().parents[1] / "src" / "tabu_lab"
    return {
        path.relative_to(package_source.parent).as_posix()
        for path in package_source.rglob("*")
        if path.is_file() and (path.suffix in {".py", ".yaml"} or path.name == "py.typed")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()

    wheel = _exactly_one(args.dist_dir, "*.whl")
    sdist = _exactly_one(args.dist_dir, "*.tar.gz")
    wheel_members = _wheel_members(wheel)
    sdist_members = _sdist_members(sdist)

    wheel_required = _wheel_required_members()
    expected_specs = {f"tabu_lab/specs/models/{model_id}.yaml" for model_id in MODEL_IDS}
    expected_specs.update(
        f"tabu_lab/specs/models/{model_id}/{version}.yaml"
        for model_id, version in MODEL_HISTORY
    )
    _require(wheel_required, expected_specs, label="package source")
    _require(wheel_members, wheel_required, label="wheel")
    if not any(member.endswith(".dist-info/licenses/LICENSE") for member in wheel_members):
        raise SystemExit("wheel is missing its Apache-2.0 license file")

    sdist_required = {
        ".github/workflows/ci.yml",
        ".github/workflows/release-candidate.yml",
        ".github/workflows/security.yml",
        "EVIDENCE_LEDGER.md",
        "MIGRATION_PROVENANCE.md",
        "ROADMAP.md",
        "docs/governance/BRANCH_RULESET.md",
        "docs/reports/README.md",
        "experiments/G000-tabuf-artificial-mask/preregistration.yaml",
        "pyproject.toml",
        "scripts/smoke_installed_wheel.py",
        "scripts/verify_action_pins.py",
        "scripts/verify_distribution.py",
        "schemas/claim.schema.json",
        "schemas/model-spec.schema.json",
        "schemas/preregistration.schema.json",
        "schemas/receipt.schema.json",
        "tests/unit/test_registry.py",
        "uv.lock",
        *(f"specs/models/{model_id}.yaml" for model_id in MODEL_IDS),
        *(f"specs/models/{model_id}/{version}.yaml" for model_id, version in MODEL_HISTORY),
    }
    _require(sdist_members, sdist_required, label="sdist")
    _reject_unsafe(wheel_members, label="wheel")
    _reject_unsafe(sdist_members, label="sdist")
    print(
        "PASS: wheel runtime payload and sdist public audit boundary are complete "
        f"({len(wheel_members)} wheel members, {len(sdist_members)} sdist members)"
    )


if __name__ == "__main__":
    main()
