from __future__ import annotations

import pytest
from pydantic import ValidationError

from tabu_lab.evidence import EnvironmentDisclosure


def _environment(**overrides: str | None) -> EnvironmentDisclosure:
    payload: dict[str, str | None] = {
        "environment_hash": "a" * 64,
        "host_class": "mps-host",
        "operating_system": "Darwin",
        "device": "mps",
        "architecture": "arm64",
        "accelerator": "Apple Metal Performance Shaders",
        "python_version": "3.11.14",
    }
    payload.update(overrides)
    return EnvironmentDisclosure.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("host_class", "gongqian-mini"),
        ("operating_system", "username=cms"),
        ("operating_system", "Darwin /Users/cms/build"),
        ("architecture", "hostname=gongqian-mini"),
        ("device", "cuda:/private/var/run"),
        ("accelerator", "Bearer abcdefghijklmnopqrstuvwxyz"),
        ("python_version", "3.11.14 /opt/private/python"),
    ),
)
def test_environment_disclosure_rejects_private_or_secret_strings(
    field_name: str, value: str
) -> None:
    with pytest.raises(ValidationError, match=r"host identity|secret|absolute local path|public"):
        _environment(**{field_name: value})


@pytest.mark.parametrize("field_name", ("hostname", "username", "python_executable"))
def test_environment_disclosure_rejects_private_extra_fields(field_name: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _environment(**{field_name: "private-machine-value"})


@pytest.mark.parametrize(
    "overrides",
    (
        {"operating_system": "Linux", "architecture": "x86_64", "device": "cpu"},
        {"operating_system": "Windows", "architecture": "AMD64", "device": "cuda:0"},
        {
            "host_class": "cuda-host",
            "accelerator": "NVIDIA GeForce RTX 4090",
            "python_version": "3.13.0rc2",
        },
    ),
)
def test_environment_disclosure_accepts_generalized_cli_capture(
    overrides: dict[str, str]
) -> None:
    assert _environment(**overrides)
