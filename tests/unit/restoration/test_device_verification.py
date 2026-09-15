"""Device entry-point routing and fail-closed checks run without a GPU."""

import json

import pytest
import torch

from tabu_lab.cli import main
from tabu_lab.models.restoration import device_verification, verification


def test_cuda_unavailable_records_failure_without_model_or_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_verification, "_probes", lambda *_: pytest.fail("ran CUDA probes"))
    monkeypatch.setattr(verification, "verify_model", lambda: pytest.fail("CPU fallback"))
    output = tmp_path / "unavailable.json"
    assert main(["restoration", "verify", "--device", "cuda:0", "--output", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["outcome"] == "failed"
    assert result["error"]["category"] == "cuda_unavailable"
    assert result["device"] == "cuda:0"
    assert result["checks"] == []
    assert len(result["component_source_sha256"]) == 64
    assert str(tmp_path) not in output.read_text()
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        main(["restoration", "verify", "--device", "cuda:0", "--output", str(output)])
    assert output.read_bytes() == before


@pytest.mark.parametrize("device_args", [[], ["--device", "cpu"]])
@pytest.mark.parametrize("components", [False, True])
def test_cpu_routes_to_existing_verifier(monkeypatch, device_args, components):
    calls = []
    for name in ("verify_components", "verify_model"):
        monkeypatch.setattr(verification, name,
                            lambda name=name: calls.append(name) or {"outcome": "passed"})
    monkeypatch.setattr(device_verification, "verify_device", lambda *_: pytest.fail("CUDA path"))
    args = ["restoration", "verify", *device_args]
    if components:
        args.append("--components-only")
    assert main(args) == 0
    assert calls == ["verify_components" if components else "verify_model"]


def test_cuda_routes_to_device_verifier(monkeypatch):
    calls = []
    monkeypatch.setattr(device_verification, "verify_device",
                        lambda device: calls.append(device) or {"outcome": "passed"})
    assert main(["restoration", "verify", "--device", "cuda:0"]) == 0
    assert calls == ["cuda:0"]
    with pytest.raises(SystemExit):
        main(["restoration", "verify", "--device", "cuda:0", "--components-only"])


def test_failure_sanitizes_exception_details(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def fail():
        raise RuntimeError("private /home/example/secret path")

    monkeypatch.setattr(torch.cuda, "is_available", fail)
    result = device_verification.verify_device()
    assert result["outcome"] == "failed"
    assert result["error"] == {"category": "RuntimeError", "stage": "cuda_availability"}
    assert "secret" not in json.dumps(result)


def test_continuation_comparison_is_stricter_than_device_tolerance():
    left = torch.tensor([1.0], dtype=torch.float64)
    pairs = [("parameter", left, left + 1e-10)]
    assert device_verification._compare("parity", pairs)["outcome"] == "passed"
    exact = device_verification._compare("continuation", pairs, exact=True)
    assert exact["outcome"] == "failed"
    assert exact["mismatches"] == ["parameter"]
    assert exact["max_absolute_differences"]["parameter"] > 0


def test_gradient_presence_mismatch_fails():
    result = device_verification._compare("gradients", [("weight", None, torch.ones(1))])
    assert result["outcome"] == "failed"
    assert result["mismatches"] == ["weight"]
