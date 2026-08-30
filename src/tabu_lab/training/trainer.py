"""Deterministic training and fail-closed exact resume for reference models."""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tabu_lab.contracts import (
    LossBundle,
    PredictionBundle,
    TruthSidecar,
    canonical_hash,
)
from tabu_lab.evidence import RunIdentity

from .objective import Objective


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    raise TypeError(f"checkpoint metadata contains unsupported value {type(value).__name__}")


def _deep_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_deep_tuple(item) for item in value)
    return value


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


@dataclass(frozen=True)
class TrainStep:
    step: int
    prediction: PredictionBundle
    loss: LossBundle
    gradient_norm: float
    gradient_norms: Mapping[str, float]


def _gradient_group(parameter_name: str) -> str:
    """Map an implementation parameter to a stable fit-diagnostic group."""

    root = parameter_name.split(".", 1)[0]
    if root in {"unit_seeds", "feature_seeds"}:
        return "carrier"
    if root == "label_dynamics":
        return "dynamics"
    if root == "label_readout":
        return "readout"
    if root in {"tokenizer", "dynamics", "readout"}:
        return root
    return "other"


class Trainer:
    """Optimizer owner with explicit evidence, identity, and RNG boundaries.

    Training without a run identity is allowed for local exploration. Exporting
    or resuming a checkpoint is not: exact resume requires a validated
    ``RunIdentity``, canonical training/execution configs, and named episode and
    sampler generators.
    """

    checkpoint_version = "tabu.training-checkpoint.v3"
    model_state_version = "tabu.model-state.v1"
    optimizer_state_version = "tabu.optimizer-state.v1"
    rng_state_version = "tabu.rng-state.v2"
    _required_named_generators = frozenset({"episode", "sampler"})

    def __init__(
        self,
        model: nn.Module,
        *,
        objective: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        learning_rate: float = 1.0e-3,
        max_gradient_norm: float | None = None,
        run_identity: RunIdentity | None = None,
        training_config: Mapping[str, Any] | None = None,
        execution_config: Mapping[str, Any] | None = None,
        named_generators: Mapping[str, torch.Generator] | None = None,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if max_gradient_norm is not None and max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive when provided")
        self.model = model
        self.objective = objective or Objective()
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate)
        )
        self.max_gradient_norm = max_gradient_norm
        self.step = 0
        self.run_identity: RunIdentity | None = None
        self.training_config: dict[str, Any] | None = None
        self.execution_config: dict[str, Any] | None = None
        self.named_generators: dict[str, torch.Generator] = {}
        if run_identity is None:
            if training_config is not None or execution_config is not None:
                raise ValueError("training/execution configs require a RunIdentity")
            if named_generators is not None:
                self.named_generators = self._validate_named_generators(
                    named_generators, identity=None
                )
        else:
            self._bind_resume_identity(
                run_identity,
                training_config=training_config,
                execution_config=execution_config,
                named_generators=named_generators,
            )

    @property
    def model_id(self) -> str:
        return str(getattr(self.model, "model_id", type(self.model).__name__))

    @property
    def contract_version(self) -> str:
        value = getattr(self.model, "contract_version", None)
        if not isinstance(value, str) or not value:
            raise ValueError("checkpointable model must expose contract_version")
        return value

    def _validate_named_generators(
        self,
        generators: Mapping[str, torch.Generator],
        *,
        identity: RunIdentity | None,
    ) -> dict[str, torch.Generator]:
        normalized: dict[str, torch.Generator] = {}
        for name, generator in generators.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("named generator keys must be non-empty strings")
            if not isinstance(generator, torch.Generator):
                raise TypeError("named generator values must be torch.Generator instances")
            key = name.strip()
            if key in normalized:
                raise ValueError(f"duplicate normalized generator name: {key!r}")
            if identity is not None:
                if key not in identity.seeds:
                    raise ValueError(f"generator {key!r} is absent from RunIdentity.seeds")
                if generator.initial_seed() != identity.seeds[key]:
                    raise ValueError(f"generator {key!r} seed does not match RunIdentity")
            normalized[key] = generator
        return dict(sorted(normalized.items()))

    def _bind_resume_identity(
        self,
        identity: RunIdentity,
        *,
        training_config: Mapping[str, Any] | None,
        execution_config: Mapping[str, Any] | None,
        named_generators: Mapping[str, torch.Generator] | None,
    ) -> None:
        if not isinstance(identity, RunIdentity):
            raise TypeError("run_identity must be a validated RunIdentity")
        if training_config is None or execution_config is None:
            raise ValueError(
                "exact-resume identity requires training_config and execution_config"
            )
        resolved_training = dict(training_config)
        resolved_execution = dict(execution_config)
        if canonical_hash(resolved_training) != identity.training_config_hash:
            raise ValueError("training_config hash does not match RunIdentity")
        if canonical_hash(resolved_execution) != identity.execution_config_hash:
            raise ValueError("execution_config hash does not match RunIdentity")
        semantic_hash = getattr(self.model, "semantic_config_hash", None)
        if semantic_hash is None:
            semantic_hash = getattr(
                getattr(self.model, "config", None), "semantic_hash", None
            )
        if semantic_hash != identity.semantic_config_hash:
            raise ValueError("model semantic config hash does not match RunIdentity")
        missing_seed_names = self._required_named_generators - set(identity.seeds)
        if missing_seed_names:
            raise ValueError(
                "RunIdentity.seeds is missing exact-resume generators: "
                + ", ".join(sorted(missing_seed_names))
            )
        if named_generators is None:
            generated: dict[str, torch.Generator] = {}
            for name in sorted(self._required_named_generators):
                generated[name] = torch.Generator(device="cpu").manual_seed(
                    identity.seeds[name]
                )
            named_generators = generated
        resolved_generators = self._validate_named_generators(
            named_generators, identity=identity
        )
        if not self._required_named_generators.issubset(resolved_generators):
            raise ValueError("named_generators must include episode and sampler")
        self.run_identity = identity
        self.training_config = resolved_training
        self.execution_config = resolved_execution
        self.named_generators = resolved_generators

    def train_step(self, evidence: Any, truth: TruthSidecar) -> TrainStep:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        first_parameter = next(self.model.parameters(), None)
        execution_device = (
            first_parameter.device if first_parameter is not None else torch.device("cpu")
        )
        execution_evidence = (
            evidence.to(execution_device) if hasattr(evidence, "to") else evidence
        )
        execution_truth = truth.to(execution_device)
        prediction = self.model(execution_evidence)
        if not isinstance(prediction, PredictionBundle):
            raise TypeError("trainable model forward must return PredictionBundle")
        loss = self.objective(prediction, execution_truth)
        if not isinstance(loss, LossBundle):
            raise TypeError("objective must return LossBundle")
        loss.total.backward()
        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
        ]
        parameters = [parameter for _, parameter in named_parameters]
        squared_by_group = {
            "carrier": 0.0,
            "tokenizer": 0.0,
            "dynamics": 0.0,
            "readout": 0.0,
            "other": 0.0,
        }
        for name, parameter in named_parameters:
            squared_by_group[_gradient_group(name)] += float(
                parameter.grad.detach().float().square().sum().item()
            )
        gradient_norms = {
            name: math.sqrt(squared)
            for name, squared in sorted(squared_by_group.items())
        }
        if self.max_gradient_norm is None:
            gradient_norm = math.sqrt(sum(squared_by_group.values()))
        else:
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(parameters, self.max_gradient_norm).item()
            )
        self.optimizer.step()
        self.step += 1
        return TrainStep(
            step=self.step,
            prediction=prediction,
            loss=loss,
            gradient_norm=gradient_norm,
            gradient_norms=gradient_norms,
        )

    def fit(
        self,
        batches: Iterable[tuple[Any, TruthSidecar]],
        *,
        epochs: int = 1,
    ) -> tuple[TrainStep, ...]:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        materialized = tuple(batches)
        if not materialized:
            raise ValueError("fit needs at least one evidence/sidecar batch")
        history: list[TrainStep] = []
        for _ in range(epochs):
            for evidence, truth in materialized:
                history.append(self.train_step(evidence, truth))
        return tuple(history)

    def _resume_contract(self) -> dict[str, Any]:
        if (
            self.run_identity is None
            or self.training_config is None
            or self.execution_config is None
        ):
            raise ValueError("checkpoint export/resume requires a bound RunIdentity")
        model_spec_hash = getattr(self.model, "model_spec_hash", None)
        if not isinstance(model_spec_hash, str) or len(model_spec_hash) != 64:
            raise ValueError("checkpointable model must expose model_spec_hash")
        semantic_hash = getattr(self.model, "semantic_config_hash", None)
        if semantic_hash is None:
            semantic_hash = getattr(
                getattr(self.model, "config", None), "semantic_hash", None
            )
        if not isinstance(semantic_hash, str) or len(semantic_hash) != 64:
            raise ValueError("checkpointable model must expose semantic config hash")
        contract = {
            "checkpoint_schema_version": self.checkpoint_version,
            "contract_version": self.contract_version,
            "execution_config_hash": self.run_identity.execution_config_hash,
            "model_id": self.model_id,
            "model_spec_hash": model_spec_hash,
            "model_state_schema_version": self.model_state_version,
            "objective_config": _json_safe(
                getattr(self.objective, "resume_config", {})
            ),
            "objective_type": _qualified_type(self.objective),
            "optimizer_defaults": _json_safe(self.optimizer.defaults),
            "optimizer_state_schema_version": self.optimizer_state_version,
            "optimizer_type": _qualified_type(self.optimizer),
            "optimizer_version": str(torch.__version__),
            "rng_state_schema_version": self.rng_state_version,
            "run_id": self.run_identity.run_id,
            "run_identity_hash": self.run_identity.identity_hash,
            "semantic_config_hash": semantic_hash,
            "training_config_hash": self.run_identity.training_config_hash,
        }
        identity_builder = getattr(self.model, "checkpoint_identity", None)
        if callable(identity_builder):
            identity = identity_builder()
            if not isinstance(identity, Mapping):
                raise ValueError("model checkpoint_identity must return a mapping")
            contract["model_identity"] = _json_safe(dict(identity))
        return contract

    def _rng_state(self) -> dict[str, Any]:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - torch environments include numpy
            raise RuntimeError("exact resume requires NumPy RNG state support") from exc

        numpy_state = np.random.get_state()
        cuda_available = torch.cuda.is_available()
        cuda_states = tuple(torch.cuda.get_rng_state_all()) if cuda_available else ()
        mps_available = torch.backends.mps.is_available()
        mps_state = torch.mps.get_rng_state() if mps_available else None
        return {
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda_available": cuda_available,
            "torch_cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "torch_cuda": cuda_states,
            "torch_mps_available": mps_available,
            "torch_mps": mps_state,
            "python": random.getstate(),
            "numpy": {
                "algorithm": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].astype("int64", copy=True)),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "named": {
                name: generator.get_state()
                for name, generator in self.named_generators.items()
            },
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Return every state component required for a deterministic resume."""

        contract = self._resume_contract()
        assert self.run_identity is not None
        assert self.training_config is not None
        assert self.execution_config is not None
        return {
            "schema": self.checkpoint_version,
            "resume_contract": contract,
            "run_identity": self.run_identity.model_dump(mode="json"),
            "training_config": dict(self.training_config),
            "execution_config": dict(self.execution_config),
            "step": self.step,
            "model": self.model.state_dict(),
            "objective": self.objective.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self._rng_state(),
        }

    def _validate_tensor_state(
        self,
        observed: Mapping[str, torch.Tensor],
        expected: Mapping[str, torch.Tensor],
        *,
        name: str,
    ) -> None:
        if set(observed) != set(expected):
            raise ValueError(f"checkpoint {name} state keys do not match")
        for key, current in expected.items():
            candidate = observed[key]
            if candidate.shape != current.shape or candidate.dtype != current.dtype:
                raise ValueError(f"checkpoint {name} tensor {key!r} is incompatible")

    def _validate_rng_state(self, rng: Mapping[str, Any]) -> None:
        required = {
            "torch_cpu",
            "torch_cuda_available",
            "torch_cuda_device_count",
            "torch_cuda",
            "torch_mps_available",
            "torch_mps",
            "python",
            "numpy",
            "named",
        }
        if set(rng) != required:
            raise ValueError("checkpoint RNG state is missing or has unknown backends")
        cuda_available = torch.cuda.is_available()
        if rng["torch_cuda_available"] is not cuda_available:
            raise ValueError("checkpoint CUDA availability differs from current execution")
        current_device_count = torch.cuda.device_count() if cuda_available else 0
        if rng["torch_cuda_device_count"] != current_device_count:
            raise ValueError("checkpoint CUDA device count differs from current execution")
        cuda_states = tuple(rng["torch_cuda"])
        if len(cuda_states) != current_device_count:
            raise ValueError("checkpoint CUDA RNG state count is incomplete")
        mps_available = torch.backends.mps.is_available()
        if rng["torch_mps_available"] is not mps_available:
            raise ValueError("checkpoint MPS availability differs from current execution")
        mps_state = rng["torch_mps"]
        if mps_available:
            if not isinstance(mps_state, torch.Tensor):
                raise ValueError("checkpoint MPS RNG state is incomplete")
            if mps_state.dtype is not torch.uint8 or mps_state.ndim != 1:
                raise ValueError("checkpoint MPS RNG state is incompatible")
        elif mps_state is not None:
            raise ValueError("checkpoint includes MPS RNG state when MPS is unavailable")
        named = rng["named"]
        if not isinstance(named, Mapping) or set(named) != set(self.named_generators):
            raise ValueError("checkpoint named generator set does not match trainer")
        numpy_state = rng["numpy"]
        if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
            "algorithm",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        }:
            raise ValueError("checkpoint NumPy RNG state is incomplete")

    def resume(self, state: Mapping[str, Any]) -> None:
        required = {
            "schema",
            "resume_contract",
            "run_identity",
            "training_config",
            "execution_config",
            "step",
            "model",
            "objective",
            "optimizer",
            "rng",
        }
        if set(state) != required:
            raise ValueError("training checkpoint is missing or has unknown top-level fields")
        if state["schema"] != self.checkpoint_version:
            raise ValueError("unsupported training checkpoint schema")
        expected_contract = self._resume_contract()
        if dict(state["resume_contract"]) != expected_contract:
            raise ValueError("checkpoint resume contract does not match trainer")
        observed_identity = RunIdentity.model_validate(state["run_identity"])
        assert self.run_identity is not None
        if observed_identity.identity_hash != self.run_identity.identity_hash:
            raise ValueError("checkpoint RunIdentity does not match trainer")
        if canonical_hash(state["training_config"]) != self.run_identity.training_config_hash:
            raise ValueError("checkpoint training config hash does not match RunIdentity")
        if canonical_hash(state["execution_config"]) != self.run_identity.execution_config_hash:
            raise ValueError("checkpoint execution config hash does not match RunIdentity")
        if dict(state["training_config"]) != self.training_config:
            raise ValueError("checkpoint training config does not match trainer")
        if dict(state["execution_config"]) != self.execution_config:
            raise ValueError("checkpoint execution config does not match trainer")
        step = state["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        self._validate_tensor_state(
            state["model"], self.model.state_dict(), name="model"
        )
        self._validate_tensor_state(
            state["objective"], self.objective.state_dict(), name="objective"
        )
        self._validate_rng_state(state["rng"])

        self.model.load_state_dict(state["model"], strict=True)
        self.objective.load_state_dict(state["objective"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        rng = state["rng"]
        torch.random.set_rng_state(torch.as_tensor(rng["torch_cpu"], dtype=torch.uint8).cpu())
        if rng["torch_cuda_available"]:
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(item, dtype=torch.uint8).cpu() for item in rng["torch_cuda"]]
            )
        if rng["torch_mps_available"]:
            torch.mps.set_rng_state(
                torch.as_tensor(rng["torch_mps"], dtype=torch.uint8).cpu()
            )
        random.setstate(_deep_tuple(rng["python"]))
        import numpy as np

        numpy_state = rng["numpy"]
        np.random.set_state(
            (
                str(numpy_state["algorithm"]),
                torch.as_tensor(numpy_state["keys"])
                .cpu()
                .numpy()
                .astype("uint32", copy=False),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        for name, generator_state in rng["named"].items():
            self.named_generators[name].set_state(
                torch.as_tensor(generator_state, dtype=torch.uint8).cpu()
            )
        self.step = step

    def save_checkpoint(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path)
        if destination.suffix != ".safetensors":
            raise ValueError("training checkpoints must use the .safetensors suffix")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        from safetensors.torch import save_file

        state = self.checkpoint_state()
        tensors: dict[str, torch.Tensor] = {}
        for name, value in state["model"].items():
            tensors[f"model.{name}"] = value.detach().cpu().contiguous()
        for name, value in state["objective"].items():
            tensors[f"objective.{name}"] = value.detach().cpu().contiguous()

        rng = state["rng"]
        tensors["rng.torch_cpu"] = rng["torch_cpu"].detach().cpu().contiguous()
        cuda_tensor_names: list[str] = []
        for index, value in enumerate(rng["torch_cuda"]):
            tensor_name = f"rng.torch_cuda.{index}"
            tensors[tensor_name] = value.detach().cpu().contiguous()
            cuda_tensor_names.append(tensor_name)
        mps_tensor_name: str | None = None
        if rng["torch_mps_available"]:
            mps_tensor_name = "rng.torch_mps"
            tensors[mps_tensor_name] = rng["torch_mps"].detach().cpu().contiguous()
        tensors["rng.numpy.keys"] = rng["numpy"]["keys"].detach().cpu().contiguous()
        named_tensor_names: dict[str, str] = {}
        for index, (name, value) in enumerate(sorted(rng["named"].items())):
            tensor_name = f"rng.named.{index}"
            tensors[tensor_name] = value.detach().cpu().contiguous()
            named_tensor_names[name] = tensor_name

        optimizer = state["optimizer"]
        optimizer_scalars: dict[str, dict[str, Any]] = {}
        optimizer_tensors: list[dict[str, str | int]] = []
        for parameter_id, parameter_state in optimizer["state"].items():
            scalars: dict[str, Any] = {}
            for field_name, value in parameter_state.items():
                if isinstance(value, torch.Tensor):
                    tensor_name = f"optimizer.{len(optimizer_tensors)}"
                    tensors[tensor_name] = value.detach().cpu().contiguous()
                    optimizer_tensors.append(
                        {
                            "tensor": tensor_name,
                            "parameter_id": int(parameter_id),
                            "field": str(field_name),
                        }
                    )
                else:
                    scalars[str(field_name)] = _json_safe(value)
            optimizer_scalars[str(parameter_id)] = scalars
        header = {
            "schema": state["schema"],
            "resume_contract": state["resume_contract"],
            "run_identity": state["run_identity"],
            "training_config": state["training_config"],
            "execution_config": state["execution_config"],
            "step": state["step"],
            "optimizer_param_groups": _json_safe(optimizer["param_groups"]),
            "optimizer_state_scalars": optimizer_scalars,
            "optimizer_tensor_fields": optimizer_tensors,
            "rng": {
                "torch_cuda_available": rng["torch_cuda_available"],
                "torch_cuda_device_count": rng["torch_cuda_device_count"],
                "torch_cuda_tensor_names": cuda_tensor_names,
                "torch_mps_available": rng["torch_mps_available"],
                "torch_mps_tensor_name": mps_tensor_name,
                "python": _json_safe(rng["python"]),
                "numpy": {
                    "algorithm": rng["numpy"]["algorithm"],
                    "position": rng["numpy"]["position"],
                    "has_gauss": rng["numpy"]["has_gauss"],
                    "cached_gaussian": rng["numpy"]["cached_gaussian"],
                },
                "named_tensor_names": named_tensor_names,
            },
        }
        save_file(
            tensors,
            str(temporary),
            metadata={"tabu_training_state": json.dumps(header, sort_keys=True)},
        )
        temporary.replace(destination)
        return destination

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        source = Path(path)
        if source.suffix != ".safetensors":
            raise ValueError("training checkpoints must use the .safetensors suffix")
        from safetensors import safe_open

        with safe_open(str(source), framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            encoded_header = metadata.get("tabu_training_state")
            if encoded_header is None:
                raise ValueError("checkpoint is missing TabU training metadata")
            header = json.loads(encoded_header)
            tensors = {
                name: checkpoint.get_tensor(name)
                for name in checkpoint.keys()  # noqa: SIM118 - safe_open is not iterable
            }
        required_header = {
            "schema",
            "resume_contract",
            "run_identity",
            "training_config",
            "execution_config",
            "step",
            "optimizer_param_groups",
            "optimizer_state_scalars",
            "optimizer_tensor_fields",
            "rng",
        }
        if set(header) != required_header:
            raise ValueError("checkpoint header is missing or has unknown fields")
        optimizer_state: dict[int, dict[str, Any]] = {
            int(parameter_id): dict(fields)
            for parameter_id, fields in header["optimizer_state_scalars"].items()
        }
        for descriptor in header["optimizer_tensor_fields"]:
            tensor_name = descriptor["tensor"]
            if tensor_name not in tensors:
                raise ValueError("checkpoint optimizer tensor descriptor is incomplete")
            parameter_id = int(descriptor["parameter_id"])
            optimizer_state.setdefault(parameter_id, {})[descriptor["field"]] = tensors[
                tensor_name
            ]
        rng_header = header["rng"]
        required_rng_header = {
            "torch_cuda_available",
            "torch_cuda_device_count",
            "torch_cuda_tensor_names",
            "torch_mps_available",
            "torch_mps_tensor_name",
            "python",
            "numpy",
            "named_tensor_names",
        }
        if set(rng_header) != required_rng_header:
            raise ValueError("checkpoint RNG header is incomplete")
        required_rng_tensors = {
            "rng.torch_cpu",
            "rng.numpy.keys",
            *rng_header["torch_cuda_tensor_names"],
            *rng_header["named_tensor_names"].values(),
        }
        mps_tensor_name = rng_header["torch_mps_tensor_name"]
        if rng_header["torch_mps_available"]:
            if mps_tensor_name != "rng.torch_mps":
                raise ValueError("checkpoint MPS RNG tensor descriptor is incomplete")
            required_rng_tensors.add(mps_tensor_name)
        elif mps_tensor_name is not None:
            raise ValueError("checkpoint MPS RNG tensor descriptor is invalid")
        if not required_rng_tensors.issubset(tensors):
            raise ValueError("checkpoint RNG tensors are incomplete")
        expected_tensor_names = {
            *(f"model.{name}" for name in self.model.state_dict()),
            *(f"objective.{name}" for name in self.objective.state_dict()),
            *(descriptor["tensor"] for descriptor in header["optimizer_tensor_fields"]),
            *required_rng_tensors,
        }
        if set(tensors) != expected_tensor_names:
            raise ValueError("checkpoint tensor set is missing or contains unknown state")
        state = {
            "schema": header["schema"],
            "resume_contract": header["resume_contract"],
            "run_identity": header["run_identity"],
            "training_config": header["training_config"],
            "execution_config": header["execution_config"],
            "step": header["step"],
            "model": {
                name.removeprefix("model."): value
                for name, value in tensors.items()
                if name.startswith("model.")
            },
            "objective": {
                name.removeprefix("objective."): value
                for name, value in tensors.items()
                if name.startswith("objective.")
            },
            "optimizer": {
                "state": optimizer_state,
                "param_groups": header["optimizer_param_groups"],
            },
            "rng": {
                "torch_cpu": tensors["rng.torch_cpu"],
                "torch_cuda_available": rng_header["torch_cuda_available"],
                "torch_cuda_device_count": rng_header["torch_cuda_device_count"],
                "torch_cuda": tuple(
                    tensors[name] for name in rng_header["torch_cuda_tensor_names"]
                ),
                "torch_mps_available": rng_header["torch_mps_available"],
                "torch_mps": (
                    tensors[mps_tensor_name] if mps_tensor_name is not None else None
                ),
                "python": rng_header["python"],
                "numpy": {
                    **rng_header["numpy"],
                    "keys": tensors["rng.numpy.keys"],
                },
                "named": {
                    name: tensors[tensor_name]
                    for name, tensor_name in rng_header["named_tensor_names"].items()
                },
            },
        }
        self.resume(state)


def train_model(
    model: nn.Module,
    batches: Iterable[tuple[Any, TruthSidecar]],
    *,
    epochs: int = 1,
    learning_rate: float = 1.0e-3,
    objective: nn.Module | None = None,
) -> tuple[Trainer, tuple[TrainStep, ...]]:
    trainer = Trainer(
        model,
        objective=objective,
        learning_rate=learning_rate,
    )
    return trainer, trainer.fit(batches, epochs=epochs)


train = train_model


__all__ = ["TrainStep", "Trainer", "train", "train_model"]
