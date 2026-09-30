"""Composable task adapters for Axis-C query-family models.

The query-family core restores typed cells under an arbitrary target mask.  A
task adapter may add bounded, identity-bearing conditioning without redefining
the tokenizer, dynamics, geometry, terminal, or truth boundary.  Adapters never
receive a ``TruthSidecar``.

This module deliberately defines the seam before wiring it into a ModelSpec.
That keeps existing ``tabu.query.row@0.2.0`` variants and checkpoints unchanged
until a separate contract migration explicitly selects an adapter composition.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor, nn

from tabu_lab.contracts import canonical_hash

from .table_cell import LabelColumnBroadcast


class QueryTaskAdapterInsertion(StrEnum):
    """Typed insertion points available to a query-family task adapter."""

    POST_TOKENIZER_PRE_DYNAMICS = "post_tokenizer_pre_dynamics"


@dataclass(frozen=True, slots=True)
class QueryTaskAdapterSpec:
    """Immutable semantic identity for one protected task-adapter type."""

    adapter_id: str
    adapter_version: str
    insertion: QueryTaskAdapterInsertion
    target_origin: str
    compatible_models: tuple[str, ...]
    residual: bool = True
    schema_version: str = "tabu.query-task-adapter-spec.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "tabu.query-task-adapter-spec.v1":
            raise ValueError("unsupported query task adapter spec schema")
        if not self.adapter_id.strip() or not self.adapter_version.strip():
            raise ValueError("query task adapter identity cannot be blank")
        if not self.target_origin.strip():
            raise ValueError("query task adapter target_origin cannot be blank")
        if not self.compatible_models or any(
            not model_ref.strip() for model_ref in self.compatible_models
        ):
            raise ValueError("query task adapter compatible_models cannot be empty")

    @property
    def spec_ref(self) -> str:
        return f"{self.adapter_id}@{self.adapter_version}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "insertion": self.insertion.value,
            "target_origin": self.target_origin,
            "compatible_models": list(self.compatible_models),
            "residual": self.residual,
        }

    @property
    def spec_hash(self) -> str:
        return canonical_hash(self.as_dict())


class QueryTaskAdapterBase(nn.Module, ABC):
    """Shape-preserving, truth-free task specialization for query models."""

    spec: QueryTaskAdapterSpec

    @abstractmethod
    def identity_payload(self) -> dict[str, Any]:
        """Return semantic configuration for variant/checkpoint identity."""


class SupervisedResponseAdapter(QueryTaskAdapterBase):
    """Residual label conditioning for supervised response prediction.

    ``LabelColumnBroadcast`` already defines the truth-free proposal.  This
    wrapper makes its contribution an explicit residual:

    ``cells + rho * (broadcast(cells, evidence) - cells)``.

    Setting ``rho`` to zero is therefore an exact algebraic degeneration to
    the cell-restoration core.  The adapter sees model-visible evidence only;
    query truth remains outside the public forward boundary.
    """

    spec = QueryTaskAdapterSpec(
        adapter_id="tabu.query.adapter.supervised_response",
        adapter_version="0.1.0",
        insertion=QueryTaskAdapterInsertion.POST_TOKENIZER_PRE_DYNAMICS,
        target_origin="query",
        compatible_models=("tabu.query.base@0.1.0", "tabu.query.row@0.2.0"),
    )

    def __init__(
        self,
        *,
        tau: float = 1.0e-6,
        residual_gate_initial: float = 1.0e-2,
        trainable_gate: bool = True,
    ) -> None:
        super().__init__()
        gate_initial = float(residual_gate_initial)
        if not math.isfinite(gate_initial):
            raise ValueError("residual_gate_initial must be finite")
        if gate_initial < 0.0:
            raise ValueError("residual_gate_initial must be non-negative")
        self.tau = float(tau)
        self.residual_gate_initial = gate_initial
        self.trainable_gate = bool(trainable_gate)
        self.broadcast = LabelColumnBroadcast(tau=self.tau)
        gate = torch.tensor(gate_initial, dtype=torch.float32)
        if self.trainable_gate:
            self.residual_gate = nn.Parameter(gate)
        else:
            self.register_buffer("residual_gate", gate, persistent=True)

    def forward(self, cells: Tensor, evidence: Any) -> Tensor:
        proposal = self.broadcast(cells, evidence)
        residual = proposal - cells
        gate = self.residual_gate.to(device=cells.device, dtype=cells.dtype)
        return cells + gate * residual

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": "tabu.query-task-adapter-composition.v1",
            "spec_ref": self.spec.spec_ref,
            "spec_hash": self.spec.spec_hash,
            "insertion": self.spec.insertion.value,
            "target_origin": self.spec.target_origin,
            "tau": self.tau,
            "residual_gate_initial": self.residual_gate_initial,
            "trainable_gate": self.trainable_gate,
            "truth_visible": False,
        }
        return {**payload, "composition_hash": canonical_hash(payload)}

    def trace_metadata(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "residual_gate": float(self.residual_gate.detach().cpu()),
        }


__all__ = [
    "QueryTaskAdapterBase",
    "QueryTaskAdapterInsertion",
    "QueryTaskAdapterSpec",
    "SupervisedResponseAdapter",
]
