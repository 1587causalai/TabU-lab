"""Deterministic, serializable generator-mixture sampling policies."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from pydantic import Field, field_validator, model_validator

from tabu_lab.contracts import canonical_hash

from .models import (
    NodeRef,
    SamplingPolicyKind,
    SamplingPolicyNode,
    StrictModel,
    WorldMixtureNode,
)


class SamplingPolicyState(StrictModel):
    schema_version: Literal["tabu.sampling-policy-state.v1"] = (
        "tabu.sampling-policy-state.v1"
    )
    policy_ref: str
    policy_hash: str
    mixture_ref: str
    mixture_hash: str
    step: int = Field(ge=0)
    counts: dict[str, int]
    ema_losses: dict[str, float]
    last_generator: str | None = None

    @field_validator("counts")
    @classmethod
    def _valid_counts(cls, values: dict[str, int]) -> dict[str, int]:
        if any(isinstance(value, bool) or value < 0 for value in values.values()):
            raise ValueError("sampling counts must be non-negative integers")
        return dict(sorted(values.items()))

    @field_validator("ema_losses")
    @classmethod
    def _valid_losses(cls, values: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("adaptive policy losses must be finite")
        return dict(sorted(values.items()))

    @model_validator(mode="after")
    def _keys_match(self) -> SamplingPolicyState:
        if set(self.counts) != set(self.ema_losses):
            raise ValueError("policy count and loss keys must match")
        if self.last_generator is not None and self.last_generator not in self.counts:
            raise ValueError("last_generator is absent from policy state")
        return self

    @property
    def state_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="python"))


class SamplingPolicyEngine:
    """Runtime for fixed, piecewise, and bounded adaptive policies."""

    def __init__(
        self,
        policy: SamplingPolicyNode,
        mixture: WorldMixtureNode,
        *,
        state: SamplingPolicyState | None = None,
    ) -> None:
        if not policy.deterministic or not policy.serializable_state:
            raise ValueError("program runtime requires deterministic serializable policies")
        self.policy = policy
        self.mixture = mixture
        self._refs = tuple(entry.generator.ref for entry in mixture.entries)
        self._base_weights = {
            entry.generator.ref: float(entry.weight) for entry in mixture.entries
        }
        self._validate_segments()
        if state is None:
            self._state = SamplingPolicyState(
                policy_ref=policy.ref,
                policy_hash=policy.node_hash,
                mixture_ref=mixture.ref,
                mixture_hash=mixture.node_hash,
                step=0,
                counts={ref: 0 for ref in self._refs},
                ema_losses={ref: 0.0 for ref in self._refs},
            )
        else:
            self._validate_state_identity(state)
            self._state = state

    def _validate_segments(self) -> None:
        for segment in self.policy.segments:
            if set(segment.weights) != set(self._refs):
                raise ValueError("piecewise policy segment keys must equal mixture generator refs")
            total = sum(segment.weights.values())
            if not math.isclose(total, 1.0, abs_tol=1.0e-9):
                raise ValueError("piecewise policy segment weights must sum to one")

    def _validate_state_identity(self, state: SamplingPolicyState) -> None:
        expected = {
            "policy_ref": self.policy.ref,
            "policy_hash": self.policy.node_hash,
            "mixture_ref": self.mixture.ref,
            "mixture_hash": self.mixture.node_hash,
        }
        for name, value in expected.items():
            if getattr(state, name) != value:
                raise ValueError(f"sampling policy state identity mismatch at {name}")
        if set(state.counts) != set(self._refs):
            raise ValueError("sampling policy state generator set differs from mixture")

    @property
    def state(self) -> SamplingPolicyState:
        return self._state

    def _weights(self) -> dict[str, float]:
        if self.policy.policy_kind is SamplingPolicyKind.FIXED:
            return dict(self._base_weights)
        if self.policy.policy_kind is SamplingPolicyKind.PIECEWISE:
            active = self.policy.segments[0]
            for segment in self.policy.segments:
                if segment.start_step > self._state.step:
                    break
                active = segment
            return dict(active.weights)
        logits = []
        for ref in self._refs:
            base = max(self._base_weights[ref], 1.0e-12)
            score = self._state.ema_losses[ref]
            logits.append(math.log(base) + self.policy.adaptive_temperature * score)
        offset = max(logits)
        unnormalized = [math.exp(value - offset) for value in logits]
        total = sum(unnormalized)
        return {
            ref: value / total for ref, value in zip(self._refs, unnormalized, strict=True)
        }

    def choose(self, generator: torch.Generator) -> NodeRef:
        if not isinstance(generator, torch.Generator):
            raise TypeError("sampling policy requires a torch.Generator")
        weights = self._weights()
        vector = torch.tensor([weights[ref] for ref in self._refs], dtype=torch.float64)
        index = int(torch.multinomial(vector, 1, generator=generator).item())
        selected = self._refs[index]
        counts = dict(self._state.counts)
        counts[selected] += 1
        self._state = self._state.model_copy(
            update={
                "step": self._state.step + 1,
                "counts": counts,
                "last_generator": selected,
            }
        )
        return NodeRef.parse(selected)

    def observe(self, loss: float) -> None:
        if not math.isfinite(loss):
            raise ValueError("sampling policy observations must be finite")
        selected = self._state.last_generator
        if selected is None:
            raise ValueError("sampling policy cannot observe before choose")
        if self.policy.policy_kind is not SamplingPolicyKind.ADAPTIVE:
            return
        losses = dict(self._state.ema_losses)
        previous = losses[selected]
        if self._state.counts[selected] == 1:
            losses[selected] = float(loss)
        else:
            decay = self.policy.adaptive_ema
            losses[selected] = decay * previous + (1.0 - decay) * float(loss)
        self._state = self._state.model_copy(update={"ema_losses": losses})

    def restore(self, payload: SamplingPolicyState | dict[str, Any]) -> None:
        state = (
            payload
            if isinstance(payload, SamplingPolicyState)
            else SamplingPolicyState.model_validate(payload)
        )
        self._validate_state_identity(state)
        self._state = state


__all__ = ["SamplingPolicyEngine", "SamplingPolicyState"]
