from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tabu_lab.evolution import (
    EvolutionRepository,
    SamplingPolicyEngine,
    SamplingPolicyNode,
    WorldMixtureNode,
)

ROOT = Path(__file__).resolve().parents[2]


def _policy_fixture(policy_ref: str) -> tuple[SamplingPolicyNode, WorldMixtureNode]:
    repository = EvolutionRepository.load(ROOT)
    policy = repository.node(policy_ref)
    mixture = repository.node("tabu.mixture.supervised-v2-v3@1.0.0")
    assert isinstance(policy, SamplingPolicyNode)
    assert isinstance(mixture, WorldMixtureNode)
    return policy, mixture


def test_fixed_policy_state_and_rng_replay_are_exact() -> None:
    policy, mixture = _policy_fixture("tabu.policy.fixed@1.0.0")
    generator = torch.Generator().manual_seed(20260831)
    engine = SamplingPolicyEngine(policy, mixture)
    for _ in range(7):
        engine.choose(generator)
    saved_policy_state = engine.state
    saved_rng_state = generator.get_state().clone()
    expected = tuple(engine.choose(generator).ref for _ in range(20))

    restored_generator = torch.Generator()
    restored_generator.set_state(saved_rng_state)
    restored = SamplingPolicyEngine(policy, mixture, state=saved_policy_state)
    observed = tuple(restored.choose(restored_generator).ref for _ in range(20))

    assert observed == expected
    assert restored.state == engine.state


def test_adaptive_policy_decision_state_is_json_roundtrippable() -> None:
    policy, mixture = _policy_fixture("tabu.policy.adaptive@1.0.0")
    engine = SamplingPolicyEngine(policy, mixture)
    generator = torch.Generator().manual_seed(9)

    engine.choose(generator)
    engine.observe(1.25)
    payload = engine.state.model_dump(mode="json")
    restored = SamplingPolicyEngine(policy, mixture)
    restored.restore(payload)

    assert restored.state == engine.state
    assert restored.state.state_hash == engine.state.state_hash


def test_policy_state_cannot_cross_policy_identity() -> None:
    fixed, mixture = _policy_fixture("tabu.policy.fixed@1.0.0")
    piecewise, _ = _policy_fixture("tabu.policy.piecewise@1.0.0")
    state = SamplingPolicyEngine(fixed, mixture).state

    with pytest.raises(ValueError, match="policy_ref"):
        SamplingPolicyEngine(piecewise, mixture, state=state)
