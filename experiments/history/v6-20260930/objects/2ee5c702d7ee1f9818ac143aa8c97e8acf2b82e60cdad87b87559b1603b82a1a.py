"""Established V6 checkpoint/RNG/episode helpers for OpenML12 adaptation."""
import hashlib
import os
from pathlib import Path
import torch
from tabu_lab.curriculum_v53.artifacts import (
    finite_state, restore_rng as base_restore_rng, rng_state as base_rng_state, sha256,
)

def rng_state():
    result = base_rng_state()
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state):
    base_restore_rng(state)
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])



def checkpoint(root, model, optimizer, state, identity):
    if not finite_state(model.state_dict()) or not finite_state(optimizer.state_dict()):
        raise FloatingPointError("nonfinite state blocks V6 checkpoint")
    payload = dict(schema="tabu.v6.weights-only-checkpoint.v1", purpose="training",
                   identity=identity, model_config=model.config.as_dict(),
                   model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                   optimizer=optimizer.state_dict(), state=state, rng=rng_state())
    folder = root / "checkpoints"
    folder.mkdir(exist_ok=True)
    temporary = folder / ".pending.pt"
    torch.save(payload, temporary)
    digest = sha256(temporary)
    stable = folder / (digest + ".pt")
    os.replace(temporary, stable)
    alias = root / "checkpoint-progress.pt"
    link = root / ".checkpoint-progress.pt.tmp"
    link.symlink_to(Path("checkpoints") / stable.name)
    os.replace(link, alias)
    return stable, digest



def episode_seeds(plan):
    seeds = dict(plan.spec["seeds"])
    # Preserve the previous V6 episode stream for a matched objective comparison.
    namespace = "v6-target-broadcast-dgx2-20260927"
    for stream in ("masks", "codes", "windows"):
        seeds[stream] = int.from_bytes(
            hashlib.sha256(f"{seeds[stream]}/{namespace}".encode()).digest()[:8], "little"
        )
    return seeds


