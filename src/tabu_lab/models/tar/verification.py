"""Bounded local implementation probes; these do not issue experiment receipts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, save_checkpoint, source_digest
from .config import TARConfig
from .model import TabUTARModel
from .training import TARTrainer, TARTrainingConfig, score
from .types import TAREpisode, TARFeature


def mixed_fixture():
    values = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 1.0, 1.0], [4.0, 0.0, 2.0], [3.0, 1.0, 1.0], [5.0, 2.0, 0.0]]
    )
    queries = torch.zeros_like(values, dtype=torch.bool)
    queries[3, 0] = queries[4, 1] = queries[3, 2] = True
    visible = ~queries
    visible[4, 2] = False  # A natural missing cell, not a scored target.
    features = (
        TARFeature(column_id=0),
        TARFeature("nominal", ("a", "b", "c"), 1),
        TARFeature("ordinal", ("low", "mid", "high"), 2),
    )
    truth = {tuple(a): float(values[tuple(a)]) for a in queries.nonzero().tolist()}
    return TAREpisode.from_table(values, visible, queries, features, codebook_seed=7), truth


def inspect_model():
    cfg = TARConfig()
    model = TabUTARModel(cfg, device="meta")
    count = sum(p.numel() for p in model.parameters())
    if count != cfg.parameter_count:
        raise AssertionError("parameter count differs from design formula")
    return dict(
        model_id=model.model_id,
        status="local_unissued",
        parameter_count=count,
        parameter_tensors=len(list(model.parameters())),
        config=cfg.as_dict(),
        source_sha256=source_digest(),
    )


def verify_model(*, full=False):
    """Actual weights, backward, one AdamW update and strict checkpoint roundtrip."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = (
            TARConfig()
            if full
            else TARConfig(
                width=16, heads=4, ff_width=32, semantic_slots=3, blocks=2, inducing_slots=8
            )
        )
        model = TabUTARModel(cfg)
        episode, truth = mixed_fixture()
        output = model(episode)
        loss = score(output, truth)
        loss.backward()
        count = sum(p.numel() for p in model.parameters())
        if count != cfg.parameter_count:
            raise AssertionError("parameter count mismatch")
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        ):
            raise AssertionError("nonfinite gradient")
        if bool(output.carriers[4, 2].any()) or bool(output.carriers[5:, 3:].any()):
            raise AssertionError("Null drift")
        loss_value = float(loss.detach())
        del output, loss
        trainer = TARTrainer(
            model, TARTrainingConfig(effective_episode_batch=1, warmup_steps=0, optimizer_steps=2)
        )
        before = model.response_cell.detach().clone()
        update = trainer.train_step([(episode, truth)])
        if torch.equal(before, model.response_cell):
            raise AssertionError("optimizer failed to update response weights")
        # Release gradients before allocating a second full model for roundtrip.
        trainer.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            prediction = model(episode).responses
        with tempfile.TemporaryDirectory(prefix="tabu-tar-verify-") as directory:
            save_checkpoint(model, Path(directory) / "checkpoint")
            restored = load_checkpoint(Path(directory) / "checkpoint", expected_config=cfg)
            with torch.no_grad():
                torch.testing.assert_close(restored(episode).responses, prediction, rtol=0, atol=0)
        result = dict(
            model_id=model.model_id,
            status="local_unissued",
            full_default=full,
            parameter_count=count,
            initial_loss=loss_value,
            optimizer_update=update,
            checks=[
                "shape_and_count",
                "mixed_numeric_nominal_ordinal",
                "finite_backward",
                "exact_null",
                "adamw_update",
                "strict_checkpoint_roundtrip",
            ],
            source_sha256=source_digest(),
        )
        return result
    finally:
        torch.set_num_threads(previous_threads)
