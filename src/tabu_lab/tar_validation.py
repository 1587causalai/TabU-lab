"""Operational validation defaults, separate from immutable model implementation identity."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from tabu_lab.tar_sizes import DEFAULT_VALIDATION_SIZE, config_for_size


def verify_size(size=DEFAULT_VALIDATION_SIZE):
    import torch

    from tabu_lab.models.tar import TabUTARModel, TARTrainer, TARTrainingConfig
    from tabu_lab.models.tar.checkpoint import load_checkpoint, save_checkpoint, source_digest
    from tabu_lab.models.tar.verification import mixed_fixture, verify_model

    cfg = config_for_size(size)
    if size == "standard":
        result = verify_model(full=True)
    else:
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = TabUTARModel(cfg)
            count = sum(p.numel() for p in model.parameters())
            if count != cfg.parameter_count:
                raise AssertionError("parameter count mismatch")
            episode, truth = mixed_fixture()
            trainer = TARTrainer(
                model,
                TARTrainingConfig(effective_episode_batch=1, warmup_steps=0, optimizer_steps=2),
            )
            before = model.response_cell.detach().clone()
            update = trainer.train_step([(episode, truth)])
            if torch.equal(before, model.response_cell):
                raise AssertionError("optimizer failed to update")
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise AssertionError("nonfinite gradient")
            trainer.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                out = model(episode)
                if any(p.status != "ok" for p in out.predictions):
                    raise AssertionError("invalid prediction status")
                if bool(out.carriers[4, 2].any()):
                    raise AssertionError("missing-cell Null drift")
                if bool(out.carriers[5:, 3:].any()):
                    raise AssertionError("bank-corner Null drift")
            with tempfile.TemporaryDirectory(prefix="tar-size-verification-") as directory:
                checkpoint = Path(directory) / "checkpoint"
                save_checkpoint(model, checkpoint)
                restored = load_checkpoint(checkpoint, expected_config=cfg)
                with torch.no_grad():
                    torch.testing.assert_close(
                        restored(episode).responses, out.responses, rtol=0, atol=0
                    )
            result = dict(
                model_id=model.model_id,
                status="local_unissued",
                full_default=False,
                parameter_count=count,
                initial_loss=update["loss"],
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
        finally:
            torch.set_num_threads(threads)
    result.update(
        model_size=size,
        config=cfg.as_dict(),
        config_sha256=hashlib.sha256(
            json.dumps(cfg.as_dict(), sort_keys=True).encode()
        ).hexdigest(),
    )
    return result
