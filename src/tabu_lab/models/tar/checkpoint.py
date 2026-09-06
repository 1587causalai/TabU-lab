"""Local TAR checkpoints; safe tensor data and strict config/code identity binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tabu_lab.registry import get_model_spec

from .config import TARConfig
from .model import TabUTARModel


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_digest():
    root = Path(__file__).parent
    entries = {f.name: digest(f.read_bytes()) for f in sorted(root.glob("*.py"))}
    entries["model_spec"] = get_model_spec("tabu.tar").model_dump(mode="json")
    return digest(json.dumps(entries, sort_keys=True).encode())


def save_checkpoint(model, path, *, trainer=None):
    path = Path(path)
    if trainer is not None and trainer.model is not model:
        raise ValueError("trainer belongs to a different model")
    path.mkdir(parents=True, exist_ok=False)
    config = model.config.as_dict()
    weights = path / "weights.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(weights)
    )
    manifest = dict(
        format="tabu.tar.local-checkpoint.1",
        model_id=model.model_id,
        status="local_unissued",
        config=config,
        design_sha256=get_model_spec("tabu.tar").upstream.sha256,
        config_sha256=digest(json.dumps(config, sort_keys=True).encode()),
        source_sha256=source_digest(),
        weights_sha256=digest(weights.read_bytes()),
    )
    if trainer is not None:
        state_path = path / "optimizer.pt"
        torch.save(
            dict(
                optimizer=trainer.optimizer.state_dict(),
                step=trainer.step,
                training_config=asdict(trainer.config),
                torch_rng=torch.get_rng_state(),
            ),
            state_path,
        )
        manifest["optimizer_sha256"] = digest(state_path.read_bytes())
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_checkpoint(path, *, device="cpu", expected_config=None, restore_trainer=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest.get("model_id") != "tabu.tar"
        or manifest.get("format") != "tabu.tar.local-checkpoint.1"
    ):
        raise ValueError("checkpoint is not the TAR model identity")
    config = manifest["config"]
    if digest(json.dumps(config, sort_keys=True).encode()) != manifest["config_sha256"]:
        raise ValueError("checkpoint config digest mismatch")
    cfg = TARConfig(**config)
    if expected_config is not None and cfg != expected_config:
        raise ValueError("checkpoint config does not match requested model")
    if source_digest() != manifest["source_sha256"]:
        raise ValueError("checkpoint source identity mismatch; explicit migration is required")
    f = path / "weights.safetensors"
    if digest(f.read_bytes()) != manifest["weights_sha256"]:
        raise ValueError("checkpoint weight digest mismatch")
    state = load_file(str(f), device="cpu")
    dtype = state["continuous"].dtype
    model = TabUTARModel(cfg, device=device, dtype=dtype)
    model.load_state_dict(state, strict=True)
    if not restore_trainer:
        return model
    from .training import TARTrainer, TARTrainingConfig

    opt = path / "optimizer.pt"
    if not opt.is_file() or digest(opt.read_bytes()) != manifest.get("optimizer_sha256"):
        raise ValueError("optimizer missing or digest mismatch")
    saved = torch.load(opt, map_location=device, weights_only=True)
    trainer = TARTrainer(model, TARTrainingConfig(**saved["training_config"]))
    trainer.optimizer.load_state_dict(saved["optimizer"])
    trainer.step = saved["step"]
    torch.set_rng_state(saved["torch_rng"].cpu())
    return model, trainer
