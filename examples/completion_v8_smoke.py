"""One FP32 V8 completion update and weights-only replay on a synthetic table."""

import argparse
import json
import tempfile
from pathlib import Path

import torch

from tabu_lab.models.completion_v8 import (
    V8CompletionModel,
    V8Config,
    completion_loss,
    load_weights,
    prepare_completion,
    save_weights,
)
from tabu_lab.models.restoration.backbone import BackboneConfig
from tabu_lab.models.restoration.contracts import ColumnSchema, make_episode
from tabu_lab.models.restoration_v7 import V7Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--codebook-mode", choices=("inherited", "learned_pool_v1"), default="inherited"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(8)
    spec = V7Config.v73(
        backbone=BackboneConfig(width=128, layers=1, heads=4, ff_width=64, slots=3),
        unit_layers=1,
        coupling_blocks=1,
        coupling_hidden=(16,),
        center_chunk_size=4,
    )
    model = V8CompletionModel(
        V8Config(spec, codebook_mode=args.codebook_mode)
    ).to(device=args.device, dtype=torch.float32)
    schema = (
        ColumnSchema("measurement", "numeric"),
        ColumnSchema("category", "nominal", 3),
        ColumnSchema("rank", "ordinal", 3, order=(2, 0, 1)),
        ColumnSchema("target", "numeric"),
    )
    x = torch.linspace(-1, 1, 12, device=args.device)
    category = torch.arange(12, device=args.device) % 3
    values = (x, category, category, x.square() + 0.1 * x)
    observed = torch.ones(12, 4, dtype=torch.bool, device=args.device)
    query = torch.zeros_like(observed)
    query[-3:, -1] = True
    query[-1, 1:3] = True
    inputs, _, truth = make_episode(schema, values, observed, query, code_seed=17)
    episode = prepare_completion(inputs, model.config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    before = completion_loss(model(episode, decode=False), episode, truth)
    before.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("missing or nonfinite completion gradients")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(gradient_norm)) or not bool(gradient_norm > 0):
        raise FloatingPointError("completion gradient norm must be finite and positive")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        after = model(episode)
        loss_after = completion_loss(after, episode, truth)
        with tempfile.TemporaryDirectory(prefix="tabu-v8-smoke-") as folder:
            path = Path(folder) / "weights.pt"
            save_weights(model, path)
            restored = load_weights(path, device=args.device).eval()
            replay = restored(prepare_completion(inputs, restored.config))
            torch.testing.assert_close(
                replay.completed_codes, after.completed_codes, rtol=0, atol=0
            )
            for column, prediction in after.decoded.items():
                torch.testing.assert_close(replay.decoded[column], prediction, rtol=0, atol=0)
    print(json.dumps({
        "model": "completion_v8",
        "model_version": model.config.model_version,
        "evidence_level": "local_unissued",
        "claim_boundary": "synthetic integration smoke; not predictive quality or exact resume",
        "device": args.device,
        "dtype": "float32",
        "torch_version": str(torch.__version__),
        "config": model.config.as_dict(),
        "execution_precision": model.execution_precision,
        "query_cells": int(query.sum()),
        "optimizer_steps": 1,
        "loss_before": float(before.detach()),
        "loss_after": float(loss_after),
        "gradient_norm": float(gradient_norm),
        "weights_replay_exact": True,
    }, indent=2))


if __name__ == "__main__":
    main()
