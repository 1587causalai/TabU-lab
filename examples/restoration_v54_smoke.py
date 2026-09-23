"""One local CPU optimizer step for a named V5.4 size; no capability claim."""

import argparse
import json

import torch

from tabu_lab.models.restoration_v54 import (
    ColumnSchema,
    V54Config,
    V54Model,
    make_episode,
    prepare_episode,
    score_prepared_episode,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("nano", "small"), default="small")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(54)
    schema = (
        ColumnSchema("measurement", "numeric"),
        ColumnSchema("category", "nominal", 3),
        ColumnSchema("rank", "ordinal", 3, order=(2, 0, 1)),
        ColumnSchema("target", "numeric"),
    )
    x = torch.arange(8, dtype=torch.float64)
    values = (x, x.long() % 3, x.long() % 3, 0.7 * x + 0.1 * x.sin())
    observed = torch.ones(8, 4, dtype=torch.bool)
    query = torch.zeros_like(observed)
    query[-2:, -1] = True  # Supervised-row: features remain visible.
    episode = make_episode(schema, values, observed, query, code_seed=17)
    model = V54Model(V54Config.from_size(args.size)).double()
    prepared = prepare_episode(model, *episode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = score_prepared_episode(model, prepared)
    before.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("nonfinite or missing smoke gradients")
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]))
    if not bool(torch.isfinite(norm)) or not bool(norm > 0):
        raise FloatingPointError("smoke gradient norm must be finite and nonzero")
    optimizer.step()
    with torch.no_grad():
        after = score_prepared_episode(model, prepared, decode=True)
    print(json.dumps({
        "model": "restoration-v54-reference",
        "evidence_level": "local_unissued",
        "claim_boundary": "one finite local optimizer step; not a fitting result",
        "device": "cpu", "dtype": "float64", "torch_version": str(torch.__version__),
        "config": model.config.as_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "episode_kind": "supervised_row", "query_cells": int(query.sum()),
        "optimizer_steps": 1,
        "loss_before": float(before.loss.detach()), "loss_after": float(after.loss),
        "gradient_norm": float(norm),
    }, indent=2))


if __name__ == "__main__":
    main()
