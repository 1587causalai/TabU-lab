"""One deterministic local CPU optimizer step; no dataset, service or telemetry."""

import json

import torch

from tabu_lab.models.restoration_v53 import (
    ColumnSchema,
    V53Model,
    make_episode,
    prepare_episode,
    score_prepared_episode,
)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(53)
    schema = (
        ColumnSchema("measurement", "numeric"),
        ColumnSchema("category", "nominal", 3),
        ColumnSchema("rank", "ordinal", 3, order=(2, 0, 1)),
    )
    values = (
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64),
        torch.tensor([0, 1, 2, 0, 1, 2]),
        torch.tensor([0, 1, 2, 0, 1, 2]),
    )
    observed = torch.ones(6, 3, dtype=torch.bool)
    query = torch.zeros_like(observed)
    query[-1] = True
    episode = make_episode(schema, values, observed, query, code_seed=17)
    model = V53Model().double()
    prepared = prepare_episode(model, *episode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = score_prepared_episode(model, prepared)
    before.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("nonfinite or missing smoke gradients")
    gradient_norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
    if not bool(torch.isfinite(gradient_norm)) or not bool(gradient_norm > 0):
        raise FloatingPointError("smoke gradient norm must be finite and nonzero")
    optimizer.step()
    with torch.no_grad():
        after = score_prepared_episode(model, prepared, decode=True)
    print(json.dumps({
        "model": "restoration-v53-reference",
        "evidence_level": "local_unissued",
        "device": "cpu",
        "dtype": "float64",
        "torch_version": torch.__version__,
        "config": model.config.as_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "targets": len(episode[1].targets),
        "optimizer_steps": 1,
        "loss_before": float(before.loss.detach()),
        "loss_after": float(after.loss),
        "gradient_norm": float(gradient_norm),
        "numeric_query_prediction": float(after.output.columns[0].decoded[-1]),
    }, indent=2))


if __name__ == "__main__":
    main()
