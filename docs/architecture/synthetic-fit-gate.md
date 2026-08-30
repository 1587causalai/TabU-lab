# Stage 3: synthetic-data basic fitting

This gate asks one deliberately narrow question: **can a freshly built
`tabu.cell.base@0.2.0` reduce masked-response loss on a fixed synthetic law?**

The world generator (`tabubase.linear-world.v1`) creates two visible numeric
causes and a response:

$$y = 0.8x_0 - 0.4x_1 + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, 0.1^2).$$

The response is masked in the model carrier. Its full value exists only in an
evaluation sidecar used to compute standardized numeric MSE. Hidden cells are
physical zeros, so the truth cannot leak through `DenseModelInput.values`.

The runner uses one fixed training world and a separate fixed validation world,
80 Adam steps, and a small deterministic TabUBase configuration. The minimum
gate is finite loss plus a strict decrease in training loss. Validation loss is
reported as a held-out sanity signal, not as a claim of broad generalization.

Run it with:

```bash
uv run python scripts/run_tabubase_synthetic_fit.py
```

The JSON result is marked `local_unissued`. It is local diagnostic evidence only;
it does not establish real-data prediction, frozen ICL, fine-tuning lift, or
foundation-model capability.
