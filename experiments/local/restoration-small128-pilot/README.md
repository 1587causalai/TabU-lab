# Small-128 restoration pilot

This prepares a bounded, fresh single-pass restoration fit using the width,
depth, heads, FFN width and optimizer starting values of historical TAR Small-128.
It does not reuse TAR weights, its response-only task, its Muon parameter groups,
or its training budget. The new model uses the current restoration semantics.

## Frozen first attempt

| Setting | Value |
| --- | --- |
| Model | width 128, 3 layers, 8 heads, FFN 256; inducing backbone; 1,590,848 parameters |
| Restoration defaults | 256 inducing slots, presence tau 1, reference mass 1 |
| Readout | All LL, bandwidth 1, ridge 0.001 |
| Numeric codec | Visible median / half-IQR, scale floor 0.000001 |
| Data | Historical old120 `scm_mixed_v1_000`; all eight numeric columns |
| Rows | 32 rows selected from the original 204 training rows; 52 reserved rows unused |
| Masks | Four fixed cell-mask episodes, approximately one third Query per column |
| Optimizer | AdamW, LR 0.0001, betas 0.9/0.95, epsilon 0.00000001, no decay, clip 1 |
| Batch and precision | One episode per update, CPU/CUDA FP64 reference |
| Seed | Model 1729; rows 1730; masks 1731; codes 1732 |
| Budget | 200 total updates or 600 cumulative attempt seconds, checked between operations |
| Output | Initial/final retained and Query fit metrics, update log, checkpoints, terminal outcome |

The first table and columns are selected to check the complete numeric training
path. This pilot cannot establish mixed-type training, multi-table performance,
held-out-row prediction or unseen-table generalization. Its fixed fit-mask bank
is also the training bank. Any Query-error decrease is therefore a fit result.

The zero weight decay is an explicit diagnostic choice. Historical TAR used
matrix weight decay and later a mixed Muon/AdamW optimizer; those parameter
groups have different meanings in this architecture. The current 256 inducing
slots also differ from historical TAR's 128. Its parameter count consequently
differs from historical TAR Small-128's 1,267,136. These choices must be recorded
in comparisons, even though the carrier and FFN dimensions match.

## Inspect and execute

Use an isolated source checkout and set `RESTORATION_DATA` to the existing
snapshot file. The preregistration binds its exact bytes. Do not download a new
version, regenerate its split, or copy reserved rows into training.

```bash
uv run tabu-lab restoration fit \
  --preregistration experiments/local/restoration-small128-pilot/preregistration.yaml \
  --dataset "$RESTORATION_DATA" \
  --device cuda:0 \
  --output-root "$RESTORATION_OUTPUT"
```

The default command only resolves and validates the plan. It neither allocates
the training model nor writes a run directory. Execution requires an explicit
`--execute`, a committed preregistration, an available accelerator for CUDA,
and a new output directory. Device qualification and a fresh resource check
must precede a target-device fit. No experiment has been run by creating this
configuration.

For continuation, use the same preregistration and snapshot, add
`--resume /path/to/checkpoint.pt`, and choose a new output directory. Model,
optimizer, RNG, data, source and execution identities must agree. Resume does
not authorize a larger total update or time budget.

## After the pilot

First inspect finite behavior, Query-error change, time per update, memory and
checkpoint continuation. Then preregister a separate scale probe at the old
204-training-row size and representative column counts. Expanding to old120,
recent120 or real tables additionally requires a mixed-type sampler, declared
code coverage, and a new multi-table protocol. The old long-run schedules and
error-driven table weights are historical references, not defaults here.
