# old120 Small-128 restoration joint fit

Historical v1 pilot: the commands and unchanged preregistration below require
source commit `def4954`. The current v2 runner rejects the v1 protocol and its
checkpoints. For a new run, use the
[current restoration budget recipe](../restoration/README.md).

This is the first multi-table fit diagnostic for the five-step table-restoration
model. It uses all 120 frozen old120 tables and every row in each table's original
204-row training split. The 52 reserved rows remain outside the model and all
evaluation masks.

The two-round segment is intentionally bounded: one shared fresh model and one
AdamW optimizer perform one update per table in a deterministic order per round,
for 240 updates total or at most one hour. Initial and end-of-round diagnostics
reuse eight fixed masks per table. They report retained and Query encoding MSE,
numeric original-unit MSE, discrete accuracy, and coverage by column type.

The corpus contains numeric, nominal, and ordinal columns. Each Query mask hides
68 cells per column. For discrete columns, one training-row occurrence of every
observed class is protected as visible evidence before sampling the remaining
Query cells. This is required by the visible-only answer codec; it is not a
robustness claim. Tables with sparse or singleton classes are reported in the
mask metadata and are not silently removed from the loss.

## Plan and execution

From the repository root:

```bash
uv run tabu-lab restoration joint-fit \
  --preregistration experiments/local/restoration-old120-joint-fit/preregistration.yaml \
  --corpus experiments/local/tar-diverse-120-fit/corpus \
  --output-root runs/restoration-old120-small128-2rounds
```

The command above only validates the plan and prints its resolved identity. It
does not allocate the model or create an output directory. Execution is explicit:

```bash
uv run tabu-lab restoration joint-fit \
  --preregistration experiments/local/restoration-old120-joint-fit/preregistration.yaml \
  --corpus experiments/local/tar-diverse-120-fit/corpus \
  --device cuda:0 \
  --output-root runs/restoration-old120-small128-2rounds \
  --execute
```

The preregistration must be committed, the output directory must be new, and a
fresh CUDA/resource check must pass before execution. Resume uses the same
preregistration, corpus, source identity, model configuration, and a new output
directory:

```bash
uv run tabu-lab restoration joint-fit \
  --preregistration experiments/local/restoration-old120-joint-fit/preregistration.yaml \
  --corpus experiments/local/tar-diverse-120-fit/corpus \
  --device cuda:0 \
  --output-root runs/restoration-old120-small128-resume \
  --resume-checkpoint runs/restoration-old120-small128-2rounds/checkpoint-round-0001.pt \
  --execute
```

The output keeps `resolved.json`, fixed evaluation masks, per-update `updates.jsonl`,
per-round metrics, checkpoints (including an atomically updated
`checkpoint-progress.pt` for cross-table interruption recovery), and a terminal
receipt. Metrics are available by table, source family, and column type. A result
is a bounded fixed-table fit observation; it does not establish held-out-row or
unseen-table generalization, benchmark readiness, or a public capability claim.
