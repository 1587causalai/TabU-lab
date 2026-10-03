# V7 single, joint and mixed restoration

The V7 implementation uses one `V7Model` and one optimizer for all three
training modes. The joint numerical path was promoted from the audited 2026-10-03
single/joint experiment. Mixed sampling is available for experiments; its quality
advantage has **not** been established by that single-versus-joint comparison.

## Configuration and command

Start with [the runnable YAML](../../examples/v7-restoration/mixed.yaml) and its
small synthetic table. Paths inside YAML resolve relative to that file.

```bash
# Inspect the resolved run without training or creating outputs.
python -m tabu_lab.cli restoration v7-fit --config examples/v7-restoration/mixed.yaml

# Execute the explicitly bounded eight-update fixture.
python -m tabu_lab.cli restoration v7-fit --config examples/v7-restoration/mixed.yaml --execute
```

On configured training hosts use their existing launcher instead of generic
Python: `~/.local/bin/wehub-python --profile train-20260920 -m tabu_lab.cli ...`,
from the deployed project/environment that contains this package. Deploy the selected source revision on the host before running; a library update
does not update existing experiment checkouts.

Only `masking.mode` needs to change to switch modes:

| Mode | Each training episode |
| --- | --- |
| `single` | Uniformly select one eligible column and hide its Query rows |
| `joint` | Select `joint_columns` eligible columns and hide the same Query rows in all of them |
| `mixed` | With `joint_probability`, draw a joint episode; otherwise draw a single episode |

`joint_probability: 0.5` means 50% probability **per episode**. With three-column
joint episodes it does not mean half the Query cells. Finite runs need not realize
an exact 50/50 count. The actual counts and cell exposure are saved.

`columns: [0, 1, target]` defines eligible columns. Use column names or indices;
`target` resolves to each table's target (last column unless explicitly set).
For target-only supervised adaptation use `mode: single, columns: [target]`.
Omit `columns` to use all columns. A table entry may override `columns` for a
heterogeneous corpus. `joint_columns: null` uses all eligible columns; insufficient
eligible columns, duplicates, unknown fields and invalid row budgets fail explicitly.

Tables follow `tabu.tar.typed-fit-table.1`, or the existing compatible OpenML table
format, with `features`, `values`, and disjoint `splits.train/test` row addresses.
Tables receive equal visits in shuffled cycles. Episode draws are deterministic
from seed and step and use training rows only. The window is capped by the number
of training rows; at least two supports must remain. A numeric queried column needs
two distinct visible support values. A hidden nominal category without a visible
code fails training admission/scoring rather than being dropped or silently resampled.

## Model and loss

Mask selection is separate from the architecture. Current cyclic runs explicitly
set `query_init: donor`, `query_source: true`, `rounds: 4`, and `share_rounds: true`.
The constructor retains historical seed / fixed-source / K1 defaults so older
single-column callers and native checkpoints retain their meaning. All modes
support the same model parameters; `unit_layers` and the backbone shape remain
explicit model choices.

Each joint round runs the backbone once, fits each queried column's LL using only
its own true visible supports, then writes every Query recovery simultaneously.
Recovered Query cells can be sources in the next round, but never become LL
supports. Observed codes are clamped; the complete cyclic path retains gradients.
The loss averages queried cells within each Query row, then averages Query rows,
then combines rounds geometrically. Single-column loss preserves its former
reduction order. The fixture's numeric/discrete coefficients 32 / 0.25 match the
scale12 comparison, not the general constructor's 128 / 1 defaults.

The public Python API also supports different Query rows in different columns:

```python
from tabu_lab.models.restoration_v7 import (
    table_mask_task, train_step, evaluate_joint_task,
)

task = table_mask_task(
    table, rows=[0, 1, 2, 3, 4, 5],
    query_by_column={0: [4], 2: [4, 5]}, code_seed=7, donor_seed=8,
)
train_step(model, optimizer, [task], grad_clip_norm=1.0)
report = evaluate_joint_task(model, task)  # report.columns[column].predictions for all rounds
```

The old `table_task`, `prepare_episode`, and single-column `evaluate_task` result
remain available. `train_step` and `evaluate_task` dispatch joint masks automatically;
`evaluate_joint_task` always returns a column-indexed report, even for one column.

## Execution, checkpoints and evaluation

Choose `device: mps, dtype: float32` for Apple training; use the configured
`cuda/float64` on the existing CUDA training hosts. No device fallback is performed.
`gradient_checkpointing: true` recomputes the backbone during backward to save
memory. The small CPU fixture is an implementation check, not a trained candidate.

To initialize from an existing V7 checkpoint, add `init_checkpoint: /path/parent.pt`.
The runner inherits the checkpoint ModelSpec, then applies explicit `model`
overrides and loads weights strictly by name/shape. For a real parent, remove the
fixture's tiny backbone and coupling dimensions; retain only intended overrides,
for example:

```yaml
init_checkpoint: /path/parent.pt
model:
  query_init: donor
  query_source: true
  rounds: 4
  share_rounds: true
  gradient_checkpointing: true
```

This is weights-only initialization with fresh optimizer/RNG/sampling. Historical
experimental checkpoints may omit Query-as-source metadata, so explicitly declare
it as above. Their optimizer format is not assumed to be a native strict resume.

Native continuation uses the same YAML, a larger total `steps`, and a **new** output
directory:

```bash
python -m tabu_lab.cli restoration v7-fit --config run.yaml --execute \
  --resume /path/previous/final.pt --output-dir /path/continuation
```

Strict resume requires the same model, masking, optimizer, data/split, seed,
evaluation definition, source digest, device/dtype and Torch version. It restores
weights, AdamW, CPU/device RNG and the next sampler step. Changing single/joint/mixed
mode is a new objective: use weights-only initialization for that change.

Outputs include resolved configuration, per-update row/column addresses and losses,
actual single/joint counts and Query-cell exposure, `final.pt`, `evaluation.json`,
and a terminal receipt. Existing output directories are rejected. Final evaluation
uses a deterministic same-table heldout bank for target-only and joint missingness,
reports every recovered column and round, and does no checkpoint selection. An
empty test split is recorded as `no-heldout-rows`; it is not scored on training rows.
This runner is step-bounded; the frozen experiment's time/exposure checkpoint grid,
watchdog, and two-stream schedule remain in that experiment's archive.

Validation of this integration is recorded in
[the integration check](../reviews/v7-restoration-modes-20261003.md).
