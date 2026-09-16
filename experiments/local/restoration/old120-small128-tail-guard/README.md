# Old120 Small-128: protect numeric tails from Query masking

This is a fresh seed-1729 run of all 120 tables. It keeps the Small-128 FP64
model, B=1, 204 training rows, every column, all-observed-cell loss, and the
26,824-second / 768-round ceiling with a 300-second finalization reserve.
The [preregistration](preregistration.yaml) declares protocol v4. The preceding
[global-mask run](../old120-small128-global-mask/README.md) stopped at update
2,338 (19 complete rounds plus 58 tables), with its checkpoint preserved.
Its weights and historical receipts are not reused under this new policy.

## Numeric eligibility

Before artificial masking, compute each numeric column's median and half-IQR
from its fixed 204 training rows, with linear-interpolation quantiles. A cell
stays visible when

```text
abs(value - train_median) > 8 * max(train_half_IQR, encoder_epsilon)
encoder_epsilon = 1e-6
```

Equality remains eligible. The floor also handles constant or zero-IQR columns.
The threshold is explicit and configurable in `numeric_query_guard`; v2/v3
reject that field so their historical mask semantics cannot silently change.
Nonfinite inputs/calibration or an impossible Query budget fail before training.

The policy asks Query seeds to recover the eligible range. Protected extremes
remain visible supports and retained targets in the full observed-cell loss.
There is no value/prediction clipping, loss reweighting, deletion of rows or
tables, or change to the numeric encoder. The encoder still computes its own
median/half-IQR from the actual visible cells only. Pre-mask calibration belongs
to episode construction and is not passed into the model. The 52 reserved rows
are excluded. Unknown inference-time values cannot be screened with this rule.

The global budget remains 2.5% of **all** cells in each table, rounded half up:
10,054 Query cells over a complete round. After protecting numeric tails and one
visible representative of every observed discrete class, draw that exact budget
across eligible cells, retaining at least two visible cells per column. There is
no per-column quota and no silent reduction of the budget.

## Diagnosis and interpretation

The [old-run audit](spike-audit.json) replayed all 2,338 recorded masks and code
seeds exactly. Eighteen of the 20 largest losses queried a cell beyond the new
threshold. The rule protects 598 cells across 59 numeric columns in 19 tables:
0.2411% of the 248,064 numeric training cells, or 0.1487% of all 402,288 cells.
This is association; no model counterfactual replay was performed. Two high-loss
updates did not query these extremes, and retained/source effects can still
produce spikes. This run tests stability under the changed task distribution.

## Monitoring and coverage

- Every completed 120-table round records mean, median and P95 of pre-update
  losses, with equal table weight. Partial rounds do not produce a full-round
  point; their state is checkpointed for exact continuation.
- Each table has eight new fixed evaluation masks under the same tail policy,
  reused initially, every 32 rounds and at termination. Pool the eight masks
  within a table, then summarize across tables equally. Query numeric, Query
  discrete, and retained metrics have separate panels. Original-unit numeric
  MSE and discrete accuracy remain in the receipts. These Query scores describe
  the eligible domain; they do not evaluate the protected tails or the reserved
  rows, and are not directly equivalent to the preceding run's Query scores.
- `gradient_norm` is the norm **before** clipping.
  `post_clip_gradient_norm` measures the actual gradients after `clip=1`, before
  AdamW. Both have separate panels; a large pre-clip value does not mean an
  unclipped gradient reached the optimizer.
- Each update records protected counts by column/table and
  `query_numeric_tail_cells`, which must be zero. Plan
  `numeric_tail_coverage` counts unique cells once. Evaluation
  `coverage/protected_numeric_tail_cells` counts mask-cell exposures, hence
  4,784 = 598 x 8 for a complete bank. A partial bank counts only evaluated
  episodes. Divide by the matching cell/exposure denominator.

The report generator [build_report.py](build_report.py) has ten panels: seven
round/evaluation panels with mean/median/P95, two gradient panels and one
protected-tail Query violation panel. It filters one exact run ID. Default
execution only validates the report schema; `--save` creates it in W&B.
Local metrics, checkpoints and terminal receipts remain authoritative.

The unchanged budget sources are the
[TAR five-segment time record](../old120-small128-tar-budget/budget-reference.json)
and [global-mask cell accounting](../old120-small128-global-mask/mask-budget-reference.json).

## Local validation

The full repository suite passed: 877 tests, with 9 optional-dependency skips.
Ruff and whitespace checks passed. The [independent review](independent-review.json)
and [960-mask census](independent-mask-census.json) bind the reviewed executable
source hash. The census checked visibility, full target coverage, zero hidden
protected tails, and legacy v3 mask/metadata equality. All 1,216 numeric columns
also matched the existing codec's median/half-IQR threshold calculation.
These are implementation checks; launch and training results require separate
runtime receipts.

## Execution

On the final committed source, run the CUDA forward/backward/continuation check
and the largest-table update before starting the new run:

```bash
uv run --no-sync tabu-lab restoration verify --device cuda:0 --output runs/restoration/device-check.json
uv run --no-sync tabu-lab restoration joint-fit-preflight \
  --preregistration experiments/local/restoration/old120-small128-tail-guard/preregistration.yaml \
  --corpus experiments/local/tar-diverse-120-fit/corpus \
  --device cuda:0 --output runs/restoration/largest-table-check.json
uv run --no-sync tabu-lab restoration joint-fit \
  --preregistration experiments/local/restoration/old120-small128-tail-guard/preregistration.yaml \
  --corpus experiments/local/tar-diverse-120-fit/corpus \
  --device cuda:0 --output-root runs/restoration/old120-small128-tailguard8-20260916 \
  --execute
```

Use the existing SDK authentication and W&B entity/project configuration.
Set a fresh `WANDB_RUN_ID=old120-small128-tailguard8-20260916`. Source, corpus and
configuration identities must match for continuation; the older run cannot be
resumed into this protocol. This remains `local_unissued` fitting evidence.
