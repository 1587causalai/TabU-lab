# Old120 Small-128: global 2.5% Query masks

This recipe supersedes the per-column 68-cell masking experiment. It starts
from seed 1729 in a separate run, with the same 26,824-second / 768-round ceiling
and 300-second finalization reserve. The earlier run was stopped and retained;
its weights are not a continuation under this changed protocol.

## Mask budget

Historical TAR hid 68 target-column cells per table: 8,160 cells over one full
120-table round. There are 402,288 training cells, so the corresponding
corpus-wide fraction is 2.0284%. The new fraction is 2.5% of each entire table,
rounded to the nearest integer (half up): 10,054 cells per round, or 2.4992%
after rounding, 23.21% more than historical TAR. This matches the corpus-wide
fraction approximately; it does not equate every table's Query count.

| Columns | Tables | Query cells per table |
| --- | ---: | ---: |
| 6 | 10 | 31 |
| 8 | 16 | 41 |
| 12 | 36 | 61 |
| 20 | 42 | 102 |
| 32 | 16 | 163 |

The sampler first randomly protects a visible representative of every observed
discrete class, including all singletons. It draws the total Query budget across
the remaining cells without a fixed quota per column, retaining at least two
visible cells in every column. A column can have zero Query cells. Uniformity
is conditional on the protected supports and visibility constraints.

B=1 and all 204 training rows / all columns remain in each update. The scorer
still trains on all observed cells, including retained and Query cells. The 52
reserved rows are excluded. Model size, FP64, optimizer and loss weights remain
as declared in the [preregistration](preregistration.yaml).

## Monitoring definitions

- **Training:** after all 120 tables complete, report the mean, median and P95
  of their 120 pre-update scalar losses. Each table has equal weight. These are
  training-progress statistics across changing model states within the round.
  Partial-round losses are checkpointed for exact continuation but do not make
  a complete-round point.
- **Fixed evaluation:** eight masks per table are generated once under the new
  policy and reused initially, every 32 rounds and at termination. Within each
  table, pool the eight masks' relevant cells; then take mean, median and P95
  across tables with equal table weight. P95 uses linear interpolation.
- **Separate panels:** Query numeric encoding MSE; Query discrete encoding MSE
  and error rate; retained numeric and discrete metrics. Discrete accuracy is
  also recorded. P95 error rate describes the high-error tail; P95 accuracy
  describes the high-accuracy tail.
- **Coverage:** every macro metric includes contributing/eligible table counts
  and target-cell counts. Missing types or empty Query groups are excluded,
  never filled with zero. Only tables whose eight masks finished contribute.
  Partial evaluations use separate W&B paths and cannot replace full-bank points.
- **Units:** encoding MSE is the main cross-table numeric comparison. The
  existing original-unit numeric MSE remains supplementary because tables have
  different units. Fixed masks do not promise to query every training cell.

W&B paths: `train_round/loss/{mean,median,p95}` and
`evaluation/{query,retained}/{numeric,discrete}/{metric}/{mean,median,p95}`.
The x-axis is `completed_round`. Local `round-metrics.jsonl` and evaluation JSON
files remain the primary records.

```bash
export TABU_LAB_OBSERVER=wandb TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE=1
export WANDB_MODE=online WANDB_ENTITY=zj3712 WANDB_PROJECT=restoration
export WANDB_RUN_ID=old120-small128-global025-20260916
export WANDB_NAME=old120-small128-global025-20260916
export WANDB_RUN_GROUP=old120-small128
uv run tabu-lab restoration joint-fit \
  --preregistration experiments/local/restoration/old120-small128-global-mask/preregistration.yaml \
  --corpus experiments/local/tar-diverse-120-fit/corpus \
  --device cuda:0 --output-root runs/restoration/old120-small128-global025-20260916 \
  --execute
```

[Mask accounting](mask-budget-reference.json) is derived from the frozen corpus.
The time-budget reference is the unchanged
[historical five-segment record](../old120-small128-tar-budget/budget-reference.json).
