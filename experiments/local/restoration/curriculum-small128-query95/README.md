# Restoration Small-128: column protection and Query weight 0.95

This is a new sampling and objective variant of `../curriculum-small128`, with
the same frozen corpora, three stages, seed 1729, architecture, optimizers, and
62,824-second stage budgets. The historical recipe remains unchanged.

For each nonempty numeric/discrete type, compute retained and Query encoding
MSE separately, then sum `0.05 * retained_mean + 0.95 * query_mean` across
types. Empty type/state groups contribute zero; weights are not redistributed.
Every observed cell remains a target. Query loss backpropagates through the
whole model, not just the learned Query seeds.

The weights specify coefficients, not a guaranteed 5%/95% share of measured
loss or gradient. The old objective weighted states by cell counts within each
type. At 2.5% Query coverage, the new objective substantially increases Query
influence. It is not a claim that loss/gradient spikes will decrease: protected
numeric columns remain visible supports, and LL predictions are not bounded by
the Query-sampling threshold.

For synthetic random-cell episodes, the new recipe protects a whole numeric
column when `population_std > 4 * max(Q75 - Q25, 1e-6)`. Statistics use all
training rows, population variance (`correction=0`), and linearly interpolated
quartiles. Equality is eligible. IQR is the full interquartile range, not
half-IQR. A constant column stays eligible; a nonconstant zero-IQR column uses
the floor. Reserved rows do not enter calibration.

Every cell of a protected column remains visible and a retained target. Other
columns share the original 2.5% whole-table Query budget, with no equal per-column
quota. This replaces the individual-cell tail rule for the new recipe; it does
not combine both filters. Supervised-row episodes continue to ignore tail
protection and select Query rows without consulting label values. An impossible
budget fails explicitly; it never silently shrinks or skips a table. Legacy
recipes preserve their recorded individual-cell or unguarded policies.

The model still uses visible-only median/half-IQR encoding. Column calibration
is a sampling policy only, never a model input or inference-time oracle.

[Column census](column-guard-census.json) verified archived dataset digests and
all 1,920 fixed masks across 240 synthetic tables. Old120 protects 30 numeric
columns in 8 tables; recent120 protects 23 columns in 7 tables. All masks retain
the exact Query budget and have zero protected-column Query violations.
`old120/discoscm_086` protects columns 8, 9, 14 and 20 (one-based), its four numeric
columns. These are sampling checks, not evidence of spike-free training.

Per-update audits record protected column/cell counts and forbidden Query count.
The existing `protected_numeric_tail_cells` metric counts all cells protected
by the policy (whole columns for this variant); explicit column metrics remove
ambiguity. Fixed-evaluation coverage sums counts over masks and tables.

Training, initial/periodic/final evaluation, and CUDA preflight all use the
same declared objective. Unweighted Query/retained accuracy and error metrics
remain available. Updates additionally record `retained_loss`, `query_loss`,
`retained_loss_contribution`, and `query_loss_contribution`; the two weighted
contributions sum to the optimization loss. The raw losses sum the respective
per-type state means, so they differ from cell-pooled state encoding MSE.

Use a fresh initialization and a new output directory and W&B run. Source,
preregistration, and resolved objective are checkpoint identity fields; an old
objective checkpoint cannot be resumed directly under this recipe. Same-recipe
resume and stage handoff continue to use the declared weights.

```text
uv run tabu-lab restoration curriculum-fit \
  --preregistration experiments/local/restoration/curriculum-small128-query95/preregistration.yaml \
  --corpus-root /path/to/frozen-curriculum-corpora \
  --output-root /path/to/new-attempt
```

The command plans only unless `--execute` is supplied. Complete CUDA preflight
and resource checks before execution. Total optimization losses differ between
objective variants. Changing the guard also changes eligible Query cells and
fixed-mask addresses: even unweighted errors do not establish a matched-mask
improvement without an additional common eligible evaluation set.

Validation for this implementation: repository checks `943 passed, 9 skipped`
(optional sklearn/xgboost dependencies), scoped Ruff and diff checks passed,
and an independent review found no blockers (64 focused tests). The final mask
source rechecked all 1,920 census masks. The matching design manuscript was
compiled with XeLaTeX and both changed pages were visually checked. These are
implementation and sampling results; this recipe has not started training.
