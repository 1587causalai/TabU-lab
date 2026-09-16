# Old120 Small-128: threshold-4 numeric Query protection

Current recipe: [preregistration.yaml](preregistration.yaml). Status: prepared,
not launched. The owner selected threshold 4 and requested stopping the earlier
threshold-8 run and integrating the implementation into local main.

This default applies to **table random-cell masking** only. **Table supervised
row masking** (including TAR target-label row masks) defaults to no tail guard:
label magnitude must not determine whether a supervised row becomes Query.
Frozen historical random-cell protocols keep their recorded policies.

Before random-cell masking, protect numeric cells satisfying
`abs(x - train_median) > 4 * max(train_half_IQR, 1e-6)`. Calibration uses only
204 training rows and linear-interpolation quantiles. Equality stays eligible.
These cells remain visible supports and retained targets in all-cell loss.
The model codec still uses actual visible values only. Reserved rows are excluded.
The guard is an episode-construction policy, not an inference-time oracle.

The table-wide Query budget remains 2.5%, without per-column quotas. Discrete
class support and at least two visible cells per column remain required. The
sampler fails if the exact budget cannot be fulfilled. Small-128, FP64, B=1,
AdamW and the 26,824-second / 768-round ceiling are retained in the new recipe.
No existing checkpoint is resumed across configuration/source identities.

Integration validation: 882 repository tests passed, with 9 optional-dependency
skips; changed Python files passed Ruff. The integrated plan independently
reproduced 3,622 protected cells / 884 columns / 110 tables. Main's FP32,
TAR-aligned single-table masking and checkpoint performance changes are retained.
The parent TeX footnote now records threshold 4; XeLaTeX compilation and visual
inspection passed. These checks do not establish a threshold-4 training result.

## Evidence and trade-off

[CPU census](threshold-census.json): threshold 4 protects 3,622 numeric cells
(1.4601% of numeric cells), compared with 598 (0.2411%) at threshold 8.
All 120 tables retain the exact Query budget; all 960 fixed masks passed the
eligibility check. Affected tables/columns are 110/884. The global percentage
hides uneven coverage: 149 of 816 numeric cells in `discoscm_086` are protected.
These are sampling checks, not evidence of threshold-4 training stability.

[Replay of the prior run](prior-run-threshold-audit.json) covered its first 600
updates: 15 of the 20 highest losses queried numeric cells between 4 and 8
half-IQR from the training median. This is association; no counterfactual
model replay established that those values caused the spikes. Residual spikes
also occurred with every Query below threshold 4.

Smaller thresholds narrow Query coverage and may make the task easier. Report
protected counts, Query coverage, retained metrics and both pre/post-clip norms.
Do not interpret a lower Query loss across different eligibility policies as
better restoration ability. The full eight-mask bank remains fixed within a run.

## Historical stop

The threshold-8 run `old120-small128-tailguard8-20260916` stopped gracefully at
update 1,004: 8 complete rounds plus 44 tables. Its final checkpoint was saved
without error; see [stop record](stopped-threshold8-run.json). The original
[threshold-8 recipe and receipts](../old120-small128-tail-guard/README.md) remain
unchanged. No replacement training was started as part of this integration.
