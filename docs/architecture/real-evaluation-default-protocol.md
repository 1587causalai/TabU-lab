# Real-data evaluation default

## Decision

All new real-data model evaluations use the full train/test estimand by
default.  For one deterministic split, let $D = D_{\mathrm{train}} \cup
D_{\mathrm{test}}$.  The model receives

$$
E = \{(x_i, y_i): i \in D_{\mathrm{train}}\}
    \cup \{(x_j, \square): j \in D_{\mathrm{test}}\},
$$

and is scored on every held-out target $y_j$ in $D_{\mathrm{test}}$.  The
classical baselines fit exactly the same $D_{\mathrm{train}}$ and are scored
on exactly the same $D_{\mathrm{test}}$.

The runtime defaults are therefore:

- `context_policy = full_train`;
- `query_policy = all_heldout_rows`;
- `query_limit = null`;
- finite `label_budget` or `test_limit` only when explicitly requested as a
  bounded diagnostic override.

`query_chunk_rows` may bound only the terminal readout.  It must not split,
truncate, or rebuild the evidence episode.

## Metric contract

Metrics are computed on the same held-out rows for every arm and baseline; the
classification and regression tasks are summarized separately.  The primary
classification metric is normalized negative log-likelihood,

$$
\operatorname{nNLL} = -\frac{1}{n\log C}
  \sum_{j=1}^{n}\log p_{j,y_j},
$$

with accuracy, balanced accuracy, macro-F1, raw log-loss, and ROC-AUC reported
as secondary diagnostics.  The primary regression metric is target-scale
normalized RMSE,

$$
\operatorname{sRMSE} =
\frac{\sqrt{n^{-1}\sum_{j=1}^{n}(\hat y_j-y_j)^2}}
     {\max(\operatorname{std}(y_{\mathrm{train}}),\epsilon)},
$$

with raw RMSE, MAE, scaled MAE, and $R^2$ reported alongside it.  A single
cross-task average, a context-size AULC, or a fixed-$K$ endpoint is not a
full-context capability metric.

## Ripple-effect repair

The query-row OpenML `K={0,1,2,4,8,16,32}` runner, manifest, preregistration,
and CLI were removed from the active public surface.  They encoded a low-shot
estimand that could be mistaken for the main table-foundation-model result.
Shared checkpoint and baseline code now lives in
`query_row_transfer_common.py`, which has no context-size policy.

Historical Axis-B low-shot receipts and synthetic context curves are retained
only where their own contract explicitly names them as diagnostics.  They are
not valid evidence for this full-context estimand and must not be aggregated
with it.

Stage-4 scratch and Stage-6 fine-tuning entry points now default to all train
rows and all held-out rows.  A finite budget remains available only as an
explicit, visibly bounded diagnostic invocation.

## Reporting rule

Every result must report the split seed, train-row count, held-out-row count,
primary task metric, and whether the run is `local_unissued` or formal.  A
structural `status = passed` means only that the harness and integrity gates
passed; it is not a capability pass.  Frozen ICL and fine-tuning must remain
separate rows in the evaluation ladder.
