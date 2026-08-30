# TabUBase real frozen-ICL estimand correction

Date: 2026-08-30  
Status: protocol and implementation corrected; full-context evaluation not run  
Evidence status: `local_unissued`

## Error and affected evidence

The historical real-data evaluator used
$K\in\{0,1,2,4,8,16,32\}$ labeled context rows and a limited held-out query
subset. That design is a legitimate low-shot context-scaling diagnostic, but it
was incorrectly discussed as though it answered the intended downstream
frozen-ICL question. It does not.

Affected evidence includes the old6 and OpenML new6 real receipts, their AULC
comparisons, and their $K=32$ endpoint summaries. Their bytes, hashes, and
numbers are preserved; their claim role is changed to
`auxiliary_low_shot_diagnostic_only`. Synthetic held-out $K$ curves are not
invalidated because their estimand is explicitly context-length scaling.

## Correct primary estimand

For each deterministic split $s$:

$$
T_s\cap Q_s=\varnothing,
\qquad K_s=|T_s|.
$$

Every row in $T_s$ is supplied with its response label. Every row in $Q_s$ is
evaluated with predictors visible and response truth absent from the evidence
tensor. All held-out predictors remain in one transductive evidence episode.
Only the query-response terminal is processed in bounded chunks after shared
dynamics, so chunk size cannot change the evidence set.

There is no optimizer and no parameter update:

$$
\theta_{\mathrm{after}}=\theta_{\mathrm{before}}.
$$

Each frozen arm records adjacent before/after state hashes, and the aggregate
unchanged gate must be true. Classification reports normalized NLL and
accuracy. Regression reports scaled RMSE, scaled MAE, and held-out $R^2$.

## Implemented ripple changes

- `full_train` is now the default real-evaluation context policy.
- Full-context runs reject any finite `query_limit`; all held-out rows are
  evaluated.
- Episode construction rejects a context smaller than the complete train
  partition under `full_train`.
- Readout chunking occurs only after one complete transductive evidence episode
  and never truncates the train context or held-out query set.
- Full-context receipts use a distinct schema and dataset/split-specific
  context-row ledger.
- The paired comparator has a separate full-context path and does not compute
  AULC or a $K=32$ endpoint for this one-point estimand.
- Regression comparison includes $R^2$ delta and scaled-MAE gain; classification
  includes accuracy delta.
- The old low-shot path remains available only through the explicit
  `low_shot_grid` policy for reproducibility.

## Runtime and evidence boundary

Dense routing may be unable to fit the largest full contexts. Such a result is
`blocked`, not a license to shrink the context. Query-target-only terminal
chunking removes the response-terminal $N\times N$ ledger while preserving the
shared evidence computation; additional streaming/exact long-context work may
still be required before all datasets can complete.

No full-context checkpoint evaluation or empirical metric is claimed by this
correction. The governing candidate manifests are:

- [`real-full-context-frozen-icl.yaml`](../../experiments/transfer-base-v2/real-full-context-frozen-icl.yaml)
- [`real-full-context-frozen-icl-openml-new6.yaml`](../../experiments/transfer-base-v2/real-full-context-frozen-icl-openml-new6.yaml)
