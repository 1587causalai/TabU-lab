# Research reports

[Research questions](../research-questions/README.md) groups the findings by
question. The [design index](../design/README.md) ties each result to its model
version and implementation status.

## Published results

| Version | Report | Scope |
| --- | --- | --- |
| V10.4 | [CellRidge TabArena-Lite](https://github.com/1587causalai/kumo-small-cellridge-1024/blob/main/LITE.md), [8,192-support comparison](https://github.com/1587causalai/kumo-small-cellridge-1024/blob/main/support-8192/README.md), [checkpoint selection disclosure](https://github.com/1587causalai/kumo-small-cellridge-1024/blob/main/DEVELOPMENT.md) | Local capped-support runs on development-exposed splits. TabArena closed [PR #735](https://github.com/autogluon/tabarena/pull/735) after comparing the submitted Lite results to Kumo Small. |
| V10.1 | [Matched-task comparison](https://research.wehub.us/tabu-lab/experiments/research-update-20261029/) and [training bottlenecks](https://research.wehub.us/tabu-lab/blog/speed-bottleneck-breakthrough-20261029/) | Learning and cost under the stated protocols, with separate reports for predictive results and timing. |
| V9 | [Reconstruction and new observations](https://research.wehub.us/tabu-lab/experiments/v9-reconstruction/) | Whole-observation reconstruction and prediction on fresh observations from the same worlds. |
| V8.4 | [Direct value-code readout](https://research.wehub.us/tabu-lab/experiments/v842-value-code/) and [loss balancing](https://research.wehub.us/tabu-lab/experiments/v8402-loss-balance/) | Two supervised comparisons of readout and loss choices. |
| V8.2 | [Learning and retention](https://research.wehub.us/tabu-lab/experiments/v82-learning-ladder/) and [targeted training](https://research.wehub.us/tabu-lab/experiments/v82-targeted-training/) | Learning new synthetic worlds and testing whether more exposure to selected tables improves held-out prediction. |

## Historical repository reports

- [V5.4 old120 fitting handoff](v54-old120-fit-handoff-2026-09-23.md) records
  the state of the training runs on September 23, 2026. The
  [old120 replay rollout](v54-old120-replay-rollout-2026-09-22.md) and
  [replay v2 contract](../../experiments/local/v54-old120-replay-v2-20260922/README.md)
  document the follow-up budgets and replay policy.
- [TAR 120-table shared fit](../research/tar-shared-fit-20260907/README.md)
  preserves fixed-training-table evidence for the earlier TabU-TAR model.
- Earlier query-family program comparisons are indexed in the
  [evolution impact reports](evolution-impact/README.md). See the
  [experiment archive](../../experiments/README.md) for their protocols.
