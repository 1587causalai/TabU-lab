# TabU design and version index

[Documentation](../README.md) · [Research questions](../research-questions/README.md) · [V11 research progress](../../README.md#v11-research-roadmap)

Use this index to find the implementation, design, or result for a specific
version. For checkpoint compatibility, use the configuration and source
revision in that checkpoint's record.

## Published research

### V10.4 · Kumo Small + CellRidge-1024

The first TabArena submission pairs the NVIDIA Kumo Tabular Small backbone
with four additional row blocks and an affine ridge readout. The
[v1.0 release](https://github.com/1587causalai/kumo-small-cellridge-1024/releases/tag/v1.0)
contains separate classification and regression inference weights under
OpenMDW-1.1. The [release manifest](https://github.com/1587causalai/kumo-small-cellridge-1024/blob/v1.0/weights.json)
records their checkpoint identities and digests.

TabArena closed [PR #735](https://github.com/autogluon/tabarena/pull/735) on
2026-11-20 (UTC+8). The [maintainer’s note](https://github.com/autogluon/tabarena/pull/735#issuecomment-7015532014)
cites the local Lite results, where the best measured configuration trailed
the parent Kumo Small model and default CatBoost.
The [submitted adapter](https://github.com/autogluon/tabarena/pull/735/files)
defaults to a cap of 200,000 labeled support rows. The local TabArena-Lite
reports use the same frozen weights with caps of
[512 and 2,048 rows](https://github.com/1587causalai/kumo-small-cellridge-1024/blob/main/LITE.md)
and [8,192 rows](https://github.com/1587causalai/kumo-small-cellridge-1024/blob/main/support-8192/README.md).
These splits informed development and checkpoint selection.
The published results cover the measured caps above. A full run at the
200,000-row default remains unreported.

### V10.1 · Matched tasks and training cost

The [October comparison](https://research.wehub.us/tabu-lab/experiments/research-update-20261029/)
reports a matched-task experiment with TabU V10.1, TabICLv2, and Kumo Small.
The models use their own learning-rate schedules and losses, so the results
describe the systems under that protocol. The report identifies gaps in
numeric regression and training cost. The
[training-speed retrospective](https://research.wehub.us/tabu-lab/blog/speed-bottleneck-breakthrough-20261029/)
documents the timing comparisons. Whether the faster candidate preserves
predictive performance remains open. These links lead to research summaries.

For the TabICLv2 authors' results and implementation, see their
[paper](https://arxiv.org/abs/2602.11139v2) and
[official repository](https://github.com/soda-inria/tabicl).

### V9 and V8.4 · Earlier experiments

- **V9** tested whether a model trained to reconstruct visible values also
  improved predictions for fresh observations. The tested setup improved
  reconstruction while prediction on new observations changed little.
  [V9 reconstruction study](https://research.wehub.us/tabu-lab/experiments/v9-reconstruction/)
- **V8.4** explored supervised readout and loss choices. The
  [V8.4.2 direct-readout study](https://research.wehub.us/tabu-lab/experiments/v842-value-code/)
  and [V8.4.0.2 loss-balancing study](https://research.wehub.us/tabu-lab/experiments/v8402-loss-balance/)
  report the measured tradeoffs and limitations.

## Implementations and archives

| Generation | Where to start | What the record covers |
| --- | --- | --- |
| V8, V8.1, V8.2 | [Completion API](../../src/tabu_lab/models/completion_v8/README.md), [V8 integration](../reviews/v8-integration-20261011/README.md), [V8.2 validation](../reviews/v82-integration-20261012/README.md) | The merged masked-completion baseline and optional learned feature codes, linear row attention, and supervised attention. |
| V7 and V7.3 | [Restoration guide](../tutorials/v7-restoration-modes.md), [integration](../reviews/v73-integration-20261008/README.md), [reversible row encoder](../reviews/v73-row-reversible64-20261008/README.md) | Cyclic restoration, masking modes, and opt-in row encoders. The reports document scoped checks and migration experiments. |
| V6 | [Model contract](restoration-v6.md), [experiment archive](../../experiments/history/v6-20260930/README.md) | Target broadcasting, experiment source snapshots, and replay procedures. |
| V5.3 | [Implementation design](restoration-v53.md), [curriculum guide](../tutorials/v53-curriculum.md) | Affine numeric encoding, column-shared recovery, and a reproducible curriculum fixture. |
| V5.4 | [Implementation design](restoration-v54.md), [frozen TeX and PDF package](TabU_V5p4_Unified_Composition/README.md), [CPU example](../../examples/restoration_v54_smoke.py) | Shared column codes, model sizes, and supervised-row episodes. |
| V5.5 | [Implementation design](restoration-v55.md) | Explicit codec and Unit-depth versions, loss choices, and checkpoint continuation rules. |
| V4 · TabU-TAR | [TAR guide](../tutorials/tabu-tar.md), [design snapshot](#tar-implementation-design-snapshot), [120-table fit report](../research/tar-shared-fit-20260907/README.md) | Typed additive readout and historical fitting on a fixed training set. |
| V3 · Cell as Query | [Compatibility runtime](../history/compatibility-runtime.md), [query-model mapping](../architecture/query-model-runtime-mapping.md) | Query-family designs and the historical TabU-v2 contract. The runtime factory default is `tabu.v2.tabur`. |
| V2 · Cell as Unit | [Cell-base ModelSpec](../../specs/models/tabu.cell.base.yaml), [query-model mapping](../architecture/query-model-runtime-mapping.md) | The earlier Cell-as-Unit contract and its relationship to the query family. |
| V1 · Task-specific models | [TabUFL spec](../../history/imports/donor-3502fdd/model-verification/specs/models/tabufl.yaml), [TabU4Graph spec](../../history/imports/donor-3502fdd/model-verification/specs/models/tabu4graph.yaml) | Archived verification specs for early model variants. |

### More ways to find the evidence

- [Five-step restoration guide](../tutorials/table-restoration.md),
  [implementation review](../reviews/restoration-five-step-20260915/README.md),
  and [execution review](../reviews/restoration-prepared-20260916/README.md)
  explain the reference implementation and what was checked.
- The V5.4 [training and monitoring playbook](../guides/v54-training-and-monitoring.md),
  [fitting handoff](../reports/v54-old120-fit-handoff-2026-09-23.md),
  [model-growth notes](../research/v54-model-growth/README.md), and
  [next-design handoff](vnext-research-handoff-2026-09-23.md) preserve that
  research stage.
- The [pretraining curriculum source](pretraining-curriculum.tex) and
  [Python interface](../tutorials/model-independent-curriculum.md) describe
  explicitly versioned data, probes, and adapters.
- [Program manifests](../architecture/evolvable-pretraining-programs.md),
  [evaluation protocol](../architecture/real-evaluation-default-protocol.md),
  [compiler boundary](../architecture/compiler-data-boundary.md),
  [evidence semantics](../architecture/evidence-core.md), and the
  [catalog](../../catalog.json) explain how experiments are recorded.
  [MAINLINE.yaml](../../MAINLINE.yaml) selects the query-family pretraining programs.
- Browse the [experiment archive](../../experiments/README.md),
  [restoration experiments](../../experiments/local/restoration/README.md),
  [reports](../reports/README.md), [reviews](../reviews/), and
  [model specs](../../specs/models/) for the dated source records.

## TAR implementation design snapshot

[tar-model-design.tex](tar-model-design.tex) is a read-only snapshot of the
owner-maintained TAR design with embedded TikZ figures.
[tar-defaults.json](tar-defaults.json) captures the same Standard configuration.
Small and Medium are separate runtime presets. To compile the snapshot locally,
run `latexmk -xelatex -synctex=1 tar-model-design.tex` in this directory.

The [TAR ModelSpec](../../specs/models/tabu.tar.yaml) binds the entrypoint digest
and recursive source digest. Refresh the snapshot, its manifest, and its
ModelSpec together after updating the original design. The source-parity tests
check these bindings. Older experimental checkpoints use their archived source
and ModelSpec.
