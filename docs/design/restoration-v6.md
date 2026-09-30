# V6 target broadcasting and reproducible experiment history

The V6 implementation is now available as `tabu_lab.models.restoration_v6`.
Its source is the two-mode implementation used by
`v6-broadcast-oldloss-openml12-20260927` and
`v6-v55-dual718-30m-20260928`. The original files, including the earlier
joint-only variant, remain byte-preserved in the
[September 30 archive](../../experiments/history/v6-20260930/README.md).
Formatting changes in the maintained module do not rewrite historical source.
V7 and the nonlinear/mixing experimental candidates are not promoted into this API.

## Model contract

```python
from tabu_lab.models.restoration_v55 import V55Config
from tabu_lab.models.restoration_v6 import V6Model, score_training_episode

model = V6Model(V55Config(), supervision="target_only")
# inputs carries visible values and masks; truth is exclusively scorer-owned.
# score = score_training_episode(model, inputs, truth)
# score.loss.backward()
```

Each episode declares exactly one target column via its Query mask. Every
support row must have a visible target. Before the backbone, each active Cell
receives its row's target carrier. The Query target carrier is the shared seed,
not its hidden truth. Null cells remain zero, and Unit/Feature tokens are unchanged.
V6 retains all V5.5 parameter shapes and accepts the matching V5.5 state dict.
This is parameter compatibility, **not strict experiment resume**: forward mode,
supervision, data mixture, optimizer history and episode order are separate identities.

| Mode | LL response | Scored cells | Prediction |
|---|---|---|---|
| `target_only` (default) | Original per-column answer encoding | Query target cells only, existing V5.5 Query scorer | Original target codec |
| `joint_all` | $e_a(x_{ra})+e_t(y_r)$ | All observed cells in Query rows, equal column weight within each row, then equal row weight | Target encoding divided by two before target-codec decoding |

`joint_all` uses target-type scaling $\chi_t/128$, with $\chi_t=128$ for numeric
targets and $\chi_t=1$ for nominal/ordinal targets. It rejects overrides of its
request/loss contract. `target_only` requires Query-only state weights.
No hidden label is passed to `forward` or `forward_prepared`.

The model config still uses V5.5 codec/shape identities. Callers must record
`supervision` separately; a bare tensor state dict cannot distinguish modes.
The maintained API deliberately does not route `curriculum-v55` to V6 or silently
change existing CLI defaults. Historical V6 training has its own archived runners.

## Reproducing a historical experiment

Start from the [experiment index](../../experiments/history/v6-20260930/INDEX.md),
choose one host and one stage, and restore that host's source/configuration tree.
Use the archived `source/src` for an exact historical code replay; do not replace
it with current `src` merely because the tensor shapes match.

Keep the original checkpoint SHA, manifest/data SHA, fixed Query bank, RNG,
optimizer, per-table episode offsets, mixture cursor, branch RNG and consumed
successful-update budget together. The single-table repaired run also needs its
recovery-budget receipt. A changed configuration is a new run with declared
initialization; it does not overwrite or rename the historical result.

Historical execution precision is CUDA/FP64 on DGX2 and MPS/FP32 on the Macs,
using each receipt's pinned runtime (`wehub-python --profile train-20260920`
where recorded). CPU unit tests below validate contracts; they are not a new
GPU/MPS qualification run, a repeated full training experiment, or a guarantee
of bitwise MPS replay.

```bash
PYTHONPATH=src python -m pytest -q tests/unit/restoration_v6
python experiments/history/v6-20260930/archive.py verify
```

## Interpreting results

The archive retains defaults **and rejected checkpoint candidates**. Do not select
a different per-table maximum and call it a shared checkpoint. Training-row Query
fit, old-corpus retention, OpenML12 held-out test, and unseen-table transfer remain
different evaluations. Historical summaries contain dated intermediate states;
inspect the matching terminal/evaluation receipt and checkpoint hash before citing
an endpoint. No new performance claim is issued by this code consolidation.
