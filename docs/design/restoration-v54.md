# V5.4 implementation: unified composition and supervised episodes

This version adds an explicit V5.4 entry point on the shared restoration engine.
The prior code baseline is `b73635f`; V5.3 defaults, codec IDs and realizations
remain available under their original identities. Selecting V5.4 does not rename
or strictly resume a historical experiment.

The implementation reference is the owner-maintained
`TabU/latex/model-factory/table-restoration/TabU_V5p4_Unified_Composition/TabU_V5p4_Unified_Composition.tex`.
An exact [TeX snapshot](TabU_V5p4_Unified_Composition.tex) is committed alongside
this implementation, including the supervised-row/random-cell priorities and
the five named sizes. Its SHA-256 is
`e62ebca4756311fb31e2fbe9ec562f35fe77ea2c8e0acc3349a43ca6d4eb18da`.
Edit the owner source first; this snapshot is not another design authority.

## Model and value coordinates

```python
from tabu_lab.models.restoration_v54 import V54Config, V54Model

model = V54Model()  # Small, composition C, 3 Unit layers, K=1
nano = V54Model(V54Config.from_size("nano"))
unit_ablation = V54Config.from_dict({"size": "small", "unit_layers": 0})
gaussian = V54Config(codec_version="unit_gaussian_composition_v1")
```

All input and answer codes use $e_a(x)=q_a+v_a(x)$:

| Type | Value component | Decoder |
| --- | --- | --- |
| Numeric | $z_a(x)b_a$, visible-only z-score | Project onto $b_a$, then inverse z-score |
| Nominal | $b_{a,c}$, one distinct vector per visible class | Nearest $q_a+b_{a,c}$ |
| Ordinal | $r_a(c)b_a$, full declared normalized rank | Project onto $b_a$, then nearest declared rank |

`constant_weight_composition_v1` (codec ID 4) draws every base vector from raw
128-dimensional 4-hot vectors. `unit_gaussian_composition_v1` (ID 5) uses unit
Gaussian directions with the same outer composition. Neither requires
orthogonality. Episode-local random state is reproducible and not trainable.
The nominal origin is shared within a column, including across its categories.

Two 4-hot vectors can overlap: their sum is not an independent 8-hot vector and
need not have constant norm. Nominal decoding therefore uses
$\arg\max_c(\widehat e-q_a)^\top b_{a,c}$, equivalent to nearest-code distance.
Invisible nominal classes still produce `no-answer-code`; hidden truth never
creates a codebook entry. Ordinal codes use the complete predeclared order.

The model reuses the two-dimensional OTransformer, learned bias-free $W_{enc}$
initialized at $Q/8$, presence threshold 1, and column-shared LL with local free
intercepts. Unit refinement affects reference geometry and does not write back
Cell/Feature carriers. Zero Unit layers is an exact bypass. Query weight remains
1 and retained reconstruction weight 0; visible evidence still participates in
the backbone and local regression.

| Size | Backbone layers | Width | Heads | FFN | Unit layers | Slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nano | 2 | 128 | 4 | 256 | 0 | 256 |
| small (current default) | 3 | 128 | 8 | 256 | 3 | 256 |
| medium | 6 | 192 | 8 | 384 | 3 | 256 |
| standard | 12 | 256 | 8 | 512 | 6 | 256 |
| large | 24 | 384 | 12 | 768 | 6 | 256 |

`V54Config.from_dict` merges explicit backbone fields into the selected size and
`as_dict` records the resolved configuration. A Small with `unit_layers=0` is
still a Small ablation, not Nano. Nano is an equally important competing model
that must enter basic fitting comparisons. A default is a reproducible starting
point, not a performance ranking. `subtokens=1` is implemented; `K>1` is rejected
because its compilation and matching rules are not yet fully specified.

## Episodes and the versioned curriculum

The `tabu.curriculum.v54.v1` schema and `curriculum-v54` command select V5.4.
They share the existing curriculum runner while preserving separate plan,
checkpoint and receipt identities. The V5.3 command rejects V5.4 manifests,
and vice versa.

**Supervised-row is the current default for both synthetic and real tables.**
Only the target labels of selected training rows are hidden; observed features
remain visible. **Random-cell is the first alternative**, and the intended
future default once training stability and experimental experience justify it.
There is no automatic switch. Masking selects Query addresses; Query-only loss
selects which reconstruction errors have nonzero training weight.

The existing recipe structure remains explicit:

```yaml
schema: tabu.curriculum.v54.v1
model:
  size: small
# Within a stage; fraction is an experiment parameter, never silently inferred:
recipe:
  synthetic: {fraction: 0.25}  # omitted kind resolves to supervised_row
  real: {fraction: 0.25}       # the same default, irrespective of data source
```

An explicit `{kind: random_cell, fraction: 0.25}` selects the competing sampler.
The numeric column guard applies to random-cell; supervised-row does not filter
Query rows by target magnitude. Existing post-mask support checks, scorer-only
TruthSidecar, fixed probes and split boundaries remain in force. Query budgets,
optimizer settings, seeds and stage budgets stay explicit in the manifest.

The current data adapter supports one `target_column` per table. The document's
explicit multi-target supervised extension is not implemented by this adapter;
unsupported fields are rejected. The reference runner supports single-process
FP64 on CPU/CUDA and explicit MPS FP32 execution with CPU fallback disabled.
The MPS shared-LL readout keeps Cholesky and triangular solve on `mps:0`; it
does not copy sufficient statistics to CPU. V5.4 defaults to
`center_chunk_size=128` for the device-local readout, with an explicit smaller
override available for memory-constrained runs. MPS uses its own recorded
runtime and bounded numerical comparison; it is not a bitwise-equivalent
continuation of a CUDA FP64 trajectory.

## Local checks and preparation

```bash
uv run python examples/restoration_v54_smoke.py --size small
uv run python examples/restoration_v54_smoke.py --size nano
uv run python examples/curriculum_v54_fixture.py --output-root /tmp/v54-fixture
uv run tabu-lab curriculum-v54 plan --manifest /tmp/v54-fixture/manifest.json
uv run tabu-lab curriculum-v54 preflight --manifest /tmp/v54-fixture/manifest.json \
  --output-root /tmp/v54-preflight --device cpu
```

The fixture uses one deterministic synthetic table and a two-update budget;
preflight checks finite steps and same-backend checkpoint continuation. Neither
is evidence of fitting ability or generalization. Use a fresh output directory
for every run. To prepare the first alternative, pass `--episode-kind random_cell`
to the fixture generator; to compare Nano, pass `--size nano`.

## Training readout execution

The V5.4 curriculum validates and encodes the complete observation episode
before selecting the responses needed by the loss. CUDA/CPU training executes
only targets with nonzero state weight. MPS training skips wholly inactive
columns but retains the original requested rows inside each active column.
This measured implementation is used for the first Nano runs. Changing the FP32
readout GEMM row shapes exceeded the initial three-update optimizer comparison
tolerance; this is a numerical diagnostic, not evidence that row pruning is
unusable. The full table, source masks,
visible support answers and **all N shared-LL fit centers** remain unchanged.
Missing answer codes and invalid zero-weight columns still fail admission.

Fixed evaluation keeps the full readout, including retained reconstruction.
Skipped predictions are not produced or claimed to have passed output-finite
checks during training. A pruned prepared snapshot rejects a loss configuration
that enables an omitted state; reprepare the full episode for that change.
`state_weights=None` retains the original full-observation mean. V5.3 curriculum
execution keeps its historical full readout. Per-step receipts record
`readout_scope` and `readout_targets`.

The data adapter records the CPU-sampled mask audit before transfer to the
training device, avoiding three immediate device-to-host readbacks. Every
training index still resamples its original mask/code/window stream; this is
not cross-step prepared-episode caching. Precision, finite guards, stable
centered covariance and device-local solves are unchanged.

For the current exploratory fitting stage, the owner prioritizes usable training
and measured speed over matching individual gradients or optimizer updates.
Small numerical differences do not veto an optimization. Assess candidates by
runtime/finite-value checks, measured step time and fixed-Query fitting curves;
keep data isolation, episode meaning and run provenance intact. Parameter-wise
comparisons are optional diagnostics. They must not delay fitting merely because
a tolerance was exceeded. More aggressive backend-specific execution remains an
eligible candidate; record it as a separate run when it changes an active run's
implementation.

The bounded [Nano benchmark](../../experiments/local/v54-nano-speed-20260921/README.md)
compares the same Nano, table, seeds and optimizer on each backend. These speed
checks are separate from the
[single-table fit panel](../../experiments/local/v54-nano-fit-20260921/README.md).

Validation targets cover composition geometry and decoding, old codec stability,
named-size configuration, forward/backward and prepared-state contracts,
sampler/schema isolation, and exact local resume. Compare fitting variants on
common original-value metrics: changed category-code distances make training
loss alone unsuitable for ranking V5.3 and V5.4.

## Old120 loss-prioritized extra training (2026-09-22)

The owner chose a full balanced pass followed by bounded extra training. After
120 normal updates (one per table), rank their original, pre-update Query losses
once. The latest revision is additive P99 × 3, P95 × 2, P80 × 1. With 120 tables,
ceil the selected proportions to top2, top6 and top24. Execute three full top2
passes, two full top6 passes, then one top24 pass. The highest two tables thus
receive six extras each, the next four receive three, and the next eighteen
receive one. Each cycle has 120 normal + 42 extra = 162 actual optimizer updates.
Resolve loss ties by table ID to keep the allocation exact. Extra losses do not change
the current round's ranking. Extra episodes retain the supervised Query recipe
and composition encoding, using a separate episode RNG namespace and per-table
extra index; the normal episode stream remains directly addressable.

This is the selected experimental strategy, not an empirical claim that uniform
training is inferior. Existing manifests remain uniform unless they explicitly
declare a `loss_replay` policy. The first deployed policy,
`normal120_top5_top20_v1`, remains supported with its original top6 once + top24
once (30 extras). The latest policy is `normal120_p99x3_p95x2_p80x1_v2`. Both
support a single bounded 120-table stage. V2 records `normal_max_updates` (N),
the complete-cycle `start_normal_cursor` (S), and inherited
`start_extra_updates` (E). Its actual `max_updates` is N + E + (N−S)/120 × 42.
Finish the parent's pending normal/replay cycle before migration; do not
retroactively replay earlier cycles or erase inherited extra updates. V2
has been used for gongqian-mini Small-H4, dgx2 standard Small and dustinstudio Nano
as explicit migrations. Each retains its existing normal-update budget and
ancestor history. A new weights-only experiment may also start this policy with
S=E=0; that is a separate zero-exposure task, not a migration of parent updates.
Migration receipts identify the first new-policy cycle and its added budget;
historical uniform and V1 outputs retain their original semantics.

The dated [old120 fit audit](../reports/v54-old120-fit-handoff-2026-09-23.md)
found that all extra updates in the three running migrations went to numeric
target tables, while some nominal and ordinal tables remained below their own
visible-support majority reference. This is an observed limitation of ranking
raw losses across target types, not a change to the frozen V5.4 policy or proof
that an alternative policy is better. Type-stratified allocation and
reference-normalized difficulty are next-design comparison candidates.

`train_cycle` mean/median/P05/P95, plus P99 for V2, use only the normal 120 losses. Extra losses,
normal and total update counts, and per-table extra exposure are recorded
separately. Checkpoints retain the normal cursor, partial normal losses, frozen
extra queue and its position, and per-table extra episode indices. A strategy
change preserves model/optimizer/RNG/history through an explicit migration into
a new identity and output; it is not an unchanged-recipe resume or a fresh
weights-only start. Compare strategies using fixed training-row Query results
at matched actual compute, never only smoother cycle curves.
