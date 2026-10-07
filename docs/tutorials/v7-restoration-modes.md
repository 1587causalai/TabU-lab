# V7 / V7.3 single, joint and mixed restoration

The local V7 implementation now uses one `V7Model` and one optimizer for all three
training modes. The joint numerical path was promoted from the audited 2026-10-03
single/joint experiment. Mixed sampling is available for experiments; its quality
advantage has **not** been established by that single-versus-joint comparison.

## Configuration and command

Choose the [V7 YAML](../../examples/v7-restoration/mixed.yaml) or
[V7.3 scratch YAML](../../examples/v7-restoration/v73.yaml), both using the same
small synthetic table and eight-update smoke budget. Paths inside YAML resolve
relative to that file. The YAML field `model.model_version` selects `v7` or `v7.3`.

```bash
# Inspect the resolved run without training or creating outputs.
python -m tabu_lab.cli restoration v7-fit --config examples/v7-restoration/mixed.yaml

# Execute the explicitly bounded eight-update fixture.
python -m tabu_lab.cli restoration v7-fit --config examples/v7-restoration/mixed.yaml --execute
```

On configured training hosts use their existing launcher instead of generic
Python: `~/.local/bin/wehub-python --profile train-20260920 -m tabu_lab.cli ...`,
from the deployed project/environment that contains this package. Merging this
implementation does not update host checkouts or start remote experiments.

Only `masking.mode` needs to change to switch modes:

| Mode | Each training episode |
| --- | --- |
| `single` | Uniformly select one eligible column and hide its Query rows |
| `joint` | Select `joint_columns` eligible columns and hide the same Query rows in all of them |
| `mixed` | With `joint_probability`, draw a joint episode; otherwise draw a single episode |

`joint_probability: 0.5` means 50% probability **per episode**. With three-column
joint episodes it does not mean half the Query cells. Finite runs need not realize
an exact 50/50 count. The actual counts and cell exposure are saved.

`columns: [0, 1, target]` defines eligible columns. Use column names or indices;
`target` resolves to each table's target (last column unless explicitly set).
For target-only supervised adaptation use `mode: single, columns: [target]`.
Omit `columns` to use all columns. A table entry may override `columns` for a
heterogeneous corpus. `joint_columns: null` uses all eligible columns; insufficient
eligible columns, duplicates, unknown fields and invalid row budgets fail explicitly.

Tables follow `tabu.tar.typed-fit-table.1`, or the existing compatible OpenML table
format, with `features`, `values`, and disjoint `splits.train/test` row addresses.
Tables receive equal visits in shuffled cycles. Episode draws are deterministic
from seed and step and use training rows only. The window is capped by the number
of training rows; at least two supports must remain. A numeric queried column needs
two distinct visible support values. A hidden nominal category without a visible
code fails training admission/scoring rather than being dropped or silently resampled.

## Model and loss

V7 and V7.3 name implementation protocols, not training-quality rankings. Mask
selection (`single`/`joint`/`mixed`) is independent of the version. Use
`V7Config.for_version("v7")` / `V7Config.legacy()` or
`V7Config.for_version("v7.3")` / `V7Config.v73()` for new Python models.
`V7Config.cyclic()` is an alias for `v73()`; each factory accepts explicit
experimental overrides such as `V7Config.v73(rounds=8)`.

| Default choice | V7 factory | V7.3 factory |
| --- | --- | --- |
| Query initialization / source / rounds | seed / fixed true sources / K1 | donor / Query as source / shared K4 |
| Coupling bias | enabled | disabled; forward and inverse preserve zero |
| Numeric preprocessing | `legacy` | `standard_asinh_v1` |
| Optional Unit source policy | `legacy_cell_sources` | `observed` |
| Additional Unit layers | 0 unless explicitly set or inherited | 0 unless explicitly set |

V7.3 numeric coordinates are `asinh((x - mean) / scale)`, with standard-deviation
floor 0.01 and `sinh` inverse after projection. The transform is continuous,
invertible, approximately linear near zero and logarithmic in the tails.
Controlled same-input Polish-table evidence isolated extreme numeric input
magnitude as a source of cyclic amplification; this motivates the default,
without establishing universal stability or unseen-table performance.
`V7Config.v73(numeric_preprocessing="standard_softlog_v1")` retains the explicit
historical/candidate protocol: a strict visible standardized magnitude above
1e6 selects signed softlog for the whole column. Complete checkpoint configs
keep their declared numeric protocol; changing it is weights-only initialization
with a fresh optimizer/RNG, never strict resume. Different numeric target code
losses cannot be compared directly across these protocols. Standalone codec and
episode defaults remain legacy.

Recorded V7 checkpoints take precedence over these starting defaults: donor,
Query-source, K4/K8 and other explicit historical choices are preserved. A config
without `model_version` is V7. Its absent bias, numeric-preprocessing and Unit-policy
fields receive legacy meanings, while its explicit values remain unchanged.
An explicitly declared V7.3 config receives the V7.3 defaults for absent fields.
The unqualified `V7Config()` constructor is frozen to the pre-update `af7e5f51`
V7 defaults: biased coupling, legacy preprocessing and legacy Unit source policy.
It agrees with `legacy()` and an empty unversioned deserialization. Old positional
arguments and the defaults of standalone codec/episode preparation are preserved.
Direct `V7Config(model_version="v7.3")` agrees with `v73()`; explicit overrides
including `False` and K1 are not mistaken for omitted values. This default is one
historical starting point, not a reinterpretation of every prior V7 experiment.

The V7.3 `observed` Unit policy lets only rows with a real visible Cell supply Unit
sources; rows containing only Query/Null cells still receive updates. Cell
Query-as-source remains independently controlled by `query_source`. V7 preserves
the historical Unit initialization sequence; V7.3 initializes the installed Unit
stack consistently with the V7.3 backbone when that explicit variant is selected.
V7.3 also opts into the stricter attention-content numerical checks; the shared
V5/V6/V7 operator retains its historical default computation and failure boundary.
Version and Unit policy are saved in ModelSpec; changing either requires a new
weights-only initialization, not strict continuation.

Each joint round runs the backbone once, fits each queried column's LL using only
its own true visible supports, then writes every Query recovery simultaneously.
Recovered Query cells can be sources in the next round, but never become LL
supports. Observed codes are clamped; the complete cyclic path retains gradients.
The loss averages queried cells within each Query row, then averages Query rows,
then combines rounds geometrically. Single-column loss preserves its former
reduction order. The V7 fixture's numeric/discrete coefficients 32 / 0.25 match the
scale12 comparison, not the general constructor's 128 / 1 defaults.

The public Python API also supports different Query rows in different columns:

```python
from tabu_lab.models.restoration_v7 import (
    table_mask_task, train_step, evaluate_joint_task,
)

task = table_mask_task(
    table, rows=[0, 1, 2, 3, 4, 5],
    query_by_column={0: [4], 2: [4, 5]}, code_seed=7, donor_seed=8,
)
train_step(model, optimizer, [task], grad_clip_norm=1.0)
report = evaluate_joint_task(model, task)  # report.columns[column].predictions for all rounds
```

The old `table_task`, `prepare_episode`, and single-column `evaluate_task` result
remain available. `train_step` and `evaluate_task` dispatch joint masks automatically;
`evaluate_joint_task` always returns a column-indexed report, even for one column.

## Optional row-contextual dual-stream encoder

The current native source also provides the explicit experimental
`value_encoder="row_dual_stream"` branch. Omission retains `phi_lift` and historical
checkpoint meanings. Encoder choice is separate from single/joint/mixed masking.
For a new model, select it explicitly:

```python
from tabu_lab.models.restoration_v7 import DualStreamConfig, V7Config, V7Model

config = V7Config.v73(
    value_encoder="row_dual_stream",
    dual_stream=DualStreamConfig(),
)
model = V7Model(config)
```

This is a construction example, not the complete configuration of a trained model.
The branch requires code width 64, backbone width 128 and `query_source=True`.
By default, its four independent encoder blocks each apply one row-local attention increment
and one token-wise FFN increment. Their 64-wide streams concatenate to 128, with
no entry lift. Sharing the complete recovery module across rounds is a separate
setting. Static visible-or-Query masks are identical in forward and inverse;
structural Null cells have zero updates.

Each LL now predicts **128-wide input-side contextual responses**. A visible
support's response can depend on a Query estimate elsewhere in the same row;
keep this gradient and recompute responses each round. All Query predictions are
assembled simultaneously with the other cells' current encoded values, then the
whole row is inverted. The two inverse streams are averaged and only Query codes
are written back. This assembly and `mean` readback are implemented candidates,
not established optimal choices. The loss still scores 64-wide code states.

For an old `phi_lift` donor, use the explicit
`row_dual_stream_from_checkpoint(path, device=..., dtype=..., seed=...,
expected_sha256=...)` Python migration API. It inherits the donor's backbone and
LL configuration, copies compatible backbone and Unit/Feature seed tensors,
and freshly initializes the encoder. The old phi, lift and Query seeds are not
transferred. Start a fresh optimizer/RNG/sampler; this is not strict resume or a
guarantee of preserving the donor's predictions. Changing only `value_encoder`
under the YAML runner's strict `init_checkpoint` path is rejected.

New dual-stream checkpoints require this native source or a compatible successor.
Previously frozen standalone distributions are not upgraded by a documentation
change. CPU-FP64, actual MPS-FP32 and independent CUDA-FP64 functional checks have
passed; bounded training still shows numerical sensitivity and does not establish
an architecture advantage. Backbone gradient checkpointing does not yet checkpoint
the new encoder. The training code loss does not pass through the final numeric
`sinh` decoder: decoded outliers and training-gradient spikes are distinct issues.

## Optional same-column Query/visible reconstruction

The implementation now supports the optional contract in `end-to-end-design.tex`,
`v7:optional:balanced-reconstruction-loss`. Default `loss_mode: query_only` preserves
existing behavior. Select the alternate mode explicitly in the `model` section;
see [the bounded four-update fixture](../../examples/v7-restoration/balanced.yaml):

```yaml
model:
  loss_mode: balanced_reconstruction
  auxiliary_per_query: null
  auxiliary_min_group_weight: 0.05
  balanced_loss_scale: 1.0
```

With `auxiliary_per_query: null`, all true visible cells in each queried column
are auxiliary targets. A positive number `r` instead samples at most
`ceil(r * n_query)` visible cells per column, uniformly without replacement.
Sampling uses a private generator seeded from the task's recorded donor seed;
it neither consumes the donor generator nor changes the Query/visible masks.
Addresses are fixed before the unroll and reused in every round.

Each round reuses its existing LL slope to evaluate the selected visible rows,
then applies the inverse value map to obtain separate auxiliary code predictions.
For `row_dual_stream`, auxiliary predictions use a separate whole-row assembly
from the same encoded snapshot: replace auxiliary addresses, retain Query
encodings, invert and average the streams, then extract only auxiliary addresses.
It does not score the clamped facts as predictions and does not write auxiliary
predictions back into the table. The slope, true support responses, kernel,
backbone and inverse map remain differentiable. This is **in-sample visible
reconstruction**, not leave-one-out or held-out prediction; auxiliary cells remain
visible and participate in the fitted LL system. It has not been shown to improve
training stability or predictive quality.

For each column, `w = n_query / (n_query + n_aux)`. The round loss is the sum over
queried columns of `(1-w) * sum(query_cell_losses) + w * sum(aux_cell_losses)`,
multiplied by the explicit `balanced_loss_scale`. Cell losses use the configured
type coefficient divided by code dimension. There is **no additional average**
over rows, cells or columns in this mode; the document intentionally retains
the common group mass. Round weights and the final mean over episodes remain
unchanged. Consequently its scale differs from Query-only row-mean loss; matching
the learning rate alone is not a scale-controlled experiment.

An empty group or `min(w, 1-w) < auxiliary_min_group_weight` rejects the episode
before forward. There is no silent weight clamp or dropped Query. The public
`train_step` attaches the plan automatically for single, joint and mixed tasks.
Its `StepRecord.auxiliary_plans`, also written by the YAML runner to each step
receipt, records the addresses, counts, weights, seed, sampling rule and global
scale. Checkpoints serialize all loss settings. Changing the objective requires
weights-only initialization, not strict resume. Initial loss and evaluation
remain explicitly Query-only; no auxiliary initial prediction is invented.

For custom loops (including estimator-built episodes), attach a plan explicitly:

```python
from tabu_lab.models.restoration_v7 import prepare_auxiliary_reconstruction

episode = prepare_auxiliary_reconstruction(episode, model.config, seed=episode_seed)
output = model(episode, decode=False)
# score_rounds(...) for a single column, or score_joint(...) for multiple columns.
```

The balanced scorer rejects a missing plan or missing auxiliary predictions;
changing only a config flag in an old custom training loop cannot silently run
Query-only loss. Frozen historical experiment sources remain unchanged.

## Execution, checkpoints and evaluation

Choose `device: mps, dtype: float32` for Apple training; use the configured
`cuda/float64` on the existing CUDA training hosts. No device fallback is performed.
`gradient_checkpointing: true` recomputes the backbone during backward to save
memory. The small CPU fixture is an implementation check, not a trained candidate.

To initialize from an existing checkpoint, add `init_checkpoint: /path/parent.pt`.
Within its version, the runner inherits the checkpoint ModelSpec, then applies
explicit `model` overrides and loads weights strictly by name/shape. For a real parent, remove the
fixture's tiny backbone and coupling dimensions; retain only intended overrides,
for example:

```yaml
init_checkpoint: /path/parent.pt
model:
  query_init: donor
  query_source: true
  rounds: 4
  share_rounds: true
  gradient_checkpointing: true
```

This is weights-only initialization with fresh optimizer/RNG/sampling. Historical
experimental checkpoints may omit Query-as-source metadata, so explicitly declare
it as above. Their optimizer format is not assumed to be a native strict resume.

To declare a new version, set `model.model_version: v7.3` (or `v7`). A cross-version
initialization keeps the parent's tensor dimensions, Unit depth and round-sharing
mode. An unshared model also retains its number of rounds, since each round owns
parameter tensors; shared recurrence can use the selected version's default round
count. Query, bias, numeric-preprocessing and Unit source policy receive the selected
version's defaults, then explicit overrides take precedence. The receipt records both
versions. Weight loading still requires exact names and shapes: a biased V7 parent
does not silently lose its bias tensors when entering bias-free V7.3. That conversion
requires a separately explicit migration; the V7.3 scratch example needs no parent.
The dry-run manifest lists changed configuration fields and the planned inherited
state keys. This supported path inherits every state key, rebuilds no model tensor,
and starts a fresh optimizer/RNG/sampler. It rejects a partial or incompatible load;
the dry-run plan itself is not a claim that weights have already been loaded.

Native continuation uses the same YAML, a larger total `steps`, and a **new** output
directory:

```bash
python -m tabu_lab.cli restoration v7-fit --config run.yaml --execute \
  --resume /path/previous/final.pt --output-dir /path/continuation
```

Strict resume requires the same model version, model, masking, optimizer, data/split, seed,
evaluation definition, source digest, device/dtype and Torch version. It restores
weights, AdamW, CPU/device RNG and the next sampler step. Changing single/joint/mixed
mode is a new objective: use weights-only initialization for that change.
New checkpoints repeat `model_version` at the top level for inspection and verify
it against ModelSpec on load; old checkpoints without that top-level field remain
loadable through the historical compatibility rules.

The source digest intentionally still covers all Python files under `tabu_lab`.
Adding V7.3 code can therefore reject strict continuation of an old source snapshot
even if its V7 prediction graph is unchanged. Keep each old experiment bound to its
original source, complete configuration and runtime; do not relax the digest to
force it through. Current same-source continuation tests do not prove continuation
of a historical experiment in its original training environment.

Configuration validation rejects non-boolean `coupling_scale` values (including
the string `"false"`) and invalid optimizer rates, epsilon, weight decay or beta
coefficients before constructing a model or creating run outputs.

Outputs include resolved configuration, per-update row/column addresses and losses,
actual single/joint counts and Query-cell exposure, `final.pt`, `evaluation.json`,
and a terminal receipt. Existing output directories are rejected. Final evaluation
uses a deterministic same-table heldout bank for target-only and joint missingness,
reports every recovered column and round, and does no checkpoint selection. An
empty test split is recorded as `no-heldout-rows`; it is not scored on training rows.
This runner is step-bounded; the frozen experiment's time/exposure checkpoint grid,
watchdog, and two-stream schedule remain in that experiment's archive.

For nominal and ordinal columns, a trajectory also exposes `candidates`,
`probabilities`, `log_probabilities` and `probability_temperature` from the final
round. `evaluate_task` and `evaluate_joint_task` accept `probability_temperature=1.0`.
Metrics record `query_count`, `answer_code_count`, `answer_code_coverage` and
`log_loss`. A missing answer code keeps the full Query denominator and makes
whole-column log loss unavailable; it is not silently dropped from scoring.
Probabilities are fixed-code distance readouts, with no fitted calibration.

The optional native inference adapter is available from
`tabu_lab.evaluation.tabarena.estimator`; install the `tabarena-inference` extra
to include its pandas/sklearn dependencies. The standalone TabU V7 package is
generated from the same model and adapter source, with source hashes recorded.
Its default loader follows the recorded ModelSpec, including `query_source=False`.
An old `donor-source-cyclic-v1` checkpoint missing that field requires the explicit
`legacy_overlay="donor-source-cyclic-v1"` option and a recognized frozen-source
identity. An explicit field is never overridden. The recorded inference protocol
distinguishes stored settings from the selected historical external wrapper.

Verification has three separate meanings: historical predictions compared round
by round with frozen source; historical next-update weight/optimizer/RNG agreement
in the original runtime; and V7.3 creation/save/reload/package agreement. Passing one
does not establish the others. See the review's evidence table for the current scope.

## Experimental row dual-stream value encoder

The default remains `value_encoder="phi_lift"`. Select `row_dual_stream` explicitly
with the [eight-update CPU fixture](../../examples/v7-restoration/row-dual-stream.yaml)
or the Python API:

```python
from tabu_lab.models.restoration_v7 import DualStreamConfig, V7Config, V7Model

config = V7Config.v73(
    value_encoder="row_dual_stream",
    dual_stream=DualStreamConfig(),
)
model = V7Model(config)
```

Each row's 64-dimensional codes initialize two streams. Four independent
OAttention/OFFN additive coupling blocks produce a 128-dimensional contextual
value embedding, which enters the existing token dynamics directly. Attention
inside this encoder operates only among Cells in the same row. The analytical
LL readout uses the current round's input-side Support embeddings as responses.
All Query predictions are assembled simultaneously, the whole row is inverted,
and the two streams are averaged. Only Query codes are written back; observed
codes remain clamped. Joint missingness and optional auxiliary reconstruction
use the same synchronous contract.

`mean` readback and `query_ll_whole_row` assembly are named research choices.
The coupling is invertible on the full 128-dimensional state, but duplicating
64-dimensional input codes does not cover that full space. Invertibility and
finite-difference checks do not guarantee favorable conditioning, stable
multi-round training, or better predictions. Exploratory runs have shown finite
gradient spikes. Keep gradient clipping and per-run diagnostics explicit.

This encoder requires `code_dim=64`, backbone width 128, `query_source=True`, and
the coupling value-map setting. Its constructor validates those requirements.
The old `phi_lift` configuration serialization and weight names stay unchanged.

Switching encoders is a weights-only migration through
`row_dual_stream_from_checkpoint(path, expected_sha256=...)`. That API validates
the donor graph, inherits compatible token-dynamics weights and Unit/Feature
seeds, creates a new row encoder, and returns a `transfer_receipt`. It does not
restore the donor optimizer, RNG or sampler. The generic `init_checkpoint` path
rejects an encoder switch, and strict continuation still requires an identical
complete model/source/runtime identity. New dual-stream checkpoints require this
implementation or a later compatible version; a separately frozen standalone
inference package is not upgraded by merging native code.

The [integration record](../reviews/v73-integration-20261008/README.md) links the
reviewed source identities and current verification.

Validation of the version split and design alignment is recorded in
[the V7 / V7.3 review](../reviews/v7-v73-design-alignment-20261004.md).
The earlier single/joint/mixed integration retains its
[2026-10-03 check](../reviews/v7-restoration-modes-20261003.md).
