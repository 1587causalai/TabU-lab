# Five-step table-restoration reference

This is an independent table-restoration implementation, not a reinterpretation
of TAR's additive heads, checkpoints, or published fit results. Status is
`local_unissued`: implementation verification is not F0 fit, benchmark readiness,
an accepted claim, or evidence of unseen-table generalization.

## Boundaries and extension points

| Step | Stable boundary | Implemented alternatives |
| --- | --- | --- |
| 1. Input | `RestorationInput`, `RestorationRequest`, scorer-only `TruthSidecar` | Main visible/Query episodes; explicit retained/Query/Null/corrupted damage episodes; independently requested prediction addresses |
| 2. Encoding | Fixed visible `ColumnFacts` versus learned `ValueEncoder` | Nominal identity128, rotary32, mlp32, mlp256; learned numeric Fourier map; default ordinal code plus declared rank |
| 3. Backbone | Augmented carriers plus fixed visible/query role masks | Direct axial OMAB; fixed-slot column inducing collect/read followed by direct row OMAB |
| 4. Prediction | Shared Unit kernel; same-column visible supports; answer coordinates | Model-wide LL (default) or NW; numeric inverse scaling and discrete nearest-visible-code decoding |
| 5. Training | Scorer encodes truth, computes per-target MSE, then reduces | Canonical numeric/discrete means; explicit G/Q/Z/B-weighted means; equal episode-weight batch loss |

These are small module boundaries, not a plugin registry. Replacing an encoder or
backbone requires preserving its input/output semantics and re-running invariants.
The built-in configuration round-trip describes only the built-in modules.

## Quick start

```bash
uv sync --frozen --extra dev
uv run tabu-lab restoration inspect
uv run tabu-lab restoration verify --components-only
uv run tabu-lab restoration verify
uv run tabu-lab restoration verify --device cuda:0 --output device-check.json
uv run pytest -q tests/unit/restoration
```

`verify` emits JSON with check outcomes and a digest of the actual restoration
Python sources. It returns nonzero if a probe fails. The component suite retains
normal-equation, finite-difference and answer-decoding checks. Model probes cover
16 backbone/nominal-map/readout combinations and serialized model/AdamW state
continuation over two updates. They use a reduced CPU FP64 model and in-memory
checkpoint, not the default-size model or a training campaign. Tests additionally
cover row/column permutation, receiver-only insertion, request independence,
projected zero source mass, damage statistics, support failures, and aggregation.
`--output path.json` additionally records the check JSON and refuses to overwrite
an existing file. A recorded local check is not formal receipt issuance.

The CUDA entry compares a bounded mixed-type FP64 episode against CPU, including
parameter gradients, at predeclared `rtol=1e-7, atol=1e-8`. It also checks exact
two-update CUDA AdamW continuation through an in-memory checkpoint. It records
the actual device/runtime and fails explicitly if CUDA is unavailable; a passing
probe qualifies only that configuration, not full-scale training or throughput.

## Model and training API

```python
import torch
from tabu_lab.models.restoration import (
    ColumnSchema, RestorationModel, make_episode, score_episode,
)

schema = (ColumnSchema("x", "numeric"),)
values = (torch.tensor([0., 1., 2., 3.]),)
observed = torch.ones(4, 1, dtype=torch.bool)
query = torch.tensor([[False], [False], [False], [True]])
inputs, request, truth = make_episode(
    schema, values, observed, query, code_seed=42,
)
model = RestorationModel().double()
output = model(inputs, request)  # no truth argument
prediction = output.columns[0].decoded

# Scorer validates the entire episode before neural forward, then computes loss.
score = score_episode(model, inputs, request, truth)
score.loss.backward()
```

`RestorationInput` clones masks and removes nonvisible payloads on construction.
Inputs are episode-owned; do not mutate their tensors after construction. Numeric
columns are floating vectors; discrete columns are int64 domain indices. Schema
keys are stable, unique column identifiers. Ordinal indices encode declared order;
domain size/order must not be inferred from hidden truth. All model/input tensors
must share a device. CPU FP64 is the reference path. Device probes qualify bounded
CUDA configurations; they do not establish MPS, mixed-precision or large-table support.

Requests are unique int64 `[row, column]` pairs. Column predictions carry positions
in the original request, restoring arbitrary request order without padding
different answer widths. Zero supports return `no-support`; one support is valid
for inference. Training requires at least two supports in every supervised column.
Training preflight requires the complete original observed target set; arbitrary
subsets are inference-only and cannot hide missing answer codes. `make_episode`
constructs all observed targets and requires nonempty Query for
main restoration. It never silently retries masks or skips invalid episodes.

## Shared numeric coordinates and learned input features

Numeric columns establish one codec from their actual finite visible inputs,
including corrupted visible values. With Type-7 sample quantiles (linear
interpolation at zero-based sorted index $(n-1)q$), its coordinates are

$$
m = Q(1/2),\qquad s = \max\{(Q(3/4)-Q(1/4))/2,\varepsilon\},\qquad
e(x) = (x-m)/s.
$$

`NumericAnswers.median` stores the median; `scale` stores the half-IQR bounded
below by `EncoderConfig.epsilon` (default $10^{-6}$ in the column's raw units).
This is a central scale without a normal-consistency factor. Single-support,
constant and repeated-value columns use the same floor when necessary; there is
no standard-deviation fallback. Empty support retains `no-support` with undefined
center and scale.

Input coordinates, support answers, scorer-only truth encoding and final decoding
reuse this fixed codec. Hidden truth and newly added Query rows cannot alter its
statistics. Answers remain linear and reversible, without clipping. For example,
visible values $8,9,\ldots,16$ give $m=12$ and $s=2$; changing only the last value
to $16000$ leaves both unchanged. This example does not imply invariance to every
pattern of corruption. A positive unit change $x'=\gamma x+\beta$ preserves the
coordinates when the floor is also changed to $\varepsilon'=\gamma\varepsilon$.

The learned 64-frequency sin/cos map transforms these coordinates only for input
features, then feeds a shared bias-free 128-to-d projection initialized by thin QR
divided by eight; $d\ge128$. Decoding never inverts the Fourier map.

The former `eps_scale` and `sigma_min` configuration fields are removed and are
rejected rather than interpreted as `epsilon`. Old mean/std configurations and
checkpoints require an explicit migration decision before reuse.

Nominal identity codes have exactly eight ones. The deterministic per-episode
generator hashes explicit seed, stable column key and sorted visible identities;
it rejects exact duplicates and does not consume the process RNG. Each class has
one code used by both input encoding and the answer codec. No global vocabulary
or hidden-only class code is synthesized.

| Nominal map | Raw/answer width | Input features |
| --- | --- | --- |
| `identity128` (default) | 128 | Identity |
| `rotary32` | 32 | Four learned block-diagonal pair rotations, concatenated to 128; T^T T = 4I |
| `mlp32` | 32 | Affine-GELU-affine MLP to 128 |
| `mlp256` | 256 | Affine-GELU-affine MLP to 128 |

MLP hidden width defaults to 128 and is recorded, not a frozen mathematical
constant. Ordinal remains the declared **128/8 code + normalized rank** input and
raw 128-dimensional answer under every nominal-map setting. Alternative ordinal
lifts are deliberately not inferred from the nominal alternatives.

Discrete prediction is nearest visible identity code, with schema-order ties;
hard decoding is not used for training. Loss is coordinate-mean squared error,
without the historical factor one-half, classification NLL, or probability
smoothing. A truth class without a visible answer code invalidates the episode.

## Backbone and readout realization

Reference defaults are d=128, two layers, four heads, FFN width 256, 256 inducing
slots, presence tau=1, reference mass eta=1, RMS epsilon=1e-6. Layer/head/FFN counts
and positive constants are explicit implementation choices, not benchmark-tuned
defaults. Slots never shrink with table height and have independent per-layer
parameters shared across original columns.

Q/K/V read raw carriers. Attention and FFN have separate identity-initialized
bias-free presence readouts. Source presence enters both numerator and denominator;
the fixed positive zero-value reference mass is always present. Ineligible sources
are removed before projection. Presence uses scaled FP64 log-mass accumulation,
so an underflowed square does not turn a nonzero source into an empty source.
Inducing collect uses the same exact-zero criterion. FFN uses learned-scale, bias-free token-local RMS
normalization and bias-free GELU layers. Empty-source OMAB still executes the local
FFN. The Unit extension column has no inducing slots, but retains this local update.
Collect seed residuals are not evidence when all collect-projected source mass is
zero. Query, Unit and Feature tokens remain receiver-only; Null stays exactly zero.

Unit Gaussian logits are shared across target columns. All same-column visible
supports, including self-support, are selected before normalization. LL is centered
positive-ridge local linear regression with unpenalized intercept and possibly
negative equivalent coefficients. It applies to every answer type, including Null
targets. NW is a separate model-wide weighted-mean alternative. No per-type mode
routing, top-K search, cache, learned answer head, or inverse MLP is introduced.
The LL solver is a dense FP64 reference, not yet a scalable dual/chunked solver.

Within one table, backbone attention batches columns or rows and all heads into
tensor operations. Readout pads the active columns' supports, targets and answer
widths for a single batched NW/LL call, with exact support and target masks.
Numeric codecs group columns with equal support counts for batched quantiles;
learned features group by schema type and share one projection over visible cells.
Readout gathers geometry and Cell content across columns in batch. Standard numeric
truth encoding and inverse scaling are also batched, while custom codec overrides
remain available. Inducing collect reuses its computed source presence for gating.
Column-level Python loops still organize metadata and unpack ragged outputs. This
does not provide parallel execution of multiple table episodes or replace the
dense primal LL solver with a dual solver.

## Damage and loss configuration

Passing `null=...` and/or `replacements={(row, column): value}` to `make_episode`
explicitly opts into the damage extension. Masks must partition original observed
addresses consistently. Replacement values are the actual visible supports and
therefore determine input statistics, codebooks and terminal answers. The original
clean observations and G/Q/Z/B labels remain scorer-only. Retained values may
self-support; corrupted values also retain self-support. This makes no automatic
robustness claim. Sampling recipes, severity sweeps and curricula are deferred.

Default loss is the mean over numeric targets plus the mean over discrete targets,
omitting empty branches. `LossConfig(state_weights=(g,q,z,b))` instead sums weighted
within-type, within-state means. Targets with equal answer width share MSE calls;
type/state reductions are tensor operations, with one host transfer for reported
state counts and means. `batch_loss` averages episode losses equally,
serially; it is not parallel batch execution. Mask sampling and loss weighting are
independent choices. Reports separate retained, Query, Null and corrupted counts
and coordinate MSE. Do not equate visible reconstruction with hidden recovery.

Persist `model.config.as_dict()`, `model.state_dict()`, optimizer state, RNG state
and the episode source/schema/mask/code-seed identity for reproducible continuation.
Configuration is restored with `RestorationConfig.from_dict`. The verifier tests
model and optimizer continuation. `restoration fit` adds a bounded diagnostic
runner with committed preregistration, fixed training rows/masks, immutable output
directories, finite-state checks and resumable model/optimizer/RNG state. The
`restoration-small128-latest-pilot` preregistration uses fresh initialization and
at most 200 updates or 600 cumulative seconds. Its metrics use the fixed fit mask
bank; reserved rows are not scored. Execution results must be read from the
terminal record, separately from preparation and component checks. Large-table
solver optimization must demonstrate forward, gradient, optimizer and continuation
equivalence before replacing this reference. Existing TAR runners and model factory
defaults remain unchanged.

The `restoration joint-fit` route extends the same contract to the frozen old120
corpus: one shared mixed-type Small-128 model receives one fresh, class-supported
mask per table and round, while a bounded eight-entry prepared cache serves the
fixed evaluation masks. It reports retained/Query encoding MSE, numeric error in
original units, discrete accuracy, and coverage by table, source family, and
column type. The route remains a training-row fit diagnostic; the reserved 52 rows
and unseen-table generalization are outside this result.

The current runnable recipe is
[`restoration/old120-small128-tar-budget`](../../experiments/local/restoration/README.md).
The earlier `restoration-old120-joint-fit` preregistration is a historical v1
pilot bound to source `def4954`; preserve its bytes and use that exact source to
reproduce it. The current v2 runner rejects its old protocol and checkpoints.

### Live restoration observation

`restoration joint-fit --execute` can mirror its events to the W&B project
`restoration` when the optional `telemetry` dependency is installed. Observation
is disabled by default. Enable it only for an authorized experiment:

```bash
export TABU_LAB_OBSERVER=wandb
export TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE=1
export WANDB_PROJECT=restoration
export WANDB_RUN_ID=restoration-small128-budget-001
export WANDB_NAME=restoration-small128-budget-001
```

The SDK uses existing authentication, including `.netrc`; a `WANDB_API_KEY`
environment variable is not required. `WANDB_ENTITY` and `WANDB_RUN_GROUP` are
optional (`WANDB_GROUP` is also accepted). Use a unique
`WANDB_RUN_ID` for a fresh model and the same ID for a checkpoint continuation.
The observer uses `resume="allow"` and plots against the model's `update` counter.
W&B history steps remain SDK-managed so evaluation and resumed events do not
rewind the tracker history.

The mirror includes training loss, gradient norm, update duration, memory,
evaluation progress, retained/Query metrics, coverage, and terminal status.
An evaluation cut short by the deadline records `complete=false`, its completed
and expected episode counts, and `stop_reason=wall_limit` in the mirror.
Numeric hyperparameters and source/data/config digests identify the run. Raw
tables, masks, checkpoints, credentials and filesystem paths are excluded.
Per-table metric curves require `TABU_LAB_WANDB_TABLE_METRICS=1`; grouped metrics
are included by default. Git/code capture, console capture and automatic system
metrics are disabled. The backend may still record host metadata independently,
which is why hosted mode retains the explicit disclosure opt-in.

Local JSON/JSONL receipts remain the result source. A tracker failure disables
the mirror while training continues to write those receipts. When live monitoring
is required, verify W&B initialization and the run URL before starting the long
budget; an offline mirror or a dry-run plan does not establish online readiness.

## Prepared execution for repeated episodes

`prepare_episode(model, inputs, request, truth)` creates an owned snapshot for
repeated updates on one fixed episode. It prepares visible numeric statistics,
category codes, typed coordinates, input addresses, readout packing and target
ordering once. Its scorer-only section holds detached truth encodings and loss
groups. Only the visible section reaches `model.forward_prepared`; neither clean
Query truth nor damage-state labels become model inputs.

```python
from tabu_lab.models.restoration import prepare_episode, score_prepared_episode

prepared = prepare_episode(model, inputs, request, truth)
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    score = score_prepared_episode(model, prepared)
    score.loss.backward()
    optimizer.step()

# Explicit prediction/reporting when needed:
score = score_prepared_episode(model, prepared, decode=True, report=True)
```

The training replay defaults to `decode=False, report=False`: `decoded` fields and
`by_state` are then `None`. Encoding MSE, learned forward, backward, and numerical
checks still run every step. Skipping inverse scaling does not certify finite
original-unit predictions; request decoding during evaluation to check those.
`score_episode` and ordinary model inference keep their full decoding/reporting
behavior. Model-only replay is available through `model.prepare(inputs, request)`
and `model.forward_prepared(prepared_visible, decode=True)`.

Prepared data contain no learned features, Unit logits, LL covariance/factorization,
or retained autograd graph. Those quantities are recomputed after every parameter
update. The snapshot does not follow edits to the caller's original tensors:
changed values, roles, targets, code seeds, schema or preprocessing configuration
require a new preparation. Ordinary in-place edits to prepared tensors are rejected
using tensor version metadata, without device scalar reads. Do not bypass version
tracking with `.data` or external storage writes; prepare outside `inference_mode`.
Replacing the encoder or its configuration requires rebuilding the plan. Only the
built-in parameter-independent `ValueEncoder` supports this replay API; custom
encoders retain the ordinary uncached path.

The numeric fit runner now maintains a run-local LRU of at most eight prepared
mask/seed episodes. Table values move to the selected device once for this bank;
mask/seed selection and equal-weight sequential gradient accumulation are unchanged.
An evicted entry is rebuilt deterministically. Resume rebuilds the bank from the
existing sampler cursor and fixed identities; prepared objects are never serialized
as model state. Checkpoint capture, finite-state checks, and interruption recovery
retain their existing behavior.

A bounded comparison is available as:

```bash
uv run tabu-lab restoration prepared-benchmark --device cpu --output new-check.json
```

It checks loss, encodings and parameter-gradient agreement, then separately times
fresh full scoring, prepared full scoring, and prepared scoring without decoding
or Python reports. The fixed six-row mixed-damage panel uses FP64 and one CPU
thread (or explicit `--device cuda:0`), two warmups and four alternating rounds.
One-time preparation is reported separately; optimizer updates and checkpoint I/O
are outside these timings. JSON binds the source files and refuses overwrite.
This is an implementation check, not a fitting campaign or a throughput claim for
other table sizes or hardware.
