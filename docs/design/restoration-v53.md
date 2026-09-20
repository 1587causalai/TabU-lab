# V5.3 reference implementation

Status: local implementation preparation, `local_unissued`; no formal fit,
reserved evaluation, GPU qualification, reviewed claim or release.

This path implements the V5.3 default design while keeping historical
Restoration/TAR model and checkpoint identities separate. The previous working
implementation is preserved in Git commit `ae5a6ca`; new runs use the explicit
codec and loss identities below.

The protected canonical design in the parent TabU project remains
`latex/model-factory/table-restoration/end-to-end-design.tex`. V5.3 is a
versioned Gen-5.x revision within the table-restoration lineage, not Gen-6
and not a replacement for that canonical file. The restoration task,
visible-evidence/scorer-truth boundary, episode contract and training paradigm
remain fifth-generation; encoding, readout and version-specific admission
rules must still be identified explicitly. The parent topic README records
the document lineage.

Versioned design source adopted by this implementation:
`latex/model-factory/table-restoration/TabU_V5p3_Refined_Complete/TabU_V5p3_Refined_Complete.tex`.
The main TeX source is authoritative for current design decisions. This
implementation note records a particular implementation state only: when its
snapshot language conflicts with the main TeX source, the TeX source wins and
this note must be updated before it is used as design evidence.
The initial source snapshot read on 2026-09-19 had SHA256
`e6803a713e807a39f484169d64e9f55485bb4f2329f8353dd8b971cde76ff7e7`.
The earlier 2026-09-20 Gaussian-default snapshot had SHA256
`8d4ba6b21d0dd8d9395fd2bbaf72c1260e19386f04d8debe09cf12cafc3688e2`.
The earlier combinatorial-default source snapshot had SHA256
`b4ff4d612b2d577548930839ed88b1c8c87ab327c751b4ae634f4f71e04dfd2b`.
The small-presence/isometric-input revision has SHA256
`d5436f057f2781b6403e58b87a7de7addaf854bd3445565d1a3f54240cd170b4`.
Main Steps 1–5 are the contract: raw constant-weight compositional coding is
the default, while unit-Gaussian coding is an explicitly selected comparison
mode; historical mechanisms require explicit opt-in.
The living manuscript can advance independently of this recorded snapshot;
this file must not be used to infer a newer default from an older snapshot.

## Implemented path

| Design | Code / behavior |
|---|---|
| Numeric value space | Visible z-score with epsilon floor; $e=q_a+zb_a$ for input and answer. Gaussian unit bases or two distinct raw 128/4 bases |
| Nominal values | One unit Gaussian or raw 128/8 identity per visible category; scorer rejects hidden classes lacking a visible code |
| Ordinal values | $e=q_a+r_a(c)b_a$: one shared affine line per column. The default uses unit Gaussian or raw 128/4 base/direction vectors; the complete declared rank domain is fixed before scoring |
| Input projection | Shared bias-free $W_{\rm enc}$, unscaled thin QR initialization and isometric QR parameterization throughout training; single shared Cell/Unit/Feature seeds |
| Backbone | Existing `AxialBackbone`: column collect/read, then direct row; default 256 slots, width 128, V5.3 presence threshold $10^{-6}$; direct-axis control remains configurable |
| Unit refinement | `unit_layers=0` is exact identity; positive depth uses OMAB with `visible.any(-1)` eligibility and does not write back Cells/Features |
| Regression geometry | `regression_width=None` is identity; explicit width enables learned bias-free $P_R$ |
| Column-shared LL | `readout.py`: all $N$ current Units as centers, $\pi_r=1/N$; one FP64 Cholesky factorization per requested supported column |
| Evaluation | $\widehat e_{ra}=\bar e_{ra}+B_a(c_{ra}-\bar c_{ra})$; numeric centered projection and inverse scale, nominal nearest visible code, ordinal projection onto the shared rank direction and nearest declared rank |
| Loss | All original observations remain in the scoring contract; numeric $128\times$ coordinate MSE and discrete coordinate MSE. Separate type/state means; Query coefficient 1, retained coefficient 0 by default |
| Fixed episode reuse | Visible-only preparation snapshots, mutation guards; carriers, Unit weights and slopes recomputed after every parameter update |

The default solves full 128-dimensional answer right-hand sides. Numeric scalar
right-hand-side elimination is a future equivalent optimization, not required
for correctness. Decoder projection does not replace the full-vector loss.

## Run a bounded local check

From this checkout, with the repository environment available:

```sh
uv run python examples/restoration_v53_smoke.py
uv run pytest tests/unit/restoration_v53 -q
```

The smoke example uses a declared six-row mixed synthetic table, default
128-wide model, FP64 CPU arithmetic and one optimizer step. It prints a JSON
observation; it does not download data, publish a checkpoint, start W&B, or
register evidence. A lower toy loss is not evidence of generalization.

Minimal training API:

```python
from tabu_lab.models.restoration_v53 import (
    V53Model, prepare_episode, score_prepared_episode,
)

model = V53Model().double()
# inputs, request, truth come from the existing make_episode contract.
fixed = prepare_episode(model, inputs, request, truth)
score = score_prepared_episode(model, fixed)
score.loss.backward()
```

`model(inputs, request)` accepts visible data and requested addresses only.
It also works under `torch.inference_mode()`: preparation temporarily creates
ordinary, gradient-free owned snapshots so their version counters remain
available, then forward resumes the caller's inference context. Prepared
snapshot mutation checks remain active for inference and training reuse.
`TruthSidecar` belongs to the scorer. Inference can request a subset; training
must include every original observation, retain at least two supports per
supervised column, and retain all required nominal answer codes. The current
codec also requires at least two distinct visible values per supervised numeric
column; ordinal supports may have equal ranks, which is a degenerate fitting
case, not a missing-code error. Inference
returns explicit `no-support` for empty columns. Invalid training episodes fail
before the neural forward, with no silent target removal. Direct V5.3 scorer
preparation also requires nonempty Query, including callers that manually
construct the input/request/sidecar instead of using `make_episode`.

New curriculum random-cell recipes default to the manuscript's whole-column
numeric guard: population std greater than twice the full IQR (floor 1e-6)
keeps that column visible. Supervised-row recipes keep this guard disabled;
explicit `none` and historical guard recipes remain available. This protection
changes eligible Query coverage and is reported separately from fit metrics.

The default loss is $\mathcal L_Q+\lambda_{\rm restore}\mathcal L_V$ with
$\lambda_{\rm restore}=0$, represented by `V53LossConfig(state_weights=(0,1,0,0))`.
Each nonempty numeric/discrete branch is averaged separately within each state.
Set `(lambda_restore,1,0,0)` to enable retained reconstruction; explicit `None`
retains the historical mixed-state mean. State weights do not change masking,
curriculum sampling, or which truth-bearing addresses must pass preflight.
The historical fit CLI has not been switched to V5.3; callers must use this entry point so its loss scale
cannot silently fall back to the old scalar-answer assumptions.

## Codec identity and historical candidates

New construction defaults to `V53Config(codec_version="constant_weight_v1",
numeric_scaling="zscore")`. Numeric statistics use only visible values,
$\sigma_a^2=n_a^{-1}\sum_i(x_i-\bar x_a)^2$, with scale
$\max(\sigma_a,\varepsilon)$. `median_half_iqr` remains an independent option.

```python
from tabu_lab.models.restoration_v53 import V53Config, V53LossConfig

combinatorial = V53Config()  # constant_weight_v1 + zscore
gaussian = V53Config(codec_version="unit_gaussian_v2")
old_gaussian = V53Config(codec_version="unit_gaussian_v1")
legacy = V53Config(codec_version="legacy_v53", numeric_scaling="median_half_iqr")
old_loss = V53LossConfig(state_weights=None)
```

The default ordinal lift is $e_a(c)=q_a+r_a(c)b_a$, matching numeric's shared
affine line. The declared order is carried by normalized rank, and inputs,
answers and scorer reuse the same fixed column-level base and direction.
Decoding projects onto $b_a$ and matches the nearest declared rank. Training
still retains all 128 error coordinates.

`constant_weight_v1` uses raw binary vectors without dividing by $\sqrt{k}$:
numeric and ordinal bases have four ones; nominal identities have eight.
Numeric and ordinal bases are distinct. Capacity overflow fails explicitly.
Sampling uses a local CPU generator, stable column identity and code seed. The
nominal visible class set fixes its sparse codebook; ordinal uses the full
declared rank domain.
Numeric decoding divides the centered dot product by $\|b_a\|^2$ (1 or 4).
Consequently numeric loss is $(\hat z-z)^2$ in G and $4(\hat z-z)^2$ in C;
nominal and ordinal retain coordinate MSE. A different sparse normalization
would require its own codec version.

`unit_gaussian_v1` preserves the historical shared-origin Gaussian ordinal line
and projected nearest-rank decoder. `unit_gaussian_v2` preserves the explicit
category-identity-plus-rank candidate. `legacy_v53` preserves 128/8 discrete
answer codes and ordinal rank addition to the input only. Replaying the
previous training objective also requires `old_loss`.

`as_dict` records both fields; `from_dict` rejects configurations missing either
field rather than treating an old configuration as the new default. A model
state dictionary also records `_codec_signature`, and loading rejects a
conflicting signature even with `strict=False`. Unversioned historical weights
require both the explicit historical codec and the input-geometry selection
below. This is codec compatibility, not permission to treat a changed codec or
data population as a strict optimizer resume. Record the full config,
`code_seed`, schema, data/mask identity and source revision with experiment
artifacts; prepared tensors are recreated from that visible episode.

## Input isometry and presence scale

V5.3's exported `BackboneConfig` defaults to `tau_presence=1.0`, making a
unit-norm readout the half-participation point. The historical
`restoration.backbone.BackboneConfig` retains the same default. The existing
presence function and its independently learned readouts are unchanged. A
smaller threshold such as `1e-6` is an explicit ablation rather than the
default. The reference mass, norm epsilon, matching bandwidth, loss and clip
threshold are independent settings and are not rescaled by this choice.

`input_projection="isometric_qr"` is the new default. Its effective matrix obeys
$W_{\rm enc}^{\top}W_{\rm enc}=I_{128}$ for both square and tall matrices. A
PyTorch parameterization maps the optimizer's full-rank coordinate matrix to
the positive-diagonal thin QR factor. Therefore a plain AdamW step, the existing
mixed Muon/AdamW optimizer, or checkpoint loading cannot silently release the
isometry constraint. The raw coordinates are stored under
`encoder.projection.parametrizations.weight.original`; `projection.weight`
is the computed isometric matrix, not a separately updated parameter. The
curriculum's existing optimizer partition keeps these coordinates in the
non-decayed AdamW group, including after switching backbone weights to Muon.

The encoder computes that matrix once per forward. Nonfinite or numerically
rank-deficient coordinates fail explicitly. The runner validates the effective
Gram matrix after each update, and checkpoint loading validates it again.
This preserves raw codec energy (e.g. nominal C energy eight), not a unit-norm
normalization of all values. No gradient-clip or fit improvement follows solely
from this algebraic property; matched measurements are required.

Historical replay must explicitly set `input_projection="legacy_scaled"`
and the saved `backbone.tau_presence` (original default 1), alongside its codec
and loss. That mode retains the original unscaled-parameter storage and $Q/8$
initialization. `from_dict` requires explicit input geometry and presence
identity; new curriculum manifests resolve and record both defaults. State
dictionaries record `_geometry_signature`, rejecting different projection or
presence configurations even with `strict=False`. Older states without that
signature are accepted only in the explicitly selected legacy projection
mode. A new parameterization is not a strict resume of old optimizer state.

The QR implementation has CPU float32/float64 tests. Each accelerator must pass
the existing device preflight; these checks do not qualify a CUDA/MPS backend
or replace the separate FP32 readout compatibility work.

## Feature slope extension seam

The default is `V53Config(slope_source="shared_ll")`, with no Feature-to-slope
parameters. For the candidate $\operatorname{vec}(B_a)=W_B f_a$, subclass
`FeatureSlopeProvider` with `forward(feature)` returning `[128, d_R]`, then
inject it into `V53Model(config, feature_slope=provider)` with
`config.slope_source="feature"`. Shape, device and finite checks apply, and
the shared local-mean correction remains in the readout.

This deliberately selects a different estimator. It does not minimize the
default ridge objective and does not automatically preserve numeric affine
closure. Full vector loss still penalizes deviations from the legal answer
line. The provider is an extension interface, not an enabled or validated
research result. Its parameters are in `state_dict`; its constructor must be
supplied explicitly again when loading. Ordinary V5.3 configs round-trip via
`as_dict` / `from_dict` and need no provider.

## Reference cost and remaining execution work

Let $n_a$ be column supports, $d_u$ Unit width, $d_R$ regression width, $p=128$,
and $T_a$ returned targets. The reference centers each support population
before computing its weighted covariance and cross moment, then averages these
over all $N$ centers. Center chunks of size $b$ materialize
$b\times n_a\times d_R$ and $b\times n_a\times p$ centered tensors. Flattening
their first two axes permits accumulation into one covariance and cross moment
without an $N\times d_R\times d_R$ covariance stack. Per-column fit cost is

$$
O\!\left(Nn_a(d_u+d_R+p+d_R^2+d_Rp)
          +d_R^3+d_R^2p\right).
$$

Returned target evaluation adds
$O(T_an_a(d_u+d_R+p)+T_ad_Rp)$.
Center chunk size $b$ controls current kernel/weight temporaries and centered
populations: $O(bn_a(d_u+d_R+p))$, including the reference kernel's pairwise
difference tensor, plus $O(d_R^2+d_Rp)$ for aggregate moments. Supports are not tiled.
Autograd retains tensors across center chunks; there is no bounded training
memory guarantee. Raw Unit distances are semantically common across columns
but this initial reference recomputes them per column. Cross-column reuse,
source tiling, scalar numeric solves and backward recomputation are explicit
future optimizations that must retain the full support set and fixed centers.

Supports first share a translated origin, then each center subtracts its own
local means before products are formed. This costs more than subtracting
aggregate uncentered second moments, but avoids their silent cancellation under
concentrated kernels and large Cell spans. The joint objective, all supports,
all fixed centers, ridge and gradient paths are unchanged; there is still one
Cholesky factorization per supported requested column. A lower-cost implementation
must retain the same numeric result on the concentration/large-span regression.
Nonfinite intermediates and failed Cholesky solves are reported as numerical
failures; no hidden jitter, diagonal fallback or replacement zero loss occurs.

Existing components reused without modification: `restoration.contracts`,
`answers`, `backbone`, kernel validation, tensor mutation guards, and scorer
preflight. New numeric encoding, shared fit, model configuration and loss live
under `src/tabu_lab/models/restoration_v53/`. No historical model is renamed.

Before a formal run, the separate work is to register model/experiment identity,
connect the intended sampler and fit/checkpoint evaluator, qualify the target
device and memory, then run a bounded preregistered fit with train/reserved
isolation. The current local checks do not authorize capability claims.

## Validation

On 2026-09-20, the regression selection below passed **365 tests**; Ruff and
`git diff --check` also passed. Numeric probe NMSE measures scalar-coordinate
error independently of the answer-code norm and training loss multiplier.

Regression checks cover both current codecs and the historical candidates:
visible-only statistics/codebooks, complete declared ordinal identities,
nearest-code decoding with unequal norms and rank-order ties, raw sparse scale,
full-vector ordinal loss, Query-only and optional retained loss, hidden-truth
isolation, mutation guards, row/column equivariance, and checkpoint identity.
The curriculum tests exercise both current modes through training, evaluation,
and exact optimizer/RNG continuation across AdamW-to-Muon stages.

```sh
uv run pytest tests/unit/restoration_v53 tests/unit/curriculum_v53 \
  tests/unit/test_curriculum_catalog.py \
  tests/unit/restoration/test_answers.py tests/unit/restoration/test_readout.py \
  tests/unit/restoration/test_model.py tests/unit/restoration/test_prepared.py -q
```

These are local CPU implementation checks, not training-fit or generalization
results. Earlier implementation snapshots and their validation notes remain in
Git history. Adding rows changes the LL center population and is not promised
to preserve existing slopes or predictions.

### Input geometry validation, 2026-09-20

The updated geometry passed **591 tests** across V5.3, its curriculum,
historical restoration and the curriculum catalog; Ruff and diff checks passed.
Coverage includes float32/float64 square/tall isometries after ordinary AdamW
updates, QR finite differences, rank-deficient/nonfinite failure, exact
AdamW/Muon checkpoint continuation, and projection/presence identity rejection.

`examples/v53_gradient_audit.py` compares all four combinations of old/new
projection and old/small presence threshold. The recorded CPU FP64 audit used
three existing C1 tables, 32 updates per combination, the same masks/codecs,
matched initial parameters outside the projection, and four fixed train-row
evaluation masks. Learning rate, loss weights and clipping threshold **1**
were held fixed. The compact receipt includes all 384 update losses and norms,
episode hashes, source identity and the two subsequent cosmetic source diffs:
[`v53-input-geometry-20260920.json`](../research/v53-input-geometry-20260920.json).

| Table | Old median gradient norm | New default | Old/new median retained gradient fraction |
|---|---:|---:|---:|
| discoscm_048 | 64.06 | 8.97 | 1.57% / 11.15% |
| xor_classification | 125.60 | 13.27 | 0.80% / 7.54% |
| linear_numeric | 146.19 | 15.59 | 0.69% / 6.42% |

These are the global optimizer gradient norms in each parameterization, not a
parameterization-invariant sensitivity measure. Clipping remained active on
**100% of updates** at threshold 1. Thus the change reduced the measured
gradient magnitude and severity of clipping by about 7–9 times; it did not
resolve the frequency of clipping. The small-threshold-only control did not
consistently improve norms, and isometry with tau=1 had smaller norms than
the new default in this short audit. The chosen small threshold expresses
the intended presence semantics; it is not an empirically optimal setting.

The new default's fixed-probe numeric NMSE changed from 1.0874 to 1.0673,
1.0858 to 1.0751, and 1.1163 to 1.1123, respectively. These short runs do not
establish useful fitting capacity, categorical improvement, long-run stability
or generalization. Loss normalization and any threshold calibration remain
separate questions; neither was silently changed to reduce the clipping count.
