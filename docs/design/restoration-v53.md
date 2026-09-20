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
The initial source snapshot read on 2026-09-19 had SHA256
`e6803a713e807a39f484169d64e9f55485bb4f2329f8353dd8b971cde76ff7e7`.
The current codec defaults follow the 2026-09-20 source snapshot with SHA256
`8d4ba6b21d0dd8d9395fd2bbaf72c1260e19386f04d8debe09cf12cafc3688e2`.
Main Steps 1–5 are the contract: Gaussian and raw constant-weight codecs are
selectable modes; historical mechanisms require explicit opt-in.
The living manuscript can advance independently of this recorded snapshot.

## Implemented path

| Design | Code / behavior |
|---|---|
| Numeric value space | Visible z-score with epsilon floor; $e=q_a+zb_a$ for input and answer. Gaussian unit bases or two distinct raw 128/4 bases |
| Nominal values | One unit Gaussian or raw 128/8 identity per visible category; scorer rejects hidden classes lacking a visible code |
| Ordinal values | $e=q_{a,c}+r_a(c)b_a$: one identity per declared category and a shared rank direction. Both use unit Gaussian or raw 128/4 vectors; the complete schema codebook is fixed before scoring |
| Input projection | Shared bias-free $W_{\rm enc}$, thin QR $Q/8$ initialization; single shared Cell/Unit/Feature seeds |
| Backbone | Existing `AxialBackbone`: column collect/read, then direct row; default 256 slots, width 128; direct-axis control remains configurable |
| Unit refinement | `unit_layers=0` is exact identity; positive depth uses OMAB with `visible.any(-1)` eligibility and does not write back Cells/Features |
| Regression geometry | `regression_width=None` is identity; explicit width enables learned bias-free $P_R$ |
| Column-shared LL | `readout.py`: all $N$ current Units as centers, $\pi_r=1/N$; one FP64 Cholesky factorization per requested supported column |
| Evaluation | $\widehat e_{ra}=\bar e_{ra}+B_a(c_{ra}-\bar c_{ra})$; numeric centered projection and inverse scale, nominal nearest visible code, ordinal nearest full declared identity-plus-rank code (lower declared rank on ties) |
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
before the neural forward, with no silent target removal.

The default loss is $\mathcal L_Q+\lambda_{\rm restore}\mathcal L_V$ with
$\lambda_{\rm restore}=0$, represented by `V53LossConfig(state_weights=(0,1,0,0))`.
Each nonempty numeric/discrete branch is averaged separately within each state.
Set `(lambda_restore,1,0,0)` to enable retained reconstruction; explicit `None`
retains the historical mixed-state mean. State weights do not change masking,
curriculum sampling, or which truth-bearing addresses must pass preflight.
The historical fit CLI has not been switched to V5.3; callers must use this entry point so its loss scale
cannot silently fall back to the old scalar-answer assumptions.

## Codec identity and historical candidates

New construction defaults to `V53Config(codec_version="unit_gaussian_v2",
numeric_scaling="zscore")`. Numeric statistics use only visible values,
$\sigma_a^2=n_a^{-1}\sum_i(x_i-\bar x_a)^2$, with scale
$\max(\sigma_a,\varepsilon)$. `median_half_iqr` remains an independent option.

```python
from tabu_lab.models.restoration_v53 import V53Config, V53LossConfig

gaussian = V53Config()  # unit_gaussian_v2 + zscore
combinatorial = V53Config(codec_version="constant_weight_v1")
old_gaussian = V53Config(codec_version="unit_gaussian_v1")
legacy = V53Config(codec_version="legacy_v53", numeric_scaling="median_half_iqr")
old_loss = V53LossConfig(state_weights=None)
```

The new ordinal lift is $e_a(c)=q_{a,c}+r_a(c)b_a$. Identities cover the entire
declared domain independently of support visibility and hidden truth. Inputs,
answers and scorer reuse the same fixed tensors. Decoding minimizes full squared
distance, including candidate norms; projection onto $b_a$ or dot-product-only
comparison is generally incorrect. Training retains all 128 error coordinates.

`constant_weight_v1` uses raw binary vectors without dividing by $\sqrt{k}$:
numeric bases and ordinal identities/directions have four ones; nominal
identities have eight. Numeric bases are distinct; same-column identities are
sampled without replacement. The ordinal direction is independent and may
overlap an identity. Capacity overflow fails explicitly. Sampling uses a local
CPU generator, stable column identity and code seed. The nominal visible class
set also fixes its sparse codebook; ordinal uses the full declared domain.
Numeric decoding divides the centered dot product by $\|b_a\|^2$ (1 or 4).
Consequently numeric loss is $(\hat z-z)^2$ in G and $4(\hat z-z)^2$ in C;
nominal and ordinal retain coordinate MSE. A different sparse normalization
would require its own codec version.

`unit_gaussian_v1` preserves the old shared-origin ordinal line and projected
nearest-rank decoder. `legacy_v53` preserves 128/8 discrete answer codes and
ordinal rank addition to the input only. Both remain explicit historical
candidates. Replaying the previous training objective also requires `old_loss`.

`as_dict` records both fields; `from_dict` rejects configurations missing either
field rather than treating an old configuration as the new default. A model
state dictionary also records `_codec_signature`, and loading rejects a
conflicting signature even with `strict=False`. Unversioned historical weights
are accepted only after constructing the explicit historical configuration
above. This is codec compatibility, not permission to treat a changed codec or
data population as a strict optimizer resume. Record the full config,
`code_seed`, schema, data/mask identity and source revision with experiment
artifacts; prepared tensors are recreated from that visible episode.

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
