# V5.3 reference implementation

Status: local implementation preparation, `local_unissued`; no formal fit,
reserved evaluation, GPU qualification, reviewed claim or release.

This path implements the V5.3 default design while keeping historical
Restoration/TAR model and checkpoint identities separate. The base is the
locally checked `main` commit `431cb86a4a86e2d886fec98fa8efa35f240d0ec9`.
Existing checkouts and their uncommitted work were not copied into this branch.

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
`b40f5c28a8f015992aae0d0f07d933f7dc8258b45c784fb5735555c74b1ec137`.
Main Steps 1–5 are the contract; appendix alternatives require explicit opt-in.
The living manuscript can advance independently of this recorded snapshot.

## Implemented path

| Design | Code / behavior |
|---|---|
| Numeric value space | `encoding.py` / `answers.py`: visible mean and population standard deviation, with epsilon floor; independent normalized Gaussian $q_a,b_a\in\mathbb R^{128}$; $e=q_a+zb_a$ for input and answer |
| Nominal values | One independent unit Gaussian code per visible category; scorer rejects hidden classes lacking a visible code |
| Ordinal values | $e=q_a+r_a(c)b_a$ for input and answer, with normalized rank from the complete declared order; declared but unseen ranks remain scorable |
| Input projection | Shared bias-free $W_{\rm enc}$, thin QR $Q/8$ initialization; single shared Cell/Unit/Feature seeds |
| Backbone | Existing `AxialBackbone`: column collect/read, then direct row; default 256 slots, width 128; direct-axis control remains configurable |
| Unit refinement | `unit_layers=0` is exact identity; positive depth uses OMAB with `visible.any(-1)` eligibility and does not write back Cells/Features |
| Regression geometry | `regression_width=None` is identity; explicit width enables learned bias-free $P_R$ |
| Column-shared LL | `readout.py`: all $N$ current Units as centers, $\pi_r=1/N$; one FP64 Cholesky factorization per requested supported column |
| Evaluation | $\widehat e_{ra}=\bar e_{ra}+B_a(c_{ra}-\bar c_{ra})$; numeric centered projection and inverse scale, nominal nearest visible code, ordinal centered projection and nearest declared rank (lower rank on ties) |
| Loss | `training.py`: all original observed targets; numeric $128\times$ coordinate MSE, discrete coordinate MSE, separate type means |
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

The default loss gives each nonempty type its own mean and adds the two.
`V53LossConfig` also permits explicit discrete and state weights. These do not
change masking, curriculum sampling, or the codec. The historical fit CLI has
not been switched to V5.3; callers must use this entry point so its loss scale
cannot silently fall back to the old scalar-answer assumptions.

## Codec identity and historical candidates

New construction defaults to `V53Config(codec_version="unit_gaussian_v1",
numeric_scaling="zscore")`. For each numeric column, the visible population
standard deviation uses denominator $n_a$, and the scale is
$\max(\sigma_a,\varepsilon)$. Constant/singleton inference uses the epsilon floor;
empty columns return `no-support`. All types share the same fixed 128-dimensional
input and answer lift. Gaussian bases are sampled with a local CPU generator
from the episode seed and stable column key; nominal realizations also include
the declared category index, so another visible class does not remap old codes.
These tensors are nonlearned visible snapshots protected by mutation guards.

The historical codec is an explicit candidate:

```python
legacy = V53Config(codec_version="legacy_v53", numeric_scaling="median_half_iqr")
```

It preserves median/half-IQR numeric scaling, 128/8 constant-weight discrete
answers and the old ordinal rank addition to the input only. The two fields are
independent explicit choices, allowing robust scaling with the new discrete
codec or z-score with the historical discrete codec. Neither choice changes
the numeric $128$ versus discrete $1$ loss multipliers.

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

The 2026-09-20 codec-default update passed **45 V5.3 tests** and **176 adjacent
historical codec/readout/model/prepared tests** (**221 passed** together), plus
Ruff and `git diff --check`. Added coverage includes visible population z-score,
constant/singleton and large-offset inputs, Gaussian nominal identity and
missing-code rejection, full-schema ordinal rank geometry and lower-rank ties,
ordinal continuous loss scaling, truth isolation, new codec mutation guards,
explicit legacy behavior, config/state-dictionary identity checks and numeric
diversity preflight. These are CPU implementation checks; they do not establish
single-table fitting or multi-table generalization.

The final main-checkout check on 2026-09-19 passed the full committed-scope suite:
**982 passed**, with two expected W&B fallback warnings. It used CPU,
PyTorch 2.13.0 and scikit-learn 1.9.0 for the optional classical-baseline tests.
Ruff passed for the V5.3 implementation, tests and example. This is local
implementation validation, not GPU qualification or a training result.

Initial local validation on 2026-09-19, CPU / PyTorch 2.13.0:

- 23 V5.3 tests plus 176 selected historical codec/readout/model/prepared tests:
  **199 passed**.
- Ruff passed for the new implementation, tests and smoke example.
- The default smoke model had 1,066,112 parameters, restored 18 targets, and
  completed a finite forward/backward and one AdamW update.

The combined regression selection was:

```sh
uv run pytest tests/unit/restoration_v53 \
  tests/unit/restoration/test_answers.py tests/unit/restoration/test_readout.py \
  tests/unit/restoration/test_model.py tests/unit/restoration/test_prepared.py -q
```

These are local implementation checks, not issued research receipts.

The subsequent numerical-stability and inference-context correction passed all
**27 tests** in `test_readout.py` and `test_model.py`, plus Ruff on those tests and
the changed `readout.py` / `model.py`. It adds an analytic two-support regression
at Cell span $10^8$ and Unit separation $9$: the requested prediction is about
$0.48132068$, whereas uncentered second-moment subtraction silently returned
approximately zero. Three center chunk sizes agree. Public inference-mode
forward and preparation are compared with `no_grad` output, including inputs
created inside inference mode and rejection of later snapshot mutation.

Correction-check commands, from the repository root with its development
environment available:

```sh
uv run pytest \
  tests/unit/restoration_v53/test_readout.py \
  tests/unit/restoration_v53/test_model.py -q
uv run ruff check \
  src/tabu_lab/models/restoration_v53/readout.py \
  src/tabu_lab/models/restoration_v53/model.py \
  tests/unit/restoration_v53/test_readout.py \
  tests/unit/restoration_v53/test_model.py
```

The tests compare shared LL with an independently constructed joint weighted
least-squares system with one intercept per center and appended ridge rows.
They also cover finite-difference gradients, gradients through unrequested
centers, chunk/support permutation, constant and singleton supports, codec
round-trip and RNG isolation, numeric affine closure and loss normalization,
direct/inducing backbones, Unit eligibility and identity, request independence,
row/column equivariance, Query insertion into an existing empty address, hidden
truth isolation, invalid-episode preflight, prepared mutation rejection,
checkpoint/optimizer continuation, and the explicitly injected Feature seam.

Adding NEW rows changes $N$ and the fitted center population. Unlike inserting
a Query into an existing empty address, that operation is not promised to
preserve the column-shared slope or predictions.
