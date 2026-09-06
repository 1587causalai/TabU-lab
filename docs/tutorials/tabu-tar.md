# TabU-TAR: executable first realization

`tabu.tar` implements the fourth-generation **Typed Additive Readout** design in
[model-design.tex](../design/tar-model-design.tex).
It is a PyTorch correctness reference with independent weights and configuration.
Only bounded local scratch-fitting and held-out diagnostics have run; no
pretraining or broad generalization claim is established. Local implementation
checks are `local_unissued`, not formal experiment receipts or model releases.

The TAR Standard default is **54,071,520 learned parameters**: width 384, 15 blocks, 8 heads,
FFN width 768, 32 semantic slots and 128 inducing slots. Every block uses column
collect/read followed by direct row interaction. The 128 slots remain present for
small tables. CPU and CUDA accept FP32/FP64; the numeric terminal uses FP64.
CPU correctness and bounded DGX2 CUDA forward/backward have been checked;
throughput depends on episode size. See the local benchmark protocol below.

## Named model sizes

The same TAR architecture has three explicit size presets: **Small** (3 blocks,
width 96, FFN 192; 721,464 parameters), **Medium** (6 blocks, width 192, FFN 384;
5,521,008 parameters), and the unchanged **Standard** (15 blocks, width 384,
FFN 768; 54,071,520 parameters). Heads and semantic/inducing slot counts stay
8/32/128. `TARConfig()` and the default builder continue to construct Standard.

Use `tabu-lab tar sizes` to list them or `tabu-lab tar inspect --size small` to
inspect a named size without allocating weights. In Python use
`TabUTARModel(config_for_size("small"))` after importing `config_for_size` from
`tabu_lab.tar_sizes`. A fitting preregistration selects `model_size` and binds the
expected parameter count. Config hashes separate checkpoints; pass
`expected_config=config_for_size(...)` when loading a requested size.

See the [size-family experiment](../../experiments/local/tar-size-fit/README.md)
for executable inputs and the same full-data training protocol. Small and Medium
are intended for fast feedback; they do not replace the standard design or prove
its capabilities. The separate CPU `--smoke` configuration is not Small.
Subsequent implementation and fitting checks start with the
[current Small validation entry](../../experiments/local/tar-small-validation/README.md).

## Inspect and verify

From the `tabu-lab` checkout with its existing environment:

```bash
uv run --no-sync tabu-lab tar inspect
uv run --no-sync tabu-lab tar verify                 # Small by default
uv run --no-sync tabu-lab tar verify --size medium
uv run --no-sync tabu-lab tar verify --full           # explicit Standard
uv run --no-sync tabu-lab tar verify --smoke          # tiny CPU plumbing
uv run --no-sync pytest -q tests/unit/tar
```

`inspect` uses meta tensors without allocating full weights and still defaults to
Standard. `verify` now checks the actual named Small model with mixed
numeric/nominal/ordinal inputs, forward/backward, one AdamW update, exact Null
and a strict checkpoint roundtrip. Select `--size medium` or `--full` for larger
sizes; `--size standard` is equivalent to `--full`. Use `--smoke` explicitly for
the historical tiny plumbing check. These selectors are mutually exclusive.
The verification output names the actual size and parameters. This is a bounded
component probe, not a fitting experiment or pretraining schedule.

## Input and prediction

```python
import torch
from tabu_lab.models import build_model
from tabu_lab.models.tar import TAREpisode, TARFeature

model = build_model("tabu.tar")  # explicit new identity; old default is preserved
values = torch.tensor([[1., 0.], [2., 1.], [3., 0.], [4., 999.]])
visible = torch.ones_like(values, dtype=torch.bool)
visible[3, 1] = False
queries = torch.zeros_like(visible)
queries[3, 1] = True
features = (
    TARFeature(column_id=0),
    TARFeature("nominal", ("no", "yes"), column_id=1),
)
episode = TAREpisode.from_table(values, visible, queries, features, codebook_seed=7)
with torch.no_grad():
    output = model(episode)
prediction = output.predictions[0]
print(prediction.status, prediction.probabilities)
```

`from_table` physically erases every nonvisible value, including the placeholder
999. Direct `TAREpisode` construction is validated and rejects unerased hidden
values. `visible` contains factual sources; `queries` contains receivers only;
the remaining cells and the bank corner are exact Null. Numeric zero is an
observed value, never an implicit missing marker. Columns require unique stable
nonnegative IDs used only for reproducible codebook RNG, never learned IDs.

Discrete values are integer indices in an externally declared ordered domain.
Nominal carriers use nonlearned, episode-specific unit vectors for visible
classes only. Ordinal carriers use the declared order. Numeric centering and
scaling read visible values only. Carry `codebooks` and `codebook_classes` from
`TAROutput` when reproducing a particular preprocessing realization.

`TAROutput` includes full carriers and additive responses for inspection. Prediction
addresses are episode-local. Fewer than two same-column supports yields
`insufficient-support`, with no fabricated value; scoring such an episode fails.
Numerical failures raise rather than silently changing ridge, bandwidth or support.
For categorical targets, `value` is an index; `probabilities` follows the full
declared domain, including unobserved classes with the specified smoothing mass.

## Training boundary and synthetic starting prior

```python
from tabu_lab.models.tar import TabUTARModel, TARTrainer, TARTrainingConfig
from tabu_lab.tar_sizes import config_for_size
from tabu_lab.models.tar.data import synthetic_episode

# Named Small is the default scale for new validation work.
model = TabUTARModel(config_for_size("small"))
trainer = TARTrainer(model, TARTrainingConfig(effective_episode_batch=2,
                                            warmup_steps=1, optimizer_steps=4))
for update in range(4):
    batch = [synthetic_episode(update * 2 + j, rows=(16,), columns=(4,))
             for j in range(2)]
    print(trainer.train_step(batch))
```

Each item is `(truth_free_episode, address_to_truth_sidecar)`. Forward never
receives the sidecar. The objective sums the means of nonempty numeric and
discrete branches, then averages episodes. Numeric loss uses the visible-only
scale; discrete loss is smoothed-PMF NLL. The trainer accumulates one episode at a
time, clips the global gradient norm, and performs one AdamW update only after the
entire declared batch succeeds. Weight decay applies to matrices, excluding
semantic/inducing seeds, frequency parameters, normalization gains and readout
vectors. Learning-rate warmup and cosine decay use the completed update count.

For repeated training on **one fixed dataset**, reconstruct the episode each
update with `models.tar.episodes.sample_supervised_episode`: supply only the
training pool, stable row IDs, a root seed and a new episode ID. Context/query
roles are sampled uniformly by address, and nominal vectors use a separate RNG
stream. Keep the same model and optimizer. A categorical domain index is a stable
semantic label; the vector codebook representing it is **episode-specific**.
Within an episode, all occurrences of a category in the same column share a code.
Only replaying the same episode (including paired initial/final evaluation) reuses
its codebook. Never cache codebooks or context carriers across new episodes.

The corrected bounded real-data diagnostic is
[`tar-small-validation`](../../experiments/local/tar-small-validation/README.md),
using explicitly named Small: all 150 Iris
rows (120/30 stratified split) and all 442 Diabetes rows (353/89 random split).
Every row is assigned exactly once; subset coverage is rejected. Training masks
40/120 and 118/353 labels respectively. The previous small-pool full-context
protocol is historical and is not a full-dataset benchmark.
The archived fixed-episode run answers a different, narrower question.

`synthetic_episode(episode_id, seed=..., split_id=...)` implements the design's
initial DAG recipe with independent prior, missingness, mask and codebook RNG
streams. It returns a reproducible world and mask without exposing hidden truth
to the model. A split uses a different RNG namespace. Mask selection depends on
addresses and support counts. This is an explicit starting prior, not a validated
replacement for TabPFN's training distribution. Formal training still needs its
own reviewed ExperimentSpec and data/evaluation protocol; no large run is launched
by importing these helpers.

## Full train context and joint test prediction

For the current real-data diagnostic, every training episode contains the entire
fixed train split with a fresh partial label mask. At test time all train labels
are visible and all test labels are hidden; predict the complete test set in one
episode. Do not reuse a sampled training context as the test context.

```python
from tabu_lab.models.tar import predict_joint_supervised
predictions = predict_joint_supervised(
    model, train_values, train_visible, test_values, test_visible,
    features, response_column=-1, codebook_seed=7, episode_id=0,
)
```

All supplied train labels must be visible. All test label placeholders are
physically erased. This is one episode with one independent codebook realization;
known test covariates participate jointly in column evidence. It is a declared
joint-test (transductive input) configuration, not numerically equivalent to
independent-row batching. The original mathematical design's initial independent
prediction convention remains available below; the current experiment declares
its input choice explicitly in its own preregistration.

## Independent test-row inference

```python
from tabu_lab.models.tar import predict_supervised

predictions = predict_supervised(
    model, training_values, training_visible, test_covariates, test_visible,
    features, response_column=-1, codebook_seed=7,
)
```

Both input matrices include the response column; its test values are always
removed. Each output tuple entry corresponds to the same-index test row, with
an episode-local address `(len(training_values), response_column % M)`. Every
test row gets an episode containing training facts and its own covariates.
`codebook_seed` is the root of independent per-row codebooks. Pass stable unique
`episode_ids` when chunking or reordering test rows; otherwise IDs are their indices
within this call. Repeating a call with the same root and IDs replays those episodes;
use new IDs for new realizations. Changing another test row cannot change its
prediction. Training representations
are recomputed because visible test covariates can affect column dynamics.
`forward_batch` similarly keeps ragged episodes independent.

## Gates and baselines

```python
from dataclasses import replace
from tabu_lab.models.tar import TARConfig, TabUTARModel

base = TARConfig()
direct = TabUTARModel(replace(base, inducing_enabled=False))       # 35,593,440
matched = TabUTARModel(replace(base, inducing_enabled=False, blocks=23))  # 54,516,960
unit_only = TabUTARModel(replace(base, lambda_feature=0, lambda_unit=1))
```

All four gate combinations are legal. The response is
`W c + lambda_feature F w_F + lambda_unit U w_U`; there is no hidden gate hierarchy.
With the default same-column Gaussian LL/NW, the Feature term is a common
translation and cancels from prediction differences. Its exclusive parameters
receive zero task gradient mathematically (up to floating-point roundoff), and
are not weight-decayed. Distinct configurations do not automatically imply distinct
predictions. The without-inducing baseline constructs two OMABs per block and
has no inducing parameters; it is not a small-N fallback or an S=0 setting.
RecSys two-axis communication and learned attention matching remain separate
unimplemented customizations.

## Checkpoints and implementation boundaries

```python
from tabu_lab.models.tar import save_checkpoint, load_checkpoint

save_checkpoint(model, "new-checkpoint", trainer=trainer)
model, trainer = load_checkpoint("new-checkpoint", restore_trainer=True)
# Resume the next deterministic episode IDs supplied by the caller.
```

Use a new directory; existing checkpoints are never overwritten. Weights use
SafeTensors; optimizer restoration uses PyTorch's restricted `weights_only=True`
loader. The manifest binds the TAR identity, configuration, implementation files,
ModelSpec/design reference, and weight/optimizer hashes. Different config/code or
legacy weights require an explicit migration; they are not silently accepted.
The trainer restores Adam state and update count. Save episode/split IDs in the
experiment record; arbitrary external data-loader state is not serialized here.

The main modules are [configuration](../../src/tabu_lab/models/tar/config.py),
[typed input](../../src/tabu_lab/models/tar/types.py),
[compiler/backbone](../../src/tabu_lab/models/tar/model.py),
[OMAB](../../src/tabu_lab/models/tar/attention.py),
[typed terminal](../../src/tabu_lab/models/tar/terminal.py),
[objective/optimizer](../../src/tabu_lab/models/tar/training.py), and
[checkpoint](../../src/tabu_lab/models/tar/checkpoint.py).
The [contract tests](../../tests/unit/tar/test_tar.py) cover information flow,
permutation, independent gates, translation cancellation, global chunk reduction,
LL against an independent normal-equation solve, gradients and exact optimizer resume.

This reference stores the dense augmented carrier and autograd intermediates.
Attention merges source chunks under one global softmax; it does not average
independently normalized chunks. OMAB accepts `[... , receivers, d]` / `[... , sources, d]` with masks over
`[..., receivers]` / `[..., sources]`. Every batch group has independent evidence
and softmax normalization. Columns are batched in the design's four-column
window (`collect_column_chunk=4`); rows are batched in windows bounded by
`receiver_chunk_rows`. There are no per-row or per-column OMAB calls inside
those windows. The order collect → read → row → next layer is preserved.
Sources are selected by episode visibility AND exact nonzero carrier, receivers
by the episode Null mask. Empty collect groups are overwritten with zero;
empty attention suppresses the entire output projection including bias.
All-masked chunks use finite empty softmax statistics, preventing NaN gradients.
Eligibility uses tensor masks without dynamic `nonzero`/gather in OMAB.
Input/output finite checks run once per batched OMAB call. The terminal chunks
queries but currently materializes all same-column support differences per chunk.
The frozen serial oracle in `tests/unit/tar/serial_reference.py` checks batched
forward, input gradients and parameter gradients; full-model tests cover both
inducing and direct baselines and multiple column chunk sizes. The benchmark CLI
and its [preregistered protocol](../../experiments/local/tar-batched-benchmark/README.md)
compare identical weights, episodes and precision. Full multi-episode ragged
`forward_batch` still iterates episodes, and the compiler and typed terminal still
have column loops; this change targets repeated backbone axis operations.
Large-table throughput, fused kernels, offload, mixed precision and distributed
training remain separate work.
The 54M parameter match is not a throughput or accuracy equivalence claim.


The fitting receipt also records `git_source.commit` and whether that checkout
was clean or dirty. A copied source snapshot without Git metadata reports
`unavailable`; it still retains model, runner, config and data hashes. A clean
commit remains `local_unissued` in this diagnostic. The Git fields contain no
checkout path or remote URL, and do not alter checkpoint identity.
