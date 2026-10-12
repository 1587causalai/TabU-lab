# V8.1 / V8.2 integration — 2026-10-12

This integration makes the current optional V8.1 and V8.2 implementations
available alongside the inherited V8 baseline. It addresses quadratic row
attention on wide tables and implements the supervised attention rule that
Query rows may receive across rows but never supply cross-row sources.

## Scope and compatibility

- `learned_feature64_v1` learns only the feature vocabulary and preserves the
  64-coordinate donor value interface. Numeric and category directions retain
  their fixed random encoding rules. Historical `learned_pool_v1` remains an
  explicit 32-coordinate mode, not a relabeling of the current mode.
- `presence_linear_mean_v1` uses normalized Q/K and a presence-weighted linear
  summary for same-row attention. The original sum variant remains an explicit
  diagnostic option. Neither linear operator is equivalent to inherited softmax.
- `support_rows_null_v1` restricts cross-row sources to Support rows in the
  encoder, its inverse, column slots and optional Unit layers. Same-row sources
  include every real column; its mean divisor includes Null addresses. Exact-zero
  Null dynamics replace per-Cell source masks on this path.
- `configure_memory` bounds independent batches and enables activation
  recomputation without splitting the table context. Unit recomputation wraps
  `batched`, the entry shared by legacy and supervised paths. Deep copies use
  their own wrapped modules and receive their own gradients.
- Weight schemas and explicit migration helpers preserve version identity.
  Old configurations retain their old policy. Weight migration is not an
  optimizer/RNG resume. Existing factories and adopted checkpoints do not change.

Implementation commits are `b9a08382`, `76eb4dd6`, `d3b97c0c`, `6eeaadc7`, and
`3bcc352b`, based on mainline `a60b2598`. The final documentation commit adds this
record without changing executable code. No private datasets, weights, training
artifacts, or unrelated working-directory changes are included.

## Independent review and fixes

An independent Claude Code 2.1.286 / Opus 5.5 review inspected a frozen snapshot
of `6eeaadc7` plus the Unit recomputation fix, with synthetic CPU/MPS probes.
All 89 completion tests passed on both review and coordinator machines.
The review identified one P2 issue: configuring recomputation before
`copy.deepcopy` left bound original modules captured by wrapper closures.
The coordinator independently reproduced incorrect gradient ownership on CPU
and MPS. Commit `3bcc352b` captures an unbound function and calls the current
`self` in both gradient and no-gradient execution.

Four new CPU/MPS × legacy/supervised regression cases failed before that fix
and passed afterwards. They perturb clone weights, check prediction changes,
compare against a rebuilt reference, and verify that no gradient reaches the
original model. The resulting focused suite passed all 93 cases. The independent
review preceded the deepcopy fix; its closure is established by the reproducer
and regressions, not by a claimed second independent review.

The outer research workspace retains the original review snapshot, transcript,
source hashes, before/after logs, and historical experiment receipts. These are
not distributed with this repository. The self-contained regression tests are
included here. Merge-time validation at `3bcc352b` passed **1963 tests**, with one optional
`pandas` skip and three warnings, in 136.09 seconds. This includes all 93
completion regressions and both classical XGBoost contracts. Ruff and diff
checks passed. See [validation.json](validation.json) and the
[full test log](pytest-full.txt).

## Reproduce

Run from the repository root using its Python 3.11/3.12 environment:

```bash
uv sync --frozen --extra dev
uv run pytest tests/unit/completion_v8 -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run pytest tests -q
uv run python examples/completion_v8_smoke.py --codebook-mode learned_feature64_v1 --row-attention presence_linear_mean_v1 --attention-policy support_rows_null_v1
# Add --device mps on a compatible Mac.
```

The smoke example uses synthetic data, one FP32 update, and weights-only replay.
Full-suite validation uses one external BLAS/OpenMP thread, following the
previously verified local XGBoost test execution setting. This does not change
the model or any training protocol.

## Evidence boundary

This merge establishes implementation availability and regression evidence. It
does not adopt a new default model, start a training run, update remote training
checkouts, or claim predictive quality, official benchmark rank, or unseen-table
generalization. Earlier wide-table measurements refer to their recorded source
and protocol; they are not reissued as measurements of this integration.

Query preprocessing and Gaussian LL joint centers can still depend on Query
grouping. FP32 batch chunking can change rounding, amplified on ill-conditioned
small episodes. Cross-row attention isolation is not whole-model batch
independence, and receiver batching is not independent Query prediction.
