# V8 completion baseline integration — 2026-10-11

This change moves the previously uncommitted V8 completion implementation into
the public mainline, with its tests and a self-contained synthetic example.
It preserves the **64-coordinate V8 baseline**. V8.1's proposed 32-coordinate
learnable codebooks are a separate design and are not implemented by this change.
No factory default, adopted model, real-data training budget or historical
checkpoint is changed.

## Imported scope

- `src/tabu_lab/models/completion_v8/`: typed configuration, prepared completion
  episodes, shared Query seed, four reversible whole-table coupling blocks,
  inherited V7/V7.3 Token Dynamics, Gaussian column-shared local-linear recovery,
  terminal masked code MSE, explicit weights-only migration, save/load and
  per-instance execution precision.
- Optional positive/signed attention readouts and their Cell-only execution
  path are retained with explicit configuration. Their research status remains
  paused; Gaussian LL remains the default. Inclusion is not a performance claim.
- Four narrowly scoped shared dependencies: `restoration/backbone.py` and
  `restoration/readout.py` accept instance/call precision overrides;
  `restoration_v7/readout.py` threads that dtype through kernel, accumulation and
  solve; `_dtype.py` only clarifies the legacy policy. With no override, the
  existing V5/V7 default computation remains selected.
- Tests for algebraic inverse, masking and truth isolation, synchronous latent
  assembly, gradients, migration, checkpoint validation/replay, precision
  isolation and CPU/MPS execution. The V8 test directory is a package to avoid
  collisions with legacy `test_readout.py` modules during whole-suite collection.
- A runnable synthetic smoke example and navigation/documentation.

The branch starts at `f8e2f6963bd8f4aad7890d7d5e3c5aa61b444970`, not at the old
V5.4 working branch that contained the uncommitted files. Other uncommitted
experiments and the unrelated local V7 value-encoding refactor are excluded.
The import manifest records original file hashes and the assembled candidate
hashes, including the small test/document integration repairs.

## Reproduce the integration checks

Use the repository's Python 3.11/3.12 environment:

```bash
uv sync --frozen --extra dev
uv run python examples/completion_v8_smoke.py
uv run python examples/completion_v8_smoke.py --device mps  # if available
uv run pytest tests/unit/completion_v8 -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run pytest tests -q
```

The example uses a small generated numeric/nominal/ordinal table, one FP32
optimizer step, and a temporary weights-only checkpoint. It checks finite,
nonzero gradients and exact saved-model replay. It does not claim that one
step must lower held-out loss, reproduce a training campaign, or restore an
optimizer. `--device cuda` is available for an explicitly selected CUDA host;
CUDA is not implied by local CPU/MPS checks.

The local whole-suite run uses one external BLAS/OpenMP thread. An earlier
two-thread run stalled inside the pre-existing XGBoost classical-ICL test's
OpenMP barrier and was interrupted; that test passed independently with the
one-thread environment. This changes the local test execution environment, not
model formulas or a training recipe. The initial duplicate-test-module collection
error was repaired by packaging the new V8 tests before the whole-suite rerun.

Validation results are recorded in `validation.json`. The accompanying
`smoke-cpu.json` and `smoke-mps.json` identify the actual execution settings;
`checkpoint-load.json` records a local real-checkpoint compatibility check.
These are integration evidence (`local_unissued`), not predictive-quality
measurements or a new ranking.

## Independent review

An independent read-only reviewer inspected the candidate worktree against the
mainline base. The first pass found no blocking correctness issue and separately
ran 39 V8 tests (including MPS) and 127 affected legacy/V7 tests. Review covered
hidden-truth isolation, response/inverse gradients, complete latent assembly,
weights migration, checkpoint round trips, precision instance isolation and
backward-compatible defaults. The final review also covers the smoke example,
documentation and test collection repair; its result is recorded in validation.

The owner explicitly requested committing and merging this baseline on
2026-10-11. This records the authorization for the scoped integration, not a
new authorization to train or implement V8.1.

## Historical experiment boundary

The outer TabU research workspace retains the original `preparations/` reports,
frozen execution sources, dataset splits, model checkpoints and optimizer/RNG
states. They are **not part of this repository checkout**. Module README paths
to those materials are historical references, not broken promises that a public
clone includes private artifacts. This merge makes the implementation and a
synthetic execution path reproducible; full real-data replay still requires the
recorded external inputs and historical execution source.

The local reference checkpoint identified by SHA256
`4c7a1261839393a20d2d2a3c9937c18ce30d015be222819a566871f15e8120aa`
was labelled 924.1 under its historical 51-table, up-to-128-Query protocol.
That was a local approximate comparison against reference methods evaluated on
different test coverage, not an official TabArena ranking or independent
unseen-table claim. This integration only checks loading that checkpoint and
preserving its stored tensor values; it does not rerun its ranking.

The package's `load_weights` preserves stored parameter dtype. FP32 checkpoints
select the current per-instance fast precision policy by default; historical
mixed/reference arithmetic requires its explicit policy. Stored weight identity
alone therefore does not guarantee historical prediction identity. Exact old
run replay uses its frozen code, inputs, codec assignments and evaluation setup.

No new large-window qualification or CUDA run is part of this integration.
Historical activation recomputation and execution overlays remain in their
recorded run packages; the core model does not silently enable them. Previous
training and model-quality limitations remain unchanged.
