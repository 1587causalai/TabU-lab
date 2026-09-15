# Robust numeric codec and bounded fit preparation

Scope: a shared visible-only median/half-IQR numeric codec, the additive
`restoration fit` CLI, and one frozen numeric Small-128 pilot configuration.
This records local implementation checks. It is not a fit result, GPU
qualification, formal receipt, or approval to publish a checkpoint.

## Review and corrections

An independent non-author review checked the numeric codec, the actual
preregistration and historical snapshot, source identity, target coverage,
failure persistence and checkpoint continuation. Review findings corrected
before this preparation was committed include:

- Parse JSON before the YAML fallback so JSON exponent literals in a `.yaml`
  file remain numbers. The actual preregistration now resolves successfully.
- Preserve only a completed, finite update boundary after partial optimizer
  mutation or interruption; failed restore must not create a random checkpoint.
- Take terminal counters from the saved boundary, including an interruption
  between boundary assignment and counter assignment.
- Reject nonfinite parameters/optimizer state and original-unit metric overflow
  while preserving a serializable failed terminal outcome.
- Bind source, protocol, data, masks, budget and execution settings to resume;
  include the actual CUDA device properties when execution initializes CUDA.

The reviewer independently injected interruption at the counter-assignment
boundary and observed matching terminal/checkpoint step and cursor. No
unresolved blocker was found within the reviewed CPU preparation scope.

## Verification

Python 3.11.14 / PyTorch 2.13.0 / CPU:

- Restoration tests: **131 passed** (113 model/codec/readout, 18 pilot runner).
- Full repository: **673 passed, 2 failed, 7 skipped**, with two expected
  observer-degradation warnings. Both failures are the unchanged classical ICL
  tests importing scikit-learn from an environment where XGBoost is present but
  scikit-learn is absent. Running that test file against the unchanged `a0f34f0`
  source reproduced both failures. The full suite is therefore not all green.
- Scoped Ruff and `git diff --check`: passed.
- `uv run --no-sync tabu-lab restoration verify`: all eight reference probes
  passed, including 16 reduced model variants and exact optimizer continuation.
- Actual snapshot/preregistration CLI plan: passed without model allocation or
  a created run directory. The machine-generated [dry plan](dry-plan.json)
  requests `cuda:0` but was resolved in a CPU process; it is not a CUDA check.

The snapshot has 256 rows, an original 204/52 train/reserved split and eight
numeric columns. All 32 sampled rows belong to the train split. Each of four
masks requests all 256 selected observed cells, with 11 Query and 21 visible
values per column. Query payload is zeroed. Reserved rows are not used.

Commands used with this checkout's `src` on `PYTHONPATH` and the existing
development interpreter:

```bash
python -m pytest tests/unit/restoration -q
OMP_NUM_THREADS=1 python -m pytest -q
python -m ruff check src/tabu_lab/models/restoration src/tabu_lab/restoration_fit.py src/tabu_lab/cli.py tests/unit/restoration
uv run --no-sync tabu-lab restoration verify
uv run --no-sync tabu-lab restoration fit \
  --preregistration experiments/local/restoration-small128-pilot/preregistration.yaml \
  --dataset "$RESTORATION_DATA" --device cuda:0 \
  --output-root "$RESTORATION_OUTPUT"
```

The model/runner optimizer updates above are confined to correctness fixtures.
No actual pilot, held-out evaluation, GPU throughput test or training campaign
was executed. The first real attempt still needs target-device qualification
and available compute. Iterative recovery and mixed-type/multi-table sampling
remain outside this pilot.
