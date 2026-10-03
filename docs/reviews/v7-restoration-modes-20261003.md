# V7 restoration-mode integration — 2026-10-03

This integration adds V7 on top of the published V6 mainline
(`b5472cdbcef9756362f1740629cdb5a2308dc3f7`). Single, joint and mixed tasks use
one model, with explicit cyclic settings, typed codecs, row-equal losses,
per-column trajectories, and a bounded configuration-driven training command.
It does not change the V5.3/V5.5/V6 model defaults or the pretraining program pointer.

## Scope

The integration contains the maintained V7 module, its affine-coupling dependency,
a public synthetic fixture, tests, CLI routing and documentation. Historical
experiment directories, data, checkpoints, host logs, and the unrelated V5.5
residual encoder exploration stay outside this code integration.

Three experiment-specific test files (16 cases) remain with their local historical
scripts; they are not silently skipped in the public suite. The earlier development
checkout's 130 V7 passes included those 16 cases and is a different test scope.
Review-driven regression cases are included in this integration.

## Verification

The clean main-based checkout at
`2de07e381df7659f1ec9d9562ba26b4d4d87b951` passed the **complete repository suite:
1597 passed, 3 warnings, 115.21 seconds**. Its final focused V7/affine-coupling
suite passed all 157 cases. The complete submitted V7 source and tests pass
the repository Ruff rules. The runtime was local macOS Python 3.11 / Torch 2.13.0;
imports were explicitly resolved from the integration checkout.

The source digest over the sorted `src/tabu_lab/**/*.py` relative names and bytes
was `6d84ee3147c771a4fee22a3ff515b4c7f0f4f8d55a1d47e017ea105214191a18`.
The final receipt-only commit changes documentation, with this reviewed and
tested source unchanged.

Full-suite execution used one OpenMP/MKL/OpenBLAS thread to avoid a local mixed
OpenMP/XGBoost wait, and retained failed-test temporary directories only to bound
disk use. An earlier full-suite attempt exposed inherited strict deterministic
settings in MPS tests and an uncontrolled ill-conditioned random inverse fixture.
Tests now restore the caller's backend settings and use three fixed, moderately
perturbed non-identity FP64 coupling fixtures, with both inverse directions checked
at `atol=1e-10, rtol=0`. Production deterministic-algorithm settings are unchanged.

CPU/FP64 and MPS/FP32 exercised all three configured modes, including exact
resumed-versus-uninterrupted final weights and sampling counts. Checks cover
hidden-truth isolation, support/donor selection, one backbone snapshot per round,
row-equal loss weights, exact single-column compatibility, gradient recomputation,
unshared rounds, sampler determinism, train/test isolation, legacy native
checkpoint loading, weights-only parent initialization, and strict-resume drift.

Prior to clean-main integration, a real Mini parent with 125 state tensors loaded
strictly into the new model. CPU/FP64 and MPS/FP32 single- and three-column cases
matched the frozen experiment in all four recovered states, loss, and all 122
active parameter gradients. These are within-device comparisons, not bitwise
cross-device equivalence. The parent and frozen reference are not distributed
with this code package; this remains a local engineering receipt:
[numerical comparison](v7-restoration-modes-20261003-parity.json).

The command-line eight-update synthetic fixture was also rerun from the initial
clean-main candidate, completing three single and five joint episodes (72 Query
cells). Its saved checkpoint SHA256 was
`b15fcbb4d0a26c140498d420008e1afc8f1337e3dfe67536318a3e63d8dae059`.
All 240 frozen scale12 input files retained their hashes. Neither this fixture nor
the migration checks establish mixed-training gains or unseen-table generalization.

Commands from a clean repository checkout:

```bash
python -m pytest tests/unit/restoration_v7 tests/unit/test_coupling.py \
  tests/unit/restoration_v6 tests/unit/test_program_cli.py tests/unit/test_tabur_cli.py -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python -m pytest -q -o tmp_path_retention_policy=failed -o tmp_path_retention_count=1
python -m tabu_lab.cli restoration v7-fit --config examples/v7-restoration/mixed.yaml
python -m tabu_lab.cli restoration v7-fit --config examples/v7-restoration/mixed.yaml \
  --execute --output-dir /path/to/new/smoke-output
```

## Independent review

The `mini-reviewer` agent on `gongqian-mini` was invoked in read-only review mode
through its documented direct Hermes profile. Both invocations exited successfully.
The initial snapshot contained 228 hash-verified files and the main-based patch at
`a7882cb22113188b52592374c4c24ba80e09589c`. Static review covered hidden-truth
isolation, visible-only codecs and LL supports, simultaneous cyclic writes,
single/joint loss reduction, split isolation, checkpoint identity and continuation.
Independent CPU probes reproduced three P2 findings; all were fixed:

| Finding | Correction and independent follow-up evidence |
| --- | --- |
| V6 migration discarded the parent's codec identity | Require the declared composition-v2/zscore identity and exact int64 signature before constructing a model or changing RNG; eight invalid-signature probes rejected without those side effects. |
| Quoted `coupling_scale: "false"` silently enabled scale | Require real booleans in both config and primitive constructors; 24 invalid-input probes across three entry points passed. |
| Invalid optimizer settings passed validation | Check types, finiteness, ranges and beta length during configuration resolution; 31 invalid configurations rejected before model construction, with CPU RNG unchanged. |

The focused follow-up at `2de07e381df7659f1ec9d9562ba26b4d4d87b951`
returned **APPROVE**, verified the nine changed files against the supplied hashes,
closed all three findings and found no remaining blocker in that delta. It also
independently checked all six inverse fixture combinations; maximum absolute error
was `1.7763568394002505e-15`. Codex verified the corresponding source changes and
completed the full 1597-case suite on the same candidate.

The reviewer used Python 3.12.13 / Torch 2.13.0 and bounded CPU probes only. Its
runtime lacked pytest; it did not install dependencies, run the maintained suite,
train models, or verify MPS/CUDA. Its approval is independent code/probe evidence;
the complete regression result above is the integration checkout's local evidence.

## Boundaries

The native runner has a new, explicitly recorded sampler; it does not recreate
the historical column-block order or time/exposure checkpoint grid. CUDA and
remote deployment require their own environment-specific qualification. See the
[configuration guide](../tutorials/v7-restoration-modes.md) for checkpoint boundaries.
