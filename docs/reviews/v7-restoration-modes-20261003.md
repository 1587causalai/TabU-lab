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

The V7 public unit suite has 114 cases. Three experiment-specific test files
(16 cases) remain with their local historical scripts; they are not silently
skipped in the public suite. The earlier development checkout's 130 V7 passes
included those 16 cases and is a different test scope.

## Verification

Before final integration review, the clean main-based checkout passed **153
checks** across V7, affine coupling, V6 (including its archive checks), and the
existing command-interface suites. The complete submitted V7 source and tests
pass the repository Ruff rules. Full-suite and independent-review outcomes are
recorded in the final integration receipt when available.

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

The prior command-line eight-update synthetic fixture completed with three single
and five joint episodes (72 Query cells). Its saved checkpoint SHA256 was
`26414c18424cbd4016ba4fa4a4a4a5d6a1dd9be3f3ad403152d2091a52a4fb3d`.
All 240 frozen scale12 input files retained their hashes. Neither this fixture nor
the migration checks establish mixed-training gains or unseen-table generalization.

Commands from a clean repository checkout:

```bash
python -m pytest tests/unit/restoration_v7 tests/unit/test_coupling.py \
  tests/unit/restoration_v6 tests/unit/test_program_cli.py tests/unit/test_tabur_cli.py -q
python -m tabu_lab.cli restoration v7-fit --config examples/v7-restoration/mixed.yaml
python -m tabu_lab.cli restoration v7-fit --config examples/v7-restoration/mixed.yaml \
  --execute --output-dir /path/to/new/smoke-output
```

The native runner has a new, explicitly recorded sampler; it does not recreate
the historical column-block order or time/exposure checkpoint grid. CUDA and
remote deployment require their own environment-specific qualification. See the
[configuration guide](../tutorials/v7-restoration-modes.md) for checkpoint boundaries.
