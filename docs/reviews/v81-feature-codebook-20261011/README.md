# V8.1 feature-only codebook implementation — 2026-10-11

The selected change is a trainable **feature** vocabulary, with numeric and
category directions continuing to use the baseline random rule. The 10000×32
feature bank adds 320000 parameters. Directions are constants, not optimizer
parameters. V8 remains the default; `V8Config(..., codebook_mode="learned_pool_v1")`
selects V8.1. This is local engineering validation, not a quality claim.

The implementation reuses one completion pipeline. C64/8 sampling becomes
weight-8 binary directions in 32 dimensions; G64 becomes unit-Gaussian directions
in 32 dimensions. The old codec entry and donor config remain unchanged. The
query/value interface is 32D, reversible branches are 16+16, and the 128D
backbone, Q/K head width 32, FFN hidden width 64, Gaussian LL and MSE semantics
are preserved. The reference target stays detached.

Preparation holds a feature-index plan and fixed random offsets. Every forward
binds current feature weights, and its output retains the bound codec for loss
and decoding. Partial recoding mixes indices **and** random directions; no stale
learned-code cache is retained across steps. The new weights schema is
`completion_v81_weights_v1`; old configs and schemas continue loading as V8.

Validation:

- [Full receipt and source hashes](validation.json): 1921 passed, one optional
  pandas skip, three existing warnings; Ruff and diff checks passed.
- Focused completion suite: 51 passed, including CPU/MPS two-step optimization,
  feature gradient finite differences with fixed targets, hidden-truth isolation,
  unselected feature-row zero data gradients, partial recoding, mixed-type decoding,
  ordinal domain size above 10000, inverse/recomputation gradients and strict replay.
- [CPU](smoke-cpu.json) and [MPS](smoke-mps.json): one synthetic FP32 update,
  bitwise saved-weight prediction and decoded-value replay on each device.
- [Baseline parity](legacy-parity.json): same initialization seed under main
  a60b2598 and this source, all 83 small-model state entries and predictions exactly
  equal with the new mode disabled. Config serialization is unchanged.
- [Real donor migration](migration.json): existing 924.1 weight file loaded,
  86 backbone/seed entries copied exactly, 47 new/interface entries initialized,
  all parameters finite. This was an in-memory migration check, not a new saved
  checkpoint, optimizer resume or real-data forward run.

Run from this checkout with its source selected:

```bash
PYTHONPATH=src python -m pytest tests/unit/completion_v8 -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python examples/completion_v8_smoke.py --codebook-mode learned_pool_v1
PYTHONPATH=src python examples/completion_v8_smoke.py --codebook-mode learned_pool_v1 --device mps
```

The fixture supplies explicit seeds. Exact future training resume also needs the
external runner's optimizer, sampler/input seeds, partial selection and reference
seed state. No CUDA, real-dataset performance or 1024-row wide-table qualification
was performed in this change. No performance ranking is implied by these tests.
