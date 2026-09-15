# Independent review: prepared restoration execution

- Reviewed implementation: `921d13b836e72456118a8850682fb8ebfe701ebd`.
- Check-artifact commit: `f8304e2fcdd6ba607d985bc66a0b94800778ec65`.
- Frozen reference: `cf575458f7c62c09dce14ec05d6d6e40b08e8d94`.
- Status: `local_unissued`.
- Conclusion: no unresolved blocker in the prepared execution, fit-runner cache,
  CLI, tests, or documentation reviewed here.

The review was performed independently of implementation. Before this report,
local `main` at `b77c2b5c000aca380e2fb7fcb2d18d732d8e43d0` and the reviewed
implementation had the same tree, `804e2604456960b7a684dd765da4a3266d7c94e6`.
The implementation files remained unchanged while the check files were added.

## Core review and independent execution

Prepared model data contain visible-only values, statistics, codes and fixed
indices. Clean target encodings and damage-state masks stay in the scorer-owned
section. Learned encoding, backbone operations, Unit geometry and LL systems are
recomputed on every replay. The cache holds no retained autograd graph.

Independent checks extracted the complete restoration package from `cf57545`
using Git source bytes and compared it with the prepared path. They used CPU
FP64, one thread, model seed 709, the mixed damage example, reversed target
order, and AdamW with learning rate `1e-4`:

- All 16 direct/inducing, four category-map, and NW/LL combinations matched
  through three optimizer updates. Carriers, restored encodings, log weights,
  coefficients, decoded values, losses, state reports, parameter gradients,
  model state and optimizer state were bit-exact.
- A custom `ValueEncoder` subclass with a learned gain in `prepare` preserved
  ordinary-forward values and gradients against the reference, including a
  nonzero gain gradient. The prepared API rejected this unsupported preparation
  lifetime explicitly.
- In-place mutation of each of the 44 registered snapshot tensors was rejected
  before replay. Mutation through `.data` or external storage is outside the
  documented contract; preparation requires tensor version counters.
- Empty requests and inference columns with no visible support preserved the
  reference behavior.

The independently executed final command was:

```bash
uv run pytest tests/unit/restoration -q
```

Result: **245 passed in 11.11s** at the reviewed implementation. These tests also
cover truth isolation, owned snapshots, weighted losses, skipped decoding and
reporting, bounded LRU eviction/rebuild, and checkpoint continuation. The runner
rebuilds prepared entries from the fixed identities and sampler cursor on resume.

The CLI retains existing error handling and refuses to overwrite check files.
Its full-scoring and training-scoring variants state whether decoding and Python
reports are included. Numerical checks in learned computation remain active;
skipped inverse scaling does not establish finiteness in original units.

## Checked artifacts and bounded timing

The review independently recomputed all 17 source-file hashes in
[`prepared-cpu.json`](prepared-cpu.json) against both the working source and
the reviewed Git commit. It also recomputed the restoration component digest in
[`implementation-check.json`](implementation-check.json):
`fc17e7f3b7ca66f6904263fc99cea98db92e24bbb000a12fbd8511f9d68d84a5`.

The implementation check records **9/9 passed**, including 16 forward/backward
variants and exact serialized model/optimizer continuation. The prepared timing
check records zero maximum difference in loss, encodings and gradients for each
variant. All 12 timing records were positive and finite; their recomputed
medians equal the stored medians.

Commands used to produce the checked artifacts:

```bash
uv run tabu-lab restoration verify --device cpu --output docs/reviews/restoration-prepared-20260916/implementation-check.json
uv run tabu-lab restoration prepared-benchmark --device cpu --output docs/reviews/restoration-prepared-20260916/prepared-cpu.json
```

Existing output files are immutable. A rerun requires new output paths.

| Forward and backward variant | Median latency |
|---|---:|
| Fresh full scoring | 16.4888 ms |
| Prepared full scoring | 16.1981 ms |
| Prepared training scoring | 14.7816 ms |

One-time preparation took **10.8648 ms**. Prepared training scoring used
**10.35% less latency** than fresh full scoring on this check. Timing uses the
fixed six-row mixed-damage episode, FP64, one CPU thread, two warmup calls,
four alternating rounds and six calls per round. Gradient clearing is timed;
optimizer updates and checkpoint I/O are outside the timed phase. These results
establish neither GPU performance, other-size throughput, a training campaign,
nor an issued model capability claim.

Artifact SHA-256 values:

- `implementation-check.json`:
  `ef32cd272931cc9ed058e2b909696bca59054de44e6687b99967649e9f3d2bf5`.
- `prepared-cpu.json`:
  `8604b6c59a2828ddc7466dc01034d20a7fdd233a240abe957df77f9d2be49423`.

## Full-repository regression boundary

The execution workflow ran `OMP_NUM_THREADS=1 uv run pytest -q` and reported
**789 passed, 9 skipped, 12 failed**. The independent review read its terminal
log and the same-environment reference-run log, checked the reference worktree
revision, and compared all failure IDs. The 12 IDs were identical on `cf57545`.
The reference reran these four files with `PYTHONPATH=src`, the shared Python
environment, and `uv run --no-sync python -m pytest -q`:

| Reference test file | Failures |
|---|---:|
| `tests/contract/test_tar_integration.py` | 3 |
| `tests/unit/tar/test_tar.py` | 3 |
| `tests/unit/test_catalog_projection.py` | 2 |
| `tests/unit/test_training_objective_coordinates.py` | 4 |

The reference result was **12 failed, 40 passed**. Matching failure IDs were:

```text
tests/contract/test_tar_integration.py::test_tar_builder_is_protected_and_spec_cannot_be_substituted
tests/contract/test_tar_integration.py::test_tar_builds_through_both_registry_boundaries
tests/contract/test_tar_integration.py::test_tar_manifest_parity_and_source_binding
tests/unit/tar/test_tar.py::test_exact_default_and_baseline_budget[False-15-35593440]
tests/unit/tar/test_tar.py::test_exact_default_and_baseline_budget[False-23-54516960]
tests/unit/tar/test_tar.py::test_exact_default_and_baseline_budget[True-15-54071520]
tests/unit/test_catalog_projection.py::test_checked_catalog_projections_match_current_sources
tests/unit/test_catalog_projection.py::test_current_catalog_indexes_consolidated_model_sources
tests/unit/test_training_objective_coordinates.py::test_raw_loss_does_not_require_optional_context_baseline_telemetry
tests/unit/test_training_objective_coordinates.py::test_trainer_preserves_custom_objective_call_contract[evidence_keyword]
tests/unit/test_training_objective_coordinates.py::test_trainer_preserves_custom_objective_call_contract[legacy_subclass]
tests/unit/test_training_objective_coordinates.py::test_trainer_preserves_custom_objective_call_contract[two_argument]
```

These baseline failures remain unresolved. The execution workflow additionally
reported passing scoped Ruff checks and package build; those commands were not
rerun by the independent reviewer. The scoped review conclusion does not imply
that the full repository test suite is green.
