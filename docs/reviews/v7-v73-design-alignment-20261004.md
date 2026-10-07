# V7 / V7.3 design alignment — 2026-10-04

This is the historical review at the first version-boundary commit. Later asinh
and row dual-stream changes, source identity checks, and current integration
results are recorded in the [2026-10-08 integration](v73-integration-20261008/README.md).
The softlog factory default and pending checks below describe this dated snapshot.

Status: local implementation and independent review. Base: `af7e5f519df781dd42e402359e280295b28edf51` (the merged single/joint/mixed V7 implementation). This work introduces an explicit V7.3 protocol while keeping recorded V7 configurations loadable. It does not designate a new adopted checkpoint or establish a training-quality improvement.

## Version boundary

Use `V7Config.legacy()` / `for_version("v7")` for a historical starting point and `V7Config.v73()` / `for_version("v7.3")` for the revised design. YAML uses `model.model_version`. Masking modes (`single`, `joint`, `mixed`) are independent of the version.

| Choice | V7 factory | V7.3 factory |
| --- | --- | --- |
| Query state / source / rounds | seed / true sources / K1 | donor / Query as source / shared K4 |
| Coupling map | historical bias | bias-free, zero-preserving forward and inverse |
| Numeric protocol | `legacy` | `standard_softlog_v1`, standard-deviation floor 0.01 |
| Optional Unit source mask | historical Cell-source rows | rows with a true visible Cell |
| Unit initialization | historical construction order | initialize the final installed branch |
| Additional Unit layers | 0 by default | 0 by default |

Historical V7 donor/K4/K8 checkpoints retain their recorded values; the factory defaults do not rewrite them. Missing `model_version` means V7, with legacy meanings for absent new fields. The unqualified `V7Config()` constructor is frozen to the `af7e5f51` V7 defaults and agrees with `legacy()`; its original positional arguments and standalone codec/episode defaults are preserved. Direct `V7Config(model_version="v7.3")` agrees with `v73()`. The temporary mixed-default constructor was removed. A version identifies the starting protocol, while explicit experimental overrides remain serialized.

The source digest still covers all `tabu_lab` Python files. It is deliberately not relaxed: old native strict continuation must keep the original complete source/config/runtime snapshot. Adding files may correctly invalidate its old digest. The current runner is not a promise that every historical optimizer format can be resumed.

New native checkpoints store the version in both ModelSpec and top-level metadata and reject inconsistent identities. Strict resume keeps model/data/masking/optimizer/RNG/source/runtime requirements. Changing versions is a weights-only initialization with a new experiment identity. Cross-version initialization preserves parent tensor dimensions, Unit depth and sharing; nonshared models also preserve their round count. Explicit overrides then apply. Incompatible bias tensors are rejected by strict loading, rather than silently removed. The dry-run migration receipt lists configuration changes, all planned inherited state keys, an empty model-reinitialization list for this strict path, and fresh optimizer/RNG/sampler status. No partial weight migration is silently supplied.

## Changes and resolved audit findings

- Restored mainline boolean and optimizer validation that had been absent from the research copy. Invalid `coupling_scale`, bias options and optimizer parameters fail before execution. V6 warm-start signature validation still happens before model construction/RNG consumption.
- Separated optional Unit source eligibility from the Cell Query-as-source mask. V7 retains historical behavior; V7.3 excludes Query-only rows from Unit sources while allowing them to receive updates. Versioned initialization preserves old weights/RNG and applies the documented initialization to the final V7.3 Unit branch.
- Carried the revised numeric/Null and bias-free coupling implementation into the review branch. Softlog inverse handles finite raw answers even when the intermediate displacement alone would overflow. Historical numeric protocols remain explicit.
- Added nominal/ordinal fixed-candidate probabilities and log probabilities to the general codec and single/joint evaluation interfaces, with candidate ordering, temperature, complete Query counts, answer-code coverage and log loss. Tiny positive temperatures are handled without an all-negative-infinity softmax. Missing answer codes do not shrink the denominator. An unrepresentable true-class score is an explicit numerical failure.
- Kept the analytical LL solve, true support membership, synchronous per-round Query writeback and full BPTT. Optional balanced visible reconstruction is explicitly configured and is in-sample reconstruction, not held-out evidence.
- Preserved row-inducing configuration in model serialization and generated inference dependencies. The native adapter follows the complete ModelSpec, including explicit `query_source=False`. A checkpoint with the known old external architecture and a missing field is rejected unless the caller explicitly selects its overlay and the recorded frozen-source identity is recognized. Recorded and effective Query-source settings are separate in its receipt. The standalone package has the same model/adapter computations and verifies checkpoint version metadata.
- Only V7.3 enables the additional attention-content guard and empty-source backward protection; shared V5/V6/V7 operator defaults retain their original computation and failure boundary. Checked active attention content scores for positive and negative overflow. Deleted sources remain algebraically deleted. Empty-source batches remove unused Query content before the dot product, avoiding nonfinite backward intermediates while preserving the local FFN and other batches.

The design's optional learned readout projection remains a future variant; the implemented V7/V7.3 readout uses the identity projection. The design document was corrected to say so. This review does not claim implementation of every exploratory appendix.

## Evidence

The local environment was the existing Python 3.11 / Torch 2.13 environment, with pandas and its dependencies supplied by an isolated temporary import overlay. No training-host environment was changed. CPU/FP64 and local Apple MPS/FP32 paths were exercised; CUDA was not exercised in this update.

| Acceptance | Current result and limit |
| --- | --- |
| Historical prediction replay | Pending the final real-checkpoint/frozen-source comparison; synthetic legacy loading is a separate check |
| Historical next-update continuation in the original runtime | **Not performed.** No claim of equivalent next weights, optimizer or RNG for an old training experiment |
| V7.3 creation/save/reload and package agreement | Core and package final regressions pending after the boundary corrections |
| Config and current-source initialization/continuation | 229 V7 tests plus 50 masking/fit tests passed; these do not substitute for original-runtime continuation |
| Shared operator boundary | 69 Unit-contract and shared-backbone tests passed |
| Historical initialization | Unit depth 0/1/2 × shared/nonshared: exact initialized weights and RNG against the frozen V7 construction order |
| Generated source and full package snapshot | Pending immutable-revision generation and final source snapshot |
| Design document | Updated boundary text; final compiler check pending |

The earlier 1737-test core / 32-test package result preceded these boundary corrections and is not the final acceptance count. The evidence above distinguishes actual checked cases rather than using one “checkpoint compatible” label.

The first unrestricted full-suite attempt hit a native XGBoost SIGSEGV in `tests/contract/test_query_row_classical_icl.py`. The identical isolated test also crashed on the untouched `af7e5f51` mainline. A process-only `OMP_NUM_THREADS=1` setting made both tests in that file pass; the full regression uses that setting. Multiple loaded OpenMP runtimes are a possible cause, not an established diagnosis. This is recorded separately from model test assertions.

The first completed full run then exposed three MPS tests inheriting a process-wide strict-determinism setting from earlier tests. Those new tests now follow the existing MPS checks: select the supported algorithm mode for the check and restore the incoming mode in `finally`. No product-side determinism setting was changed. The MPS result verifies finite updates and interface behavior, not bitwise deterministic MPS training.

Three historical experiment-wrapper test files were not added to this branch because their experiment scripts are outside this integration. The original research tests and scripts remain in the research checkout. Existing mainline tests were retained.

The meaningful checks include strict resume, model/version mismatch rejection, legacy forward comparison, masked Query/Null Unit roles, optional Unit initialization, row-slot serialization, zero preservation after optimization, CPU/MPS joint execution, probabilities and candidate mapping, missing-answer accounting, active-score overflow, empty-source finite gradients, and generated-package loading. Passing these checks does not establish unseen-table generalization, calibration or a better training recipe.

Reproduction from this checkout with the development and inference extras installed:

```sh
OMP_NUM_THREADS=1 python -m pytest -q tests
python -m pytest -q tests/unit/restoration_v7 tests/unit/restoration tests/unit/restoration_v6 tests/unit/test_coupling.py
```

In the enclosing TabU workspace, the package is tested with its `src` and this checkout's `src` on `PYTHONPATH`:

```sh
python -m pytest -q evaluation/tabarena/package/tests
```

## Entry points and remaining boundary

- [Configuration, continuation and probability interface](../tutorials/v7-restoration-modes.md)
- [V7 example](../../examples/v7-restoration/mixed.yaml)
- [V7.3 example](../../examples/v7-restoration/v73.yaml)

The examples are bounded synthetic implementation fixtures. Historical training/evaluation results retain their old protocol identities. This local change has not been pushed, merged, deployed to training hosts, or used to replace an adopted checkpoint. A V7.3 research run needs its own explicit parent or scratch choice, budget and evaluation question.
