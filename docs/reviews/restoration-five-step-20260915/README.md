# Five-step restoration implementation review

Scope: independent table-restoration modules, their tests, the additive CLI entry,
and implementation documentation. Existing TAR formulas, identities, checkpoints,
registries, experiment histories and factory defaults are unchanged.

This report and its [machine-generated check](implementation-check.json) record
local implementation verification, not a formal Gate 1 receipt, fit result,
benchmark, accepted capability claim, or maturity promotion.

## Independent review findings and corrections

The initial independent non-author source review rejected two issues. Both were
reproduced and fixed before publication:

1. **P1, incomplete training targets.** A caller could remove an unscorable Query
   from a request and bypass whole-episode missing-code rejection. Training
   preflight now requires the complete set of original observed addresses;
   arbitrary inference request subsets remain valid. Regression:
   `test_training_cannot_drop_unscorable_query_but_inference_may_request_subset`.
2. **P2, tiny source mass.** Squaring a small FP32 presence projection could
   underflow, deleting a source whose large content logit should give it meaningful
   attention mass. Presence now uses scaled FP64 log-mass, with exact-zero source
   deletion shared by direct attention and inducing collect. Regression:
   `test_tiny_nonzero_presence_survives_content_logits_and_matches_fp64`.

Retaining the default ordinal code/rank lift under alternative nominal maps is an
explicit design-open boundary, not an accidentally mixed answer width.

Final independent re-review: **clean within the reviewed CPU source-correctness
scope**, both findings closed, no new blockers. The reviewer independently ran 97
tests excluding CLI, optimizer-continuation and default-configuration smoke, and
compared multihead OMAB outputs/gradients with a rational FP64 reference. Full-target
request reordering preserved outputs, loss and parameter gradients. Exact-zero
source deletion and tiny nonzero projected-presence derivatives also passed.
This is not merge approval or a capability assessment.

Final integration checks on Python 3.11 / CPU: **102 restoration tests passed;
644 full-repository tests passed, 9 skipped**. Skips are existing optional
scikit-learn/XGBoost checks; two expected observer-degradation warnings remain.
Scoped Ruff and sdist/wheel builds passed. The lockfile environment uses PyTorch
2.13.0. An independent wheel installation outside the checkout resolved PyTorch
2.14.0 and passed the same eight verifier probes, recorded in
[wheel-install-check.json](wheel-install-check.json). This is a separate unlocked
installation smoke, not a replacement for the frozen dependency environment.

## Reproducible verification

```bash
uv sync --frozen --extra dev
uv run ruff check src/tabu_lab/models/restoration tests/unit/restoration src/tabu_lab/cli.py
uv run pytest -q tests/unit/restoration
OMP_NUM_THREADS=1 uv run pytest -q
uv run tabu-lab restoration verify --output path/to/new-check.json
uv build --quiet
```

The checked-in JSON was generated with:

```bash
uv run tabu-lab restoration verify --output docs/reviews/restoration-five-step-20260915/implementation-check.json
```

Re-running uses a new output path; the CLI refuses to overwrite a completed check.
The JSON binds the actual restoration source bytes. Source, command, environment,
and check scope must all be inspected when comparing results.

The CPU reference tests include all 16 reduced backbone/nominal-map/readout
combinations, default 256-slot forward/backward structural smoke, independent
normal equations, finite differences, exact model/AdamW checkpoint continuation,
Query/Null insertion, row/column permutation, full-target preflight, actual damaged
support statistics, and retained/Query/Null/corrupted loss reports.

Known limits: no training campaign, accelerator qualification, production experiment
runner, large-table resource bound, dual LL optimization, or unseen-data evaluation.
Alternative ordinal lifts and damage sampling/curricula remain design-open.
