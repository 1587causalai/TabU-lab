# TAR integration: local validation

The integration adds explicit `tabu.tar` construction, Small/Medium/Standard
presets, bounded fitting, optional W&B observation and reproducible source
metadata. It preserves the existing no-argument model default, `program` and
`tabur` commands, and historical pretraining programs. Research navigation now
identifies TAR as the current direction.

This is an implementation validation record, not an independent review, formal
experiment receipt, pretrained model release or accepted capability claim.

## Changes and compatibility

The work is based on `4dcdb2d`. The integration commits are:

- `8aa1272`: separate the TAR research direction from compatibility defaults.
- `78c34d8`: TAR core, immutable builder registration, source-bound ModelSpec,
  read-only design snapshot and named Small component checks.
- `a18ba86`: full-train-pool episodes, resampled masks/codebooks, joint test
  prediction, bounded fitting, optional W&B and Git source-state records.
- `c1b2edf`: generated catalog projections and corresponding regression checks.

All 13 TAR core Python files remain byte-identical to the experimental source.
The bound model-design TeX is unchanged. Source-closure metadata is newly required
by the integrated registry, so new checkpoints have a different source identity.
Older checkpoints still require their archived source and ModelSpec; strict
loading is retained. All 99 existing contract/program files and 20 copied
protocol/data files checked against their sources were preserved.

## Validation

The dedicated worktree environment was created from the checked-in lock:

```bash
uv sync --frozen --extra dev
uv run --no-sync pytest -q
uv run --no-sync python scripts/build_model_source_manifest.py --check --contract tabu.tar
uv run --no-sync python scripts/build_public_catalog.py --check
uv run --no-sync python scripts/verify_site.py
uv build
```

The complete test suite finished with **515 passed, 9 skipped** and two expected
observer-degradation warnings. The skipped tests require optional sklearn or
xgboost dependencies absent from this frozen development environment. Tests cover
TAR operators, serial/batched parity, truth isolation, registry construction,
protected builders, checkpoint recovery, fit protocols, deadlines and retained
legacy CLI/default behavior. This run does not cover GPU timing or training.

The TAR source-closure check, generated catalog check and site verification passed.
The regenerated catalog adds one model entry (89 to 90), with zero accepted claims
and zero formal receipts. No website deployment occurred.

Both wheel and source distribution built. The source distribution contains exact
copies of the TAR design/defaults and frozen fitting inputs. The extracted wheel,
loaded outside the repository, passed real Small verification (721,464 parameters)
including forward, backward, optimizer update and strict checkpoint recovery.

Ruff passed for the changed runtime, tests and scripts. Whole-repository Ruff has
79 inherited findings, primarily in historical imports; comparison against the
base found 80 findings and **no newly introduced findings**. The full repository
lint is therefore not claimed to pass. Local documentation links were checked.

## Remaining review and experiment boundaries

The branch is a candidate for review; neither local nor remote main was changed
by this integration. Existing unrelated main-versus-remote commits need their own
merge scope. No old experiment receipt was reissued and no GPU run was launched.
New fitting attempts record a Git commit and clean/dirty status when available;
source snapshots without Git report `unavailable`. All diagnostic outcomes remain
`local_unissued`, even when the source checkout is clean.

The next experimental gate remains Small-first validation along the six-stage
ladder. These software checks do not establish synthetic fitting success, real
prediction quality, frozen ICL or fine-tuning improvement for the new source.
