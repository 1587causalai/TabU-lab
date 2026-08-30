# Migration provenance

This file records the boundary between three different artifacts observed on
2026-08-27.  It is a provenance receipt, not a claim that their maturity or licenses are
equivalent.

## Frozen observation at migration start

The facts in this section describe the initial 2026-08-27 observation. They are kept
immutable as migration provenance and must not be read as the current branch or
working-tree state; later implementation and review revisions are recorded in Git and
`docs/reports/`.

### TabU-lab

- Repository: this checkout, `tabu-lab`.
- Observed Git `HEAD`: `8e94e2521beb13b3176ee411e274c402dc866306`.
- License in this repository: Apache License 2.0 (`LICENSE`).
- Role: a new, public-facing, contract-first implementation.  The eight packaged
  `ModelSpec` files are descriptive metadata and implementation gates; they are not
  evidence of trained capability.

The working tree was not clean at observation time.  The hash above identifies the
base commit, not the uncommitted implementation in this working tree.

### Model factory

- Observed path: `../latex/model-factory` relative to the parent TabU workspace.
- Role: current owner-authored mathematical/design carriers for seven named folders and
  eight distinct contracts.  `TabU/main.tex` and `TabU/unit-pair.tex` intentionally
  define incompatible Unit choices and therefore have separate contract IDs.
- Git fact: the source workspace still showed the whole `model-factory` directory as
  untracked at observation time.  A scoped local branch was subsequently created in an
  isolated worktree: `codex/tabu-model-factory-contracts`, commit
  `53dbd4b18c8390517a7e248a493471f34a2e1a39`.  That commit contains the model-factory
  carriers only; it was not pushed or merged.  It is a review candidate, not the current
  source-workspace state.
- License fact: no `LICENSE*` or `COPYING*` file was found within the model-factory
  directory at observation time.

TabU-lab does not vendor the TeX bodies or figures.  Each ModelSpec records a readonly
relative source path and the exact SHA-256 of the observed source file.  A changed
source hash requires an explicit contract review; it must not be refreshed as a
mechanical test fix.

### Legacy `tabuf-core`

- Owner-workspace path alias:
  `causal-superintelligence/tabular-foundation-models/tabuf-core`.
- Resolved owner-workspace repository path:
  `projects/two-month-ten-conference-papers/papers/P11-tabular-foundation-models/tabuf-core`.
- Observed Git `HEAD`: `e01ac891087db4ec6def8d16c92ed3e0c1ea45b5`.
- Role: previous TabUF-focused implementation and experiment infrastructure.  Its own
  README describes a `v2-null-o` legacy carrier and an `o_augmented` candidate, not an
  accepted implementation of every model-factory contract.
- Git fact: the legacy working tree was dirty at observation time.  The `HEAD` hash is
  therefore not a content hash for the observed working tree. Its local tracking ref
  reported 71 commits ahead; no fetch was performed, so that count is not a fresh
  remote-state claim.
- License fact: no root `LICENSE*` or `COPYING*` file and no package license declaration
  were found at observation time.

The exact dirty-state and environment hashes plus the rerun `258 passed` behavior
receipt are frozen in
`tests/legacy_parity/fixtures/tabuf_core_2026-08-27.json`. That receipt is historical
donor evidence only and does not migrate any capability claim.

## Clean-room implementation boundary

No legacy source code is copied into TabU-lab by this migration.  Legacy behavior may
be studied and, where independently specified, tested through black-box fixtures or
newly written conformance tests.  A compatible behavior is not evidence of code
provenance or permission to redistribute legacy code.

Apache-2.0 in TabU-lab applies to material actually released from this repository.  It
does not retroactively license the model-factory TeX or legacy `tabuf-core`.  Before any
verbatim text, figure, code, fixture, checkpoint, or data artifact from either source is
published here, its copyright and license must be resolved independently and its
attribution/NOTICE obligations recorded.

## Hash scope

Each ModelSpec pins exactly one entrypoint digest:

- `TabUFL/main.tex` → `tabufl`
- `TabUL/main.tex` → `tabul`
- `TabUF/main.tex` → `tabuf`
- `TabU4Rec/main.tex` → `tabu4rec`
- `TabU4Graph/main.tex` → `tabu4graph`
- `TabU4Do/main.tex` → `tabu4do`
- `TabU/main.tex` → `tabu.unit_row`
- `TabU/unit-pair.tex` → `tabu.unit_pair`

The separate `specs/model-factory-source-manifest.json` recursively hashes every TeX and
graphics include reachable from each entrypoint and produces a canonical semantic-tree
hash. It records README/build context and compiled PDF projection hashes in separate
fields: neither context nor PDF is promoted to semantic authority. The manifest also
records the local scoped Git candidate but does not claim that branch was merged or
pushed.

## Canonical semantic rename (2026-08-28)

The model-factory `TabU` folder now exposes two same-level semantic roots:

- `TabU/unit-as-row.tex` → `tabu.unit_row` (row is Unit)
- `TabU/unit-as-cell.tex` → `tabu.unit_pair` (cell $(U,F)$ is Unit)

The contract ID `tabu.unit_pair` is retained for registry and experiment-lineage
continuity; its display name and current upstream source are now Unit-as-cell. The old
`TabU/main.tex` and `TabU/unit-pair.tex` paths remain compatibility aliases for readers
of historical records. Historical preregistrations, receipts, and experiment IDs are
not rewritten; a new source-manifest/spec revision carries the canonical paths and
hashes above.

The generated catalog/public projection is intentionally not hand-edited during this
rename. The catalog builder now preserves any preregistration whose embedded
ModelSpec hash differs from the current bare alias as a content-qualified historical
contract identity (for example, `tabu.unit_pair@<sha256>`), and points its
`implements` edge to that exact identity. The current bare contract remains the
canonical source entry, while `F0-020` is the append-only contract-alignment
successor. This makes the migration fail closed on accidental hash drift without
rewriting historical bindings; the public projection remains a deterministic index,
not evidence of a capability claim.

Later the same day, Unit-as-cell figures were renamed from `fig-pair-*` to
`fig-cell-*` so the folder surface matches the semantic root. Shared operator
figures `fig-omab.tex` and `fig-oattention.tex` are unchanged. Compatibility
aliases `main.tex` and `unit-pair.tex` remain as path links only; their compile
artifacts are not canonical projections. Historical preregistrations keep their
pinned source hashes; the current ModelSpec and source-manifest revision carry
the new entrypoint digest.
