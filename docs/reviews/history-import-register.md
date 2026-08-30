# Historical implementation import register

This register tracks implementation copied from the preserved dirty donor while
`origin/main` remains the only active baseline.

Policy for the consolidation phase:

1. copy selected donor files byte-for-byte into `history/imports/`;
2. record source base, scope, exclusions, and review findings;
3. do not activate, reconcile, or repair imported code in the same PR;
4. after all important history is collected on `main`, govern promotion into canonical
   package paths in separate PRs.

## Batch PR12 — model family and verification history

- Main base: `36ca11a50cc695589eb88fd361003061d859274f`
- Donor Git base: `3502fdd80539f2a8b9703cc4e4546fd01f3826ce`
- Donor state at audit: 417 default porcelain entries; 462 file-expanded entries
- Imported scope: 47 selected donor files plus `MIGRATION_PROVENANCE.md`
- Snapshot root: `history/imports/donor-3502fdd/model-verification/`
- Activation status: `staged_history_only`

### Registered integration findings

- **HIST-001 — current-schema incompatibility:** `registered_unresolved`. Historical
  model specs predate the current required upstream source-tree identity fields. Putting
  them directly under canonical `specs/models/` would make current registry validation
  fail.
- **HIST-002 — active-builder conflict:** `registered_unresolved`. The donor also has
  broad model-family edits to existing builder, model export, registry, and source
  manifest files. Those edits overlap newer identity governance already on `main` and
  are intentionally not reconciled in this import PR.
- **HIST-003 — cross-batch dependency:** `registered_unresolved`. The historical
  verification chain refers to evaluation-foundry and fit-first modules assigned to a
  later history-import batch. This snapshot is not claimed runnable in isolation.
- **HIST-004 — provenance/license boundary:** `registered_unresolved`. The included
  `MIGRATION_PROVENANCE.md` records metadata-only and clean-room boundaries for the
  model-factory and legacy sources. Importing this snapshot does not expand those
  permissions or migrate capability claims.

Independent-review findings are appended to this register without fixing them during
the consolidation phase.

## Batch PR13 — evaluation, fit, and receipt history

- Main base: `6fcbda7015e31b6bf92928687ee8f7edfeaa02f4`
- Donor Git base: `3502fdd80539f2a8b9703cc4e4546fd01f3826ce`
- Imported scope: 138 selected donor files
- Snapshot root: `history/imports/donor-3502fdd/evaluation-fit/`
- Activation status: `staged_history_only`

### Registered integration findings

- **HIST-005 — model-family dependency:** `registered_unresolved`. Fit-first and
  verification records refer to model families and builders preserved in PR12 but not
  active on current `main`.
- **HIST-006 — catalog/evidence contract overlap:** `registered_unresolved`. Historical
  catalog schemas, formal-receipt scaffolding, and public projection code overlap newer
  catalog and evidence-safety contracts already on `main`; no canonical version is
  selected in this import.
- **HIST-007 — historical data authority:** `registered_unresolved`. Dataset candidates,
  passports, freeze adapters, and diagnostic results remain historical metadata or
  `local_unissued` evidence. Import does not establish retained bytes, replayability, or
  formal dataset authority.
- **HIST-008 — site projection dependency:** `registered_unresolved`.
  `build_site_manifest.py` refers to public-site history that is outside this batch.
  The script is preserved but not activated or claimed runnable.
- **HIST-009 — execution compatibility:** `registered_unresolved`. Historical tests and
  runners were authored against the donor's combined dirty tree, including overlapping
  edits to existing registry, builder, CLI, catalog, and evidence modules. They are not
  used as current-main test evidence.
- **HIST-010 — preserved donor whitespace:** `registered_unresolved`. Three historical
  F0 preregistration YAML files contain an extra blank line at EOF, so
  `git diff --check` reports those imported paths. The bytes are intentionally preserved
  during consolidation; formatting is deferred to governance.

Independent-review findings for this batch are appended without repairing imported
code during consolidation.

## Batch PR14 — overlapping active-tree history

- Main base: `116e0975654f39fc07b4b82b7d7fa7e4555d7602`
- Donor Git base: `3502fdd80539f2a8b9703cc4e4546fd01f3826ce`
- Imported scope: donor versions of 41 paths whose active-main bytes differ
- Snapshot root: `history/imports/donor-3502fdd/overlapping-active-tree/`
- Activation status: `staged_conflict_side_only`

### Registered integration findings

- **HIST-011 — same-path semantic conflicts:** `registered_unresolved`. This batch
  preserves the donor side without overwriting active-main files. A later governance
  PR must reconcile each path explicitly rather than treating modification time or
  either tree wholesale as authoritative.
- **HIST-012 — identity-governance versus model breadth:** `registered_unresolved`.
  Current main contains newer checkpoint, source, ModelSpec, and catalog identity
  protections, while the donor versions contain broader model-family builders and
  exports. Whole-file replacement in either direction would discard important work.
- **HIST-013 — experiment implementation overlap:** `registered_unresolved`. Donor
  versions of scale, real-benchmark, response-readout, and full-context comparison
  modules differ from implementations consolidated through PR7–PR9. This import does
  not infer which individual hunks remain necessary.
- **HIST-014 — source/spec projection overlap:** `registered_unresolved`. ModelSpec,
  packaged spec, schema, and source-manifest versions must evolve together. Their donor
  sides are preserved here but not promoted independently.
- **HIST-015 — verification export overlap:** `registered_unresolved`. Active main and
  donor expose different verification surfaces. Later governance must preserve the
  current composability gate while deciding how the historical MVE surface is
  activated.

Independent review checks snapshot integrity and registration only; it does not select
conflict resolutions.

## Consolidation checkpoint after PR14

The file-expanded donor audit classified 275 dirty paths in the core implementation
and evidence roots (`src`, `scripts`, `experiments`, `tests`, `specs`, `schemas`,
`datasets`, `evaluations`, and `verification`):

- 49 paths were already byte-identical to active main;
- 167 donor-only core paths were preserved by PR12 and PR13;
- 18 additional dataset/evaluation/verification paths were preserved by PR13;
- 41 same-path conflicts were preserved by PR14 without overwriting active main.

Together, PR12–PR14 preserve 226 selected donor files in history snapshots, plus the
donor `MIGRATION_PROVENANCE.md`. The active implementations consolidated by PR7–PR11
remain in their canonical paths.

This checkpoint closes **core implementation history collection**, not donor-tree
cleanup and not semantic integration. Governance remains paused.

### Residual non-core donor surface

The same audit still reports 187 dirty paths outside the roots above:

- 157 under `site/`, including 156 generated/public projection paths;
- 23 under `docs/`;
- 2 under `.github/`;
- one each at `pyproject.toml`, `catalog.json`, `ROADMAP.md`, `README.md`, and
  `MIGRATION_PROVENANCE.md`.

The migration provenance file is already preserved by PR12, leaving 186 residual paths
to classify if broader donor closure is required.

- **HIST-016 — public projection authority:** `registered_unresolved`. Generated
  `site/public` output must not be bulk-imported before its canonical source and current
  projection status are identified.
- **HIST-017 — residual docs/workflow/top-level history:** `registered_unresolved`.
  These paths are outside the completed core-code collection. They require a separate
  importance and provenance decision, not automatic migration.

No HIST finding is repaired at this checkpoint. Semantic governance begins only after
the owner accepts the consolidation scope or requests additional history batches.

## Governance station 1 — TabUBase canonical vertical slice

- Main base: `c18f8393c20a8d67d8d0e598cde79e1b9f41b7d3`
- Scope: `Unit semantics → ModelSpec → canonical builder → runtime components → Step 1/2 local verification`
- Contract boundary: existing `tabu.cell.base@0.2.0`; no ModelSpec byte or version change
- Evidence boundary: `local_unissued`; no training or scientific evaluation

This station activates no historical model family or broad MVE runner. It adds a
current-main-compatible component binding for TabUBase, preserves the existing
one-axis composability gate, and gives Step 1/2 results a strict local evidence schema.

- **HIST-002 — active-builder conflict:** `partially_resolved_tabubase_only`. The
  canonical TabUBase builder now proves its runtime composition against the exact
  registered ModelSpec before returning. Historical broad model-family builders remain
  staged and unresolved.
- **HIST-015 — verification export overlap:** `partially_resolved_tabubase_only`. The
  active composability surface is preserved and now consumes the ModelSpec-bound
  composition identity. Historical broad MVE exports remain staged and unresolved.

All other HIST findings retain their previous status.

### Independent review findings for governance station 1

Reviewed commit: `886cae9bcfca94190c0c8e63607ba55dfabd37ec`.

- **GOV1-001 — prediction/model evidence binding:** `resolved_in_follow_up`. Step 2
  now requires each prediction and trace to bind the corresponding model variant, and
  requires both sides to share the episode and input hash.
- **GOV1-002 — terminal label/class agreement:** `resolved_in_follow_up`. Component
  resolution now checks the concrete numeric terminal class instead of trusting its
  mutable semantic label.
- **GOV1-003 — undeclared MAB substitution:** `resolved_in_follow_up`. MAB remains a
  code-only non-O ablation and cannot pass as a ModelSpec-declared component
  substitution.
- **GOV1-004 — lint and DCO hygiene:** `resolved_in_follow_up`. The reported line-length
  violation is corrected; the reviewed commit is superseded by a signed follow-up
  commit before merge.

The independent review reported no P0 or P1 finding. These corrections remain within
the first-station identity boundary and do not add training or scientific evaluation.

## Governance station 2 — component extension contract

- Main base: `ff796162c7874b30556cc9cb8c47c5f17c9ee942`
- Scope: `ComponentSpec → protected registry → typed manifest → resolved modules → identity-bound local verification`
- Compatibility boundary: the default `tabu.cell.base@0.2.0` variant and checkpoint
  identity remain unchanged
- Evidence boundary: experimental extensions are `local_unissued` and are not ModelSpec
  alternatives or accepted claims

This station adds a controlled growth seam for tokenizer, dynamics, and readout roles.
It activates no historical broad model family and does not promote the code-only MAB
ablation. Component source identity covers the direct runtime class; repository-level
source closure remains a separate run/receipt responsibility.

- **HIST-002 — active-builder conflict:** remains `partially_resolved_tabubase_only`.
  TabUBase now has a namespaced component extension boundary, but historical broad
  model-family builders remain staged and unresolved.
- **HIST-015 — verification export overlap:** remains
  `partially_resolved_tabubase_only`. A typed local component-extension verification is
  now active; historical broad MVE exports remain staged and unresolved.

Independent-review findings for this station are registered below after review and are
not repaired silently.

### Independent review findings for governance station 2

Reviewed commit: `3f3dbf7c686a64f1417be154c7c079a633dd4bcc`.

- **GOV2-001 — canonical registry trust root:** `resolved_in_follow_up`. Builder-supplied
  registries must preserve every exact canonical anchor; public registration accepts
  experimental maturity only, and `model_spec_declared` is derived from authoritative
  spec membership rather than an empty experimental-axis list.
- **GOV2-002 — runtime/spec rebinding:** `resolved_in_follow_up`. Inspection now
  re-resolves the attached manifest against its registry and requires every runtime
  module to have the exact registered concrete type and current source identity.
- **GOV2-003 — recursively immutable config:** `resolved_in_follow_up`. Component specs,
  refs, manifests, and composition payloads recursively copy/freeze nested canonical
  values; prediction binding also checks the complete component identity metadata.
- **GOV2-004 — factory dependency identity:** `resolved_in_follow_up`. Factories must be
  import-resolvable module-level functions. Their direct source, bytecode, referenced
  concrete types/constants, and defaults are identity-bound and rechecked before every
  build; helper functions, module objects, callable objects, and nonlocal closures are
  rejected rather than incompletely hashed.
- **GOV2-005 — export lint ordering:** `resolved_in_follow_up`. Public `__all__` exports
  are sorted under the repository Ruff policy.

The independent review reported no P0/P1 finding. The superseding commit receives a
focused independent re-review before merge.
