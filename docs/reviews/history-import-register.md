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
