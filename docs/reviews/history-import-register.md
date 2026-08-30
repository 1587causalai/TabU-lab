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
