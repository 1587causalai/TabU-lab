# Consolidation debt register

This register captures findings intentionally deferred under the owner-approved
“merge first, govern after main is consolidated” policy.

## PR5 — typed YAML mathematics projection

Reviewed commit: `2672034dfc6b2854b9522e2860fd3ace759fc5de`.

- **P1 — ModelSpec identity policy:** adding optional `mathematics=None` changes
  default serialized shape and may change legacy canonical hashes even when YAML
  bytes are unchanged. Define a stable canonical serialization policy or an
  explicit version migration.
- **P2 — TeX label collisions:** current label normalization maps distinct ids such
  as `a_b` and `a-b` to the same label. Add rendered-label uniqueness or reversible
  encoding.
- **P2 — domain semantics:** decide whether notation `domain` is prose or raw LaTeX,
  then use a matching single-pass renderer and regression fixture.

No item above is repaired in PR5 or PR6. Governance starts after the consolidation
sequence is complete.
