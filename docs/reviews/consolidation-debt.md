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

## PR6 — bounded catalog projection

Reviewed commit: `7733b8d`.

- **P1 — wrapper/payload identity:** the catalog wrapper identity is not
  cryptographically cross-bound to the projected payload identity.
- **P2 — empty catalog policy:** the projection currently fails open when the
  bounded catalog contains no models.

## PR7 — synthetic pretraining implementation history

Reviewed commit: `45a9c815119b67a73d5efef42dba24295c25d0f9`.

- **P1 — checkpoint identity:** `load_pretrain_checkpoint` treats the adjacent
  identity sidecar as optional and does not cross-check it with identity embedded
  in safetensors metadata.
- **P2 — source hash scope:** `source_tree_sha256` omits runner, config, schema,
  and dependency-lock inputs used by promotion checks.
- **P3 — imported history hygiene:** historical reports retain trailing whitespace
  and host-local evidence paths. Preserve them for now; later project evidence
  through portable artifact manifests.

All entries remain deferred until the owner closes the consolidation sequence.
