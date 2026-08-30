# Consolidation debt register

This register captures findings intentionally deferred under the owner-approved
“merge first, govern after main is consolidated” policy.

## PR8 — consolidated real evaluation stack follow-up

Reviewed commit: `1364761b5b4668658c371ce504c555db4ccfe6a8`; follow-up is
merged from the isolated `codex/pr8-debt-followup` worktree at `1c4299e`.

- **P1 — fine-tune held-out estimand:** **resolved in implementation.** Held-out
  evaluation now constructs one transductive evidence episode and applies the
  bounded chunk size only to the response readout. The receipt binds the
  estimand, readout chunk size, context-row hash, and query-row hash. The
  calibration path uses the same helper.
- **P2 — producer source identity:** **resolved in implementation.** Full-context
  frozen/baseline receipts now require lowercase `git_commit` and
  `source_tree_sha256`, and the strict comparator rejects any mismatch before
  producing a comparison receipt.
- **P3 — Link-4 authority checker and report hygiene:** **checker resolved;
  promotion remains intentionally blocked.**
  `scripts/freeze_eval_data_authority.py check --output-root ...` now exists,
  validates the checked-in freeze schema and retained source/output byte
  identities, and fails closed on unknown keys, unsafe paths, or hash/size drift.
  Historical report trailing whitespace and host-local evidence paths remain
  preserved until portable artifact manifests are introduced; they are not
  treated as formal authority or public evidence.

The follow-up did not run a private freeze, checkpoint, or scientific evaluation.

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

- **P1 — checkpoint identity:** **resolved in implementation.** Loading now
  requires both the adjacent identity sidecar and embedded safetensors identity,
  requires their JSON payloads to match, and validates the bound model identity
  before loading tensor state.
- **P2 — source hash scope:** **resolved in implementation.** Promotion source
  identity now covers package code, ModelSpecs, the invoked runner, transfer YAML,
  synthetic-world schemas, `pyproject.toml`, and `uv.lock`.
- **P3 — imported history hygiene:** historical reports retain trailing whitespace
  and host-local evidence paths. Preserve them for now; later project evidence
  through portable artifact manifests.

Only entries not explicitly marked resolved remain deferred.
