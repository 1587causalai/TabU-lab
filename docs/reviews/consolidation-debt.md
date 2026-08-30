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
- **P3 — Link-4 authority checker and report hygiene:** **resolved in
  implementation; promotion remains intentionally blocked.**
  `scripts/freeze_eval_data_authority.py check --output-root ...` now exists,
  validates the checked-in freeze schema and retained source/output byte
  identities, and fails closed on unknown keys, unsafe paths, or hash/size drift.
  Historical reports now use logical execution/artifact ids and the checked-in
  `docs/reports/local-artifact-index.json`; machine-local paths are omitted and
  trailing whitespace is normalized. The underlying local artifacts remain
  unbundled and are not treated as formal authority or public evidence.

The follow-up did not run a private freeze, checkpoint, or scientific evaluation.

## PR5 — typed YAML mathematics projection

Reviewed commit: `2672034dfc6b2854b9522e2860fd3ace759fc5de`.

- **P1 — ModelSpec identity policy:** **resolved in implementation.**
  `model_spec_identity_payload` preserves the legacy serialized payload when an
  optional `mathematics` projection is absent, while binding populated
  mathematics into model, builder, and catalog identity.
- **P2 — TeX label collisions:** **resolved in implementation.** Equation ids use
  a reversible UTF-8 hexadecimal label encoding, with a regression covering
  distinct `a_b` and `a-b` ids.
- **P2 — domain semantics:** **resolved in implementation.** Notation `domain` is
  explicitly raw LaTeX and is rendered once in inline math delimiters; prose
  remains escaped.

## PR6 — bounded catalog projection

Reviewed commit: `7733b8d`.

- **P1 — wrapper/payload identity:** **resolved in implementation.** A model
  catalog entry now requires wrapper `object_id` and `version` to match payload
  `contract_id` and `contract_version`; the index hash continues to bind the
  validated wrapper and payload together.
- **P2 — empty catalog policy:** **resolved in implementation.** Both catalog
  source discovery and the typed catalog index reject an empty model set.

## PR7 — synthetic pretraining implementation history

Reviewed commit: `45a9c815119b67a73d5efef42dba24295c25d0f9`.

- **P1 — checkpoint identity:** **resolved in implementation.** Loading now
  requires both the adjacent identity sidecar and embedded safetensors identity,
  requires their JSON payloads to match, and validates the bound model identity
  before loading tensor state.
- **P2 — source hash scope:** **resolved in implementation.** Promotion source
  identity now covers package code, ModelSpecs, the invoked runner, transfer YAML,
  synthetic-world schemas, `pyproject.toml`, and `uv.lock`.
- **P3 — imported history hygiene:** **resolved in report projection.** Historical
  reports now reference content-addressed logical artifact ids through the
  portable local-artifact index, omit machine-local paths, and contain no
  trailing whitespace. One truncated historical digest was corrected from the
  preserved artifact bytes. No local run, checkpoint, cache, or private dataset
  was copied into Git.

Only entries not explicitly marked resolved remain deferred.
