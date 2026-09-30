# V6 consolidation validation — 2026-09-30

Base: remote `main` at `53793317ab586547382ae49e8c328bee728bee29` (V5.5 PR #20).
All implementation work was isolated from the active research checkout and V7 work.

## Implementation and packaging

- Full `tests/unit`: **1290 passed**, 108.03 seconds, using the existing local
  `tabu-lab/.venv` and the consolidation worktree's `PYTHONPATH=src`.
  Runtime: Python 3.11.14, Torch 2.13.0, arm64, CPU tests.
- New V6 tests: **21 passed** after the final internal symlink-chain correction.
  Coverage includes numeric/nominal/ordinal targets, both supervision modes,
  missing cells, Query-truth isolation, finite backward, V5.5 state-dict
  compatibility, exact original Query scorer, target divide-by-two decoding,
  equal row/column joint-loss weighting, invalid modes, prepared mutation,
  byte-exact reconstruction, tampering rejection, path containment and links.
- Scoped Ruff passed for the maintained V6 module, its tests and `archive.py`.
  Historical objects are intentionally unchanged and not restyled.
- `uv build --wheel` succeeded. Wheel inspection confirmed all three V6 modules:
  `__init__.py`, `model.py`, `training.py`. The large research archive is Git-only.
- The maintained model is copied from the later two-mode V6 experiment snapshot.
  The scorer has only lint-equivalent formatting/conditional simplification;
  historical copies retain their original bytes.

These are local CPU implementation and packaging checks, not new training or
accelerator qualification, and do not issue new accuracy/retention claims.

## Archive integrity and completeness

- **26,972 archived file references**, from local preparations plus three hosts.
- **9,163 unique content objects**, **523,519,410 bytes** before Git compression.
- `archive.py verify`: every archived entry's size and SHA-256 matched.
- An additional check read all **9,163 staged Git blobs** through `git cat-file`:
  every SHA-256 matched its original manifest. `.gitattributes` disables line-ending
  normalization for objects, preserving this contract across checkout platforms.
- Credential-pattern scanning found no token/key/password literals matching the
  reviewed patterns. This was a scoped scan, not a claim of universal detection.
- AppleDouble `._*` files, caches, W&B storage and host-operation folders are
  excluded. Per-host models/configurations are not collapsed into one lineage.
- Single-table split/bank/queue/manifest files and the interrupted-run recovery
  budget are archived, including the large fixed fit bank.
- OpenML12 and joint618 predecessors are archived. DGX2's joint618 source-link
  ancestor (`tabu-v55-ordinal100-continuous-dgx2-20260926`) is also included.
- Raw row data, binary checkpoints, large predictions and full update traces
  remain external, explicitly identified in manifests. Checkpoint content hashes
  come from the corresponding archived experiment receipts, not a newly claimed
  all-checkpoint readback. No data or checkpoint binary was moved or overwritten.

New code and navigation pass whitespace checks. Original objects retain historic
whitespace because changing it would invalidate their provenance hashes.

## Independent review

A separate non-author reviewer independently ran the 53 V5.5 compatibility tests,
the V6 supervision/leakage checks and archive-tool tests. The reviewer identified
the missing per-table metadata, earlier source dependencies, symlink chains and
AppleDouble noise; those findings were corrected before submission.
Final independent review: **PASS, no remaining merge blockers**. The reviewer
reconstructed Mini's 7,705 files / 3 internal links and DGX2's 8,063 files / 2
internal links into fresh temporary directories and checked every copied file's
SHA-256. The Mini two-level source chain and DGX2 joint618 ancestor source link
both resolved. The final 21 V6/archive tests passed independently (1.98 seconds).
The staged scope contained no V7 files and no modifications to existing model code.

Reviewed manifest SHA-256:

| Origin | SHA-256 |
|---|---|
| DGX2 | `7b5939e547dd0a4c9aae98133a342f06d7d62c69ccd0dc558844aeacdd257529` |
| Dustin Studio | `3a305c323db54bd918f60b557592c3893b84b5c6cf4163fa0b3415e03f3b04fa` |
| Gongqian Mini | `a20d5b3467b722e8fef38f728c263e2c58d52a4d34f188ef92f09543ba059d4a` |
| Local | `b3a8681cafa8fef82eb5c689fa2f8354961c677ccf4baf4efbcf512b461f0f1c` |
