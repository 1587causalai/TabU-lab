# Public `main` pre-push review — 2026-08-31

## Decision

`APPROVE`

The independently reviewed candidate is
`ee37b3f867994bb479d924e0db1460372210575e`. It is approved for a non-force push
to the public `main` branch. No P0 or P1 findings remain, and the P2/P3 quality
findings from the first pass were repaired and independently re-reviewed.

This review does not promote any `local_unissued` training or evaluation artifact
to formal evidence.

## Review identity

- Reviewer: `mini-reviewer`, independent of the developer.
- Runtime: Hermes on `gongqian-mini`.
- Invocation: direct, read-only Hermes one-shot against an isolated Git-bundle
  clone; no wrapper or public write path was used.
- Human approval: Gong explicitly approved the push on 2026-08-31.

## Frozen review inputs

### Full change range

- Public base: `347e987ee66f0b681f3061aeea17313fc24a0b18`.
- First candidate: `6667906ab7e762f9dfbed918e8c5552540cdb3d1`.
- Range: 41 commits, 185 files, 23,798 insertions and 190 deletions.
- Bundle SHA-256:
  `b2a2f75d2f7041a32f7978bfdc6aa6e9c8ee148504c33530d82ed8252a4de9da`.

### Finding-repair increment

- Base: `6667906ab7e762f9dfbed918e8c5552540cdb3d1`.
- Candidate: `ee37b3f867994bb479d924e0db1460372210575e`.
- Range: one commit, 13 files, 140 insertions and 133 deletions. Most changed
  lines are deterministic `__all__` ordering.
- Bundle SHA-256:
  `7c560d7c5411e3d19b82ff00ec405b5fff93474a45f15f046063a1730345ea79`.

Both bundle hashes were independently recomputed, both bundles passed
`git bundle verify`, and both candidate/base identities matched the review
packets.

## Findings and disposition

### P0 / P1

No findings.

### P2: optional scikit-learn tests failed in the default dev environment

The first review found six failures because three new test modules exercised
optional real-data paths without a skip guard while `scikit-learn` was not a
declared dev dependency. The failures were explicit `ModuleNotFoundError` /
`RuntimeError`, not silent incorrect results.

Disposition: fixed in `ee37b3f`. The affected modules now use
`pytest.importorskip("sklearn")`, matching the existing optional-dependency
convention. The incremental reviewer verified that every test in the coarsest
module-level skip depends on `load_real_dataset` and therefore loses no runnable
coverage when scikit-learn is absent.

### P3: 12 candidate-only Ruff findings

The first review reproduced six `RUF022`, one `RUF059`, one `RUF034`, and four
`RUF043` findings against a clean Ruff baseline.

Disposition: fixed in `ee37b3f`. The incremental reviewer checked all six
reordered `__all__` lists by AST and found identical multisets with no additions,
removals, or duplicates. The remaining edits are behavior-neutral unused-value,
constant-branch, and regex-literal corrections. Ruff is clean.

### Accepted informational limitations

- `scripts/build_model_source_manifest.py --check` cannot access the private
  model-factory source from a public clone. Its checked-in fallback manifest was
  available, and the three public copies were byte-identical.
- The dgx2 evaluation report cites host-local artifact paths. It consistently
  labels those artifacts `local_unissued`; public readers cannot independently
  open the checkpoint files from the repository alone.

## Independent verification

The reviewer executed or checked the following:

- Clean checkout, exact base/candidate identities, complete Git history, diff
  statistics, and whitespace checks.
- Full test suite with scikit-learn present: `394 passed, 2 skipped`; the two
  skips were optional XGBoost paths.
- Simulated scikit-learn absence after the repair: `387 passed, 6 skipped`, with
  no collection error.
- Seven focused contract/unit files before the repair: `65 passed`.
- `ruff check src/ scripts/ tests/`: the base was clean, the first candidate had
  exactly 12 findings, and the repaired candidate was clean.
- `scripts/build_evolution_impact_reports.py --check`: passed.
- `tabu-lab program validate`: `status=valid`, 38 nodes, 5 edges, and repository
  hash
  `64336ec9dc4c5ba8021964f09765cd8c2de7934b4aa0deacfcd2aa05954760af`.
  The hash was unchanged by the repair commit.
- Exact-resume probe: uninterrupted and resumed checkpoints were byte-identical.
- Warm-start projection/compatibility acceptance and rejection paths,
  non-overwriting run/checkpoint/evaluation behavior, frozen-evidence gate, and
  capped v3.1 episode-shape behavior.
- No secret/private-key patterns, conflict markers, unsafe state deserialization,
  or unbounded subprocess execution were found in the reviewed range.

The developer independently re-ran the repaired default environment and obtained
`387 passed, 6 skipped`, a clean Ruff result, and the same valid repository hash.

## Review limitations

- The reviewer did not re-run CUDA/MPS, long-duration training, or dgx2-hosted
  checkpoint evaluation; those artifacts were checked only through their
  identities, receipts, and declared hashes.
- XGBoost was absent from the incremental review environment.
- The 23,798-line full range received full diff/stat and automated coverage plus
  focused source inspection; the largest modules were sampled rather than
  manually reviewed line by line.
- This receipt is a documentation-only commit created after the reviewed
  candidate. It records the review and does not alter executable code, specs,
  snapshots, or evidence status.
