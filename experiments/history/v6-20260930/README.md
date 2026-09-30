# V6 experiment archive — 2026-09-30

This archive closes the gap between the merged V5.5 implementation and V6
experiments that previously existed only in local preparations or training-host
directories. It preserves source, configurations and reviewable evidence for
future reproduction and data/result inspection. Start with [INDEX.md](INDEX.md).

The four manifests identify independent origins:

- [local](local.json): the research project's `preparations` tree;
- [DGX2](dgx2.json): the CUDA/FP64 experiment tree;
- [Dustin Studio](dustinstudio.json): the MPS/FP32 experiment tree;
- [Gongqian Mini](gongqian-mini.json): its independent MPS/FP32 experiment tree.

All host reads were read-only. No job was launched, resumed or stopped; no
checkpoint/default reference was changed. Original files remain in place.

## What is preserved

Each `archived` manifest entry binds its original relative path to a byte count,
SHA-256 and `objects/` file. Identical content is stored once, while every origin
keeps its own mapping. This preserves host-specific source changes rather than
substituting one host's files for another's.
Original Markdown links retain their historical relative paths. They work in the
reconstructed directory tree; links inside an `objects/<hash>.md` preview may not
resolve until materialization.

The collection covers V6 experiment folders and the OpenML12, joint618 and Puma
predecessors used by their runners: full available Python source snapshots,
launch/train/evaluation scripts, configurations, splits, fixed Query banks,
parent identities, checkpoint selection/rejection history, bounded result receipts,
and the single-table backward repair and recovery budgets. Small files under
`tables/` are retained, not mistaken for raw datasets. Historical scripts include
deployment and observer operations; archiving them is not a recommendation to
execute those scripts blindly.

The initial joint-only V6 model and later `target_only` / `joint_all` model are
both retained under their original experiment paths. Random mixing, nonlinear
encoder and branch-sampling candidates remain historical experiments. The
maintained [V6 module](../../../src/tabu_lab/models/restoration_v6) contains
the two-mode broadcast model; it does not change V5.5 or V7.

## Verify and reconstruct

From a Git checkout of this repository:

```bash
python experiments/history/v6-20260930/archive.py verify
python experiments/history/v6-20260930/archive.py materialize \
  --host dustinstudio --destination /tmp/tabu-v6-dustinstudio-replay
```

The destination must not exist. The tool verifies hashes first, reconstructs
the host's collected tree, restores links to included sibling directories, and
writes `archive-materialization.json` listing external artifacts. It never runs
training, connects to hosts, downloads data, or creates links to external paths.
Restore the entire host tree so sibling source/helper imports stay available.
File contents are preserved; executable permission bits are not restored. Invoke
reviewed scripts through the recorded Python runtime or explicitly through their shell.

The content archive is distributed through Git, not bundled into the model wheel.
The wheel includes the maintained V6 model and scorer.

## External data and checkpoints

Raw row data, model/optimizer checkpoint binaries, full update traces, large
prediction artifacts, caches and W&B files remain in their original stores.
Manifests distinguish `external-checkpoint`, `external-data-or-large-artifact`,
`external-row-data`, `external-full-trace` and `symlink` entries. For external
regular files up to 32 MiB (except checkpoints), the inventory includes SHA-256.
For checkpoints it records the location/size; use the exact checkpoint SHA in
the corresponding archived parent/terminal/evaluation receipt. Files beyond this
collection's scope are not claimed as independently backed up here.

Before replay, obtain those external artifacts at the manifest's original root,
verify their recorded identities, and adapt host-specific absolute paths in a
**new** run configuration. Historical source and JSON are intentionally not
rewritten to replace paths. Source hashes and the original replay evidence would
otherwise be lost. The archive alone is not a self-contained dataset/checkpoint
release and does not promise that a fresh clone can repeat full training without
those external inputs.

For data review, the archived bank/split/manifests determine which rows and tables
were used. Recover raw values from their recorded source and compare hashes;
do not regenerate a split or substitute a newer table with the same name.

## Evidence boundaries

Read the host's checkpoint decisions with its preceding failure/candidate history.
Use a matching checkpoint, fixed bank, target type and endpoint when comparing
scores. Dated progress summaries are preserved as history and may precede later
terminal receipts. These historical results were collected, not rerun or newly
certified as capability claims. [Validation](VALIDATION.md) records the checks
performed for this consolidation.
