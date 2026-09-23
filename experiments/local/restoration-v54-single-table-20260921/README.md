# V5.4 single-table fit panel (2026-09-21)

This is the first executable V5.4 composition-fit panel. The original CPU
launch was superseded after the host configuration was corrected. The active
MPS/FP32 rerun is a fresh execution identity on commit `9ec97ec`.

Each registered table has three **independent** replicates. Each replicate has
exactly 16,384 optimizer updates, so the initial fit budget is 49,152 updates
per table. The three budgets are separate model/mask/order seed streams; they
are not one 49,152-update continuation. The training episode is the current
V5.4 default, `supervised_row`, with fraction 0.25. `random_cell` remains the
first alternative and is not mixed into this baseline.

The five first-wave tables are deliberately heterogeneous:

- `scm_mixed_v1_001`: numeric-only, width 12
- `scm_mixed_v1_017`: mixed, width 8
- `scm_mixed_v1_040`: mixed, width 12
- `discoscm_076`: mixed, width 32
- `discoscm_095`: mixed, width 32 with a large nominal component

The table bytes are copied from the frozen `tar-diverse-120-fit` corpus while
retaining their SHA-256 identities. The train split is used for fitting and
`final_test` is evaluated only from a completed checkpoint. A fit receipt is
not a generalization claim.

Initial host assignment is one table per host: `dgx2` → `discoscm_095`,
`deepthought` → `scm_mixed_v1_001`, `gongqian-mini` → `scm_mixed_v1_017`,
`zichao-mini` → `scm_mixed_v1_040`, and `dustinstudio` → `discoscm_076`.
DGX2/DeepThought use CUDA FP64. The three Mac assignments use MPS FP32 with
CPU fallback disabled. Backend and dtype are part of the run identity; metrics
from CUDA FP64 and MPS FP32 are reported per host and are not silently pooled
as a bitwise-equivalent trajectory.

Before a run, validate a manifest with `curriculum-v54 plan` and `preflight`.
Every run gets a new output directory; no historical checkpoint is resumed.
After a terminal receipt, evaluate `final_test` in a separate output directory.

The old `queue_replicates.sh` and its CPU outputs are historical invalid-device
evidence. The active `queue_replicates_mps.sh` helper advances only after a
preceding MPS/FP32 `terminal.json` says `outcome=completed`; it never retries a
failed or wall-exhausted replicate.
