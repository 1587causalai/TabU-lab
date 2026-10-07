# Historical V7 runner regression coverage

2026-10-08. Latest main already contains all 15 prior V7 product modules, the
shared backbone and coupling primitive. They match the captured canonical
baseline byte for byte. Its dual-stream test fixture includes a newer verified
MPS determinism fix, which is retained.

Three historical runner scripts and their corresponding regression tests were
still absent. This commit adds those sources without datasets, weights, run
outputs, or automatic execution:

- `experiments/v7-openml12-single-table-20260930/run_v7_openml12_single.py`: Query-label isolation and physical-row uniqueness.
- `experiments/local/v7-old120-pilot-20260930/run_v7_old120.py`: absent baseline comparisons, finite numeric output, and paired task construction.
- `experiments/v7-seed-singlepass-fit-20261001/run_comparison.py`: compatible V6 weight reuse and failed-parent rejection.

Their 16 regression cases pass on the latest-main checkout. The scripts are
historical experiment entry points with their recorded conventions; they do not
replace the current V7 CLI or authorize any new training budget.
