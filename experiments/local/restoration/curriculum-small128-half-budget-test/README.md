# Restoration Small-128 curriculum — half budget + reserved test

Amendment of the frozen [curriculum-small128](../curriculum-small128/preregistration.yaml)
three-stage recipe. Two changes, everything else identical (seeds, model,
optimizer schedule, mask policies, evaluation cadence, corpora binding):

1. **Halved budgets**: stage update budgets 92,160 / 38,400 / 9,600 →
   46,080 / 19,200 / 4,800 (all still whole cycles) and stage wall ceilings
   26,824 / 28,800 / 7,200 → 13,412 / 14,400 / 3,600 seconds.
2. **Reserved-row test evaluation** (`reserved_test: true`): alongside every
   fixed-mask evaluation point, each table is scored once on a forward-only
   episode over its reserved rows — training context fully visible (the
   deterministic evaluation window 0 for windowed `new9` tables), reserved rows
   visible on predictor columns and queried on the target column, capped at
   256 deterministically selected reserved rows per table
   (`reserved_test_max_rows`). Reserved rows never enter a training episode or
   a gradient. Metrics land in each metrics JSON under `test` (by_state /
   by_type / by_source / by_table / coverage) and mirror to W&B under
   `test_evaluation/*`.

Run (after the CUDA size preflight and resource gate pass):

```text
export TABU_LAB_OBSERVER=wandb TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE=1
export WANDB_MODE=online WANDB_ENTITY=zj3712 WANDB_PROJECT=restoration
export WANDB_RUN_ID=curriculum-small128-half-test-20260916
export WANDB_NAME=curriculum-small128-half-test-20260916
export WANDB_RUN_GROUP=curriculum-small128
tabu-lab restoration curriculum-fit \
  --preregistration experiments/local/restoration/curriculum-small128-half-budget-test/preregistration.yaml \
  --corpus-root /path/to/frozen-curriculum-corpora \
  --device cuda:0 --output-root /path/to/new-attempt \
  --execute
```
