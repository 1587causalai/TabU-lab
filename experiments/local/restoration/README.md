# Restoration experiments

W&B project: <https://wandb.ai/zj3712/restoration>.
Local receipts, source identities and checkpoints remain the primary record.
The tracker mirrors aggregate metrics and progress; table contents and model
checkpoints are not uploaded by the observer.

## Current recipe: global 2.5% masks with threshold-4 numeric tail protection

Use [old120 Small-128 threshold-4 tail protection](old120-small128-tail-guard4/README.md).
The current threshold is 4 half-IQR. Protected values stay visible and remain
full-cell loss targets; the global Query budget stays at 2.5%.
The threshold-8 run stopped at update 1,004 with a saved checkpoint, at the
owner's request. Its configuration and results remain historical records.
The threshold-4 recipe is prepared but has not been launched.

## Previous recipe: global 2.5% masks

Use [old120 Small-128 global masks](old120-small128-global-mask/README.md).
This replaces equal per-column masking with a table-wide budget and adds
complete-round loss distributions and equal-table fixed-evaluation summaries.
The earlier high-mask run below was stopped with its checkpoint preserved;
its entries remain historical records.

## Old120 Small-128, historical TAR budget

Recipe: [`old120-small128-tar-budget/preregistration.yaml`](old120-small128-tar-budget/preregistration.yaml).
The [budget reference](old120-small128-tar-budget/budget-reference.json) records
hashes and durations of the five historical TAR segments. Their total was
26,823.8711 seconds, with 768 rounds / 92,160 updates. The original 14,400-second
limit applied separately to each invocation.

This restoration run starts from seed 1729 and permits at most 26,824 cumulative
seconds or 768 rounds, whichever occurs first, including preparation and
evaluation. Resume consumes the same cumulative budget. Training stops with
300 seconds reserved for evaluation and saving; an incomplete final evaluation
is recorded explicitly. Evaluation uses eight fixed masks per table initially,
every 32 rounds, and at termination. The original 52 reserved rows are not scored.

The model is Small-128 (3 layers, 8 heads, FFN 256), with restoration's 256
inducing slots, all-column Query masking and FP64 execution. Historical TAR
used 128 inducing slots, target-column masking and FP32 parameters with its
`reference_fp64` backend. Time and table exposures are comparable budget axes;
FLOPs, objectives and losses are not equivalent.

Before training, run the bounded CUDA continuation check and the largest-table
update on the same committed source and runtime:

```bash
uv run tabu-lab restoration verify --device cuda:0 --output runs/restoration/device-check.json
uv run tabu-lab restoration joint-fit-preflight \
  --preregistration experiments/local/restoration/old120-small128-tar-budget/preregistration.yaml \
  --corpus experiments/local/tar-diverse-120-fit/corpus \
  --device cuda:0 --output runs/restoration/largest-table-check.json
```

```bash
export TABU_LAB_OBSERVER=wandb
export TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE=1
export WANDB_MODE=online
export WANDB_ENTITY=zj3712
export WANDB_PROJECT=restoration
export WANDB_RUN_ID=old120-small128-tar-budget-20260916
export WANDB_NAME=old120-small128-tar-budget-20260916
export WANDB_RUN_GROUP=old120-small128
uv run tabu-lab restoration joint-fit \
  --preregistration experiments/local/restoration/old120-small128-tar-budget/preregistration.yaml \
  --corpus experiments/local/tar-diverse-120-fit/corpus \
  --device cuda:0 --output-root runs/restoration/old120-small128-tar-budget-20260916 \
  --execute
```

W&B authentication uses the existing SDK login or `WANDB_API_KEY` supplied by
the environment. Do not put credentials in the recipe or command record. An
unset observer keeps execution local. Omitting `--execute` validates the plan
without starting training or a tracker run. Use a new output directory for each
resume attempt, with `--resume-checkpoint` pointing at its previous validated
boundary and the same `WANDB_RUN_ID` for a continuous mirror. Source, corpus and
configuration identities must match exactly; historical pilot checkpoints from
an older source cannot be resumed here.

This remains `local_unissued` fitting evidence. No heldout capability or maturity
promotion follows from the live dashboard.

## Launch verification, 2026-09-16

Execution source: `af0ebcaac4c53115509e48acd01f6591c9c719e1`, independently reviewed
on branch `codex/restoration-wandb-budget-20260916`. Local checks: 828 passed,
9 optional-dependency skips; changed Python files pass Ruff. Dedicated tests
cover mid-second-round continuation with identical model/optimizer/RNG and
update trace, failed-optimizer rollback, cumulative budgets, partial evaluations,
and completion of an interrupted final evaluation without extra updates.

The same source passed
[CUDA forward/backward and exact continuation](old120-small128-tar-budget/qualification/device-check.json)
and the
[204 x 32 largest-table update](old120-small128-tar-budget/qualification/largest-table-check.json).
These checks qualify this configuration's execution; they do not measure fit or
generalization. Runtime output and checkpoints remain in the isolated experiment
directory on the training machine.

Live run:
<https://wandb.ai/zj3712/restoration/runs/old120-small128-tar-budget-20260916>.
The dashboard is an observation mirror; completion requires the terminal local
receipt. The launch snapshot is recorded separately from the eventual result.
