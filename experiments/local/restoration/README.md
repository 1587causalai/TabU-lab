# Restoration experiments

W&B project: <https://wandb.ai/zj3712/restoration>.
Local receipts, source identities and checkpoints remain the primary record.
The tracker mirrors aggregate metrics and progress; table contents and model
checkpoints are not uploaded by the observer.

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
