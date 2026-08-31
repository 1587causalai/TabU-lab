# dgx2 Evolution Grow pilot — 2026-08-31

Status: `local_unissued`

This report records a bounded engineering and synthetic learning-dynamics run.
It is not formal evidence, a real-data transfer result, or an accepted model
claim.

## Immutable program identities

| Program | Snapshot hash | RunIdentity hash | Final receipt hash |
|---|---|---|---|
| `tabu.pretraining.query-base@1.1.0` | `d8de67cea95858f20b8cf6305693ec2f475e3ef0853a67f4c5008f861542e96b` | `77ae7671195f58d32595e4300dbf2575afe20d6bb4c308a5133a41ab703126d6` | `b42495993ba45d870c12c04c2f36e7be4949b5f331d08c84ef1c2a32c0eab342` |
| `tabu.pretraining.query-row@1.1.0` | `ef317a592bae7fb83d3a926ca1cee8df6fe8446cf5ff619b10a11a4a33228323` | `341e04bc34c2823242be064bfc6a1337a67627d70f44646f3c951dfb2cf782db` | `24704e74a217221744742adb712b9d5079c410bc5dba2b3266aaf87850b5d3db` |

Both snapshots select `tabu.training.dgx2-grow-pilot@1.0.0`: AdamW,
learning rate `3e-3`, 1500 updates, full supervised-v2 generator diversity,
and complete checkpoints every 500 updates. The Base and Row runs share data,
objective, recipe, and evaluation identities but have independent model,
component-graph, checkpoint, RunIdentity, and receipt identities.

`program impact` classifies the 1.0.0 → 1.1.0 change as `retrain` only at the
training-recipe slot and its downstream artifacts. Model contracts, component
graphs, world mixture, sampling policy, objective, and evaluation protocol are
`unchanged`.

## Execution and restart acceptance

- Host: `dgx2` / physical hostname `spark-b5b3`.
- Accelerator: NVIDIA GB10, CUDA 13.0, driver 590.48.01.
- Runtime: `wehub/ml-gpu:20260712`, PyTorch `2.12.0.dev20260322+cu130`.
- Preflight: no CUDA compute process, 0% GPU utilization, about 98 GiB available
  host memory, and 967 GiB available disk.
- Each model first produced an interrupted step-1 full-state smoke checkpoint.
  The 1500-step run exact-resumed that checkpoint with the same RunIdentity.
- Step 500, 1000, and 1500 checkpoints all contain trainer, optimizer,
  scheduler, policy cursor/state, and RNG state sidecars. The final artifact
  hashes and typed receipts passed a read-back audit.
- GPU utilization returned to 0% with no compute process after the runs and
  held-out readback.

Run root:

`/home/cms/tabubase-runs/20260831-evolution-grow-pilot-v1`

The exact portable dirty-source archive is retained under `identity/` with
SHA-256
`c988fa2234532bf1c714d5547c7c77c60333a7970763e15bf7a6f34239719cf6`.
Its provenance record has SHA-256
`539bab80f9ef777cd3d1ae08f8ecbc61966bd1b50469f24c61a22a6300cd3151`.

## Fixed held-out synthetic readback

The same 64 world addresses, root seed `424242`, objective, and evaluation
protocol were used for every checkpoint. They cover all declared supervised-v2
families, width buckets, context anchors, predictor regimes, and noise levels.
Each model scored 13,696 targets with zero abstentions.

| Program | Step 1 mean loss | Step 500 | Step 1000 | Step 1500 | Relative reduction from step 1 |
|---|---:|---:|---:|---:|---:|
| Query Base | 0.939737 | 0.764070 | 0.748782 | 0.738683 | 21.39% |
| Query Row | 0.950210 | 0.759691 | 0.745289 | 0.736889 | 22.45% |

For each run, the diagnostic quantity is

$$
\Delta L = L_{\text{step 1}} - L_{\text{step 1500}} > 0.
$$

The learning-curve receipt hashes are:

- Base: `5c786b2ec27f6d8de610c7e3fdb81ae3f50e2bcd246518bdf7f5b9d515c5c24a`.
- Row: `e695853adc4c20957461fa3efa0b5ee82b4324ce3459f6e37d5314c4a952a33a`.

This establishes finite execution, exact restart continuity, and improving
held-out synthetic loss for this bounded run. The small final-loss difference
between Base and Row is not a model-family comparison: this is one model seed,
one validation seed, and a roughly 2k-parameter executable vertical slice.

## Evidence boundary and next scale gate

The run does not establish real-data scratch performance, frozen ICL,
fine-tune transfer, multi-seed stability, or a full-scale architecture result.
The source is intentionally dirty and therefore cannot cross the Evidence
lane.

Any larger candidate should be added as new versioned graph/recipe/program
nodes. It should not rewrite these snapshots or checkpoints. Before expensive
pretraining, the next Grow gate is a versioned scale graph plus matched
multi-seed synthetic validation; real scratch, frozen ICL, and fine-tune
transfer remain independent receipts. A selected candidate may cross the
freeze boundary only from a clean, reviewable source identity.
