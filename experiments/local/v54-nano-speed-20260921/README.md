# V5.4 Nano readout speed qualification

This standalone script derives **both** models from the same seeded Nano configuration.
It loads an existing V5.4 single-table manifest, replaces its model with
`V54Config.from_size('nano', center_chunk_size=original.config.center_chunk_size)`,
and records the original manifest, actual configuration, source digest and runtime.
The full-observation readout remains the executable reference. This is a short
qualification run; it creates no model checkpoints and is not evidence of fitting.

Run from the actual repository on an authorized training host, using its existing
training launcher. Choose a fresh output filename; existing files are rejected.

```sh
~/.local/bin/wehub-python --profile train -u \
  experiments/local/v54-nano-speed-20260921/benchmark.py \
  --manifest experiments/local/restoration-v54-single-table-20260921-bd9a42d/manifests/discoscm_076.r0.json \
  --device cuda:0 --output /absolute/fresh/path/nano-speed.json
```

For MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=0` and `PYTORCH_MPS_FAST_MATH=0`
**before** Python starts, and select `--device mps`. CUDA uses FP64; MPS uses FP32.
The script calls the existing `configure_runtime` and does not relax its guards.

The first three paired updates compare loss, every parameter gradient after
clipping (including `None` patterns), model parameters/buffers and optimizer state.
CUDA tolerances are `atol=1e-9, rtol=1e-8`; MPS tolerances are
`atol=2e-5, rtol=3e-4`. These comparisons are now **diagnostic by default**:
differences remain in the receipt and do not prevent speed timing. Nonfinite
values and runtime errors still fail the run. The historical runs used a strict
comparison gate; `--strict-equivalence` explicitly reproduces that policy.
The receipt separates `numerical_comparison_passed` from `timing_completed`;
the default outcome `timing_completed` does not claim strict update equivalence.
One paired warmup precedes three alternating AB/BA rounds, each with two steps
per variant. `--rounds`, `--calls` and `--warmup` can increase these counts.
Both variants traverse identical schedule indices, masks and code seeds.

Add `--full-repeat-control` to run **both** independent copies with full readout.
This measures full-versus-full execution variation under the same tolerances.
Its default outcome is `control_timing_completed`, with numerical differences
reported separately. Under `--strict-equivalence`, the historical outcomes
`control_qualified` and `control_equivalence_failed` remain available. Its timing
ratio is recorded as a repeat comparison rather than an optimization speedup.
The historical variant keys remain `full` and `active`; `variant_readout` explicitly
identifies the second branch as a full-readout repeat in this mode.

Timing synchronizes around each full `train_step`, including episode construction,
preparation, forward/backward, clipping, update, finite checks and returned metrics.
It excludes comparison/report IO, checkpoints and periodic fixed evaluation.
The headline is the median of round means, not a comparison with old Small runs.
CUDA peak allocation and available MPS memory counters include both resident
models/optimizers. Each step emits compact JSON progress to stdout, and the reserved
JSON receipt is refreshed between phases/rounds. Interrupted writes are not atomic.

## Observed results, 2026-09-21

The measured v2 source digest is
`8db026b30a6a2b071f0c0bfc8745ed686dcf4f0f68db886d61f69393d106ba87`.
These are same-Nano comparisons with full readout, not speed comparisons with
the stopped Small experiments. All six v2 receipts completed timing.

| Host | Table | Full readout, s/step | Optimized, s/step | Ratio |
| --- | --- | ---: | ---: | ---: |
| dgx2 | [discoscm_076](results/dgx2-discoscm_076-v2.json) | 1.3807 | 0.2968 | 4.65× |
| dgx2 | [discoscm_095](results/dgx2-discoscm_095-v2.json) | 1.3820 | 0.2971 | 4.65× |
| dustinstudio | [discoscm_076](results/dustinstudio-discoscm_076-v2.json) | 0.9378 | 0.2659 | 3.53× |
| dustinstudio | [discoscm_095](results/dustinstudio-discoscm_095-v2.json) | 0.9231 | 0.3773 | 2.45× |
| gongqian-mini | [scm_mixed_v1_017](results/gongqian-mini-scm_mixed_v1_017-v2.json) | 0.4484 | 0.1572 | 2.85× |
| zichao-mini | [scm_mixed_v1_040](results/zichao-mini-scm_mixed_v1_040-v2.json) | 1.8014 | 0.5575 | 3.23× |

The v2 CUDA path reads only nonzero-loss targets; the MPS path skips inactive
columns and preserves the requested rows of each active column. These short
measurements include synchronized training steps but exclude periodic evaluation
and checkpoints. MPS timings varied across rounds; sustained-run throughput is
reported separately in the fit panel. No deepthought speed ratio was measured.

The historical v1 MPS row-pruning attempt exceeded the original optimizer
comparison tolerance, so that older script skipped timing. Its
[receipt](results/dustinstudio-discoscm_076-v1.json) remains unchanged. This does
not establish a fitting failure or prove column pruning is the fastest MPS
option. The owner's subsequent exploration policy makes such comparisons
diagnostic: a candidate's practical value is decided by measured speed and
fitting behavior. The optional full-repeat control has not been run.
