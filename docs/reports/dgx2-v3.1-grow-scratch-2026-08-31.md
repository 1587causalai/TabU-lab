# TabUBase / TabUR v3.1 dgx2 Grow scratch report

Date: 2026-08-31  
Status: `local_unissued`  
Claim boundary: this is a deterministic synthetic Grow diagnostic, not formal evidence,
transfer evidence, or an accepted capability claim.

## Outcome

The new synthetic prior was trained from fresh initialization for both independent
model lanes. Exact resume from the old v2/v3 snapshot was not used because the data
snapshot changed. Both v3.1 runs completed 1500 updates on dgx2 and passed the final
artifact audit.

- TabUBase: completed; best held-out checkpoint is step 500.
- TabUR: completed; best held-out checkpoint is step 1500.
- The old checkpoint remains eligible only for a future, explicitly projected
  warm-start efficiency arm with a new run identity.

## Source and program identity

- Branch: `codex/v3-grow-training`
- Training source commit: `7e4b4138ae588fd4d6dfc0ed4a52229073368a6d`
- Training source tree: `25a9386500fadf948f735139b64ce4c5933b9b2e`
- Source archive SHA-256:
  `2f776c13a0d337247e8f15c1f83a651cce60d5282607d7cdeec1951314a0491c`
- Evolution repository hash:
  `64336ec9dc4c5ba8021964f09765cd8c2de7934b4aa0deacfcd2aa05954760af`
- TabUBase ProgramSnapshot:
  `tabu.pretraining.query-base@1.3.0`,
  `7a502dce26800f6db53f4ec10b00345dfa68ca2caea4919395535f1bcca8510b`
- TabUR ProgramSnapshot:
  `tabu.pretraining.query-row@1.3.0`,
  `78a8069045384804d4fc45682f492a2f5bea2fc51d96d263388528e60b4d1be3`
- Local regression gate: `392 passed, 2 skipped`; skipped tests require optional
  `xgboost`. Ruff, manifest validation, impact projections, and catalog projections
  passed.

The immutable run root is:

`/home/cms/tabubase-runs/20260831-v3.1-grow-scratch-v1`

The dgx2 preflight used Python 3.12.3, PyTorch
`2.12.0.dev20260322+cu130`, CUDA 13.0, and NVIDIA GB10. Its file SHA-256 is
`0b71cd48aa9df49f98adef2944960b3647df8d5bf01c565c6e3c00eee2175ca4`.

## Why v3.0 failed and v3.1 is a new snapshot

The immutable v3.0 generator bounded materialized cells approximately by

$$
C_{\mathrm{linear}} = N M,
$$

but the current same-column terminal materializes routing state with dominant cost

$$
C_{\mathrm{routing}} = N^2 M,
$$

where $N$ is row count and $M$ is feature count including the response column.

When the v3.0 Base run resumed from step 8, deterministic step 9 drew
`rows=3323, width=18`, producing `209804251` routing pairs. The kernel recorded a
global OOM and killed the Python process with exit code 137 before the next
checkpoint. The incident receipt is:

`/home/cms/tabubase-runs/20260831-evolution-grow-pilot-v1/incidents/base-resume-to-1500-global-oom.json`

Its SHA-256 is
`fdaa558f07e6606e1b4db9082f81c7306c8cc1cd338715802eed1ffaa855bc35`.

The v3.1 generator is a new immutable identity. It retains the v3 world families but
caps routing pairs at `8000000`. The same step-9 address resolves to `rows=648` and
`7978176` routing pairs, reducing the dominant allocation by about 26.3 times. The
training recipe is also versioned to checkpoint every 50 updates. No v3.0 manifest
or artifact was overwritten.

## Training receipts

| Lane | Run identity hash | Run receipt hash | Final checkpoint SHA-256 |
|---|---|---|---|
| TabUBase | `c0dbb26aacc6ac8d2136fb0f4f6179ff8440b82e594c11325dd4ead63bf1764d` | `02f2a1c6593b448d6cbb67f6e9f2ed7ccb7bdbdb29899af28887d219576d2515` | `ade2135bd691d4c71f60461462a6a208796c503864f02d98704a8aa2ca010d5a` |
| TabUR | `8c4014d8e00f0d6eedcff18b7d4fc7f34041411ea07c0adfabda779630b98629` | `ff60eeeab9eaa704f77c00d81008308ca16fb5722c1b8bf1df0577ca848acfd8` | `b1919cc959f9358794d886580c1916a99838982882caf2d891da668be15f7ac4` |

Both receipts have `initialization.mode=cold`, `status=completed`, `step=1500`,
`target_steps=1500`, and `evidence_status=local_unissued`. Every long-run recovery
checkpoint from step 100 through step 1500 exists at 50-step intervals.

## Fixed held-out synthetic readback

Protocol: validation partition, root seed `424242`, 64 world addresses, the program's
own immutable v3.1 generator and objective. Every checkpoint scored 4001 targets and
had zero abstentions.

| Step | TabUBase mean loss | TabUR mean loss |
|---:|---:|---:|
| 1 | 3.565320 | 2.924014 |
| 10 | 2.695305 | 2.709016 |
| 50 | 2.711334 | 2.753709 |
| 500 | **2.666056** | 2.829537 |
| 1000 | 2.688084 | 2.745561 |
| 1500 | 2.706147 | **2.642145** |

- TabUBase step 1 to step 1500 relative improvement: 24.10%.
- TabUR step 1 to step 1500 relative improvement: 9.64%.
- TabUBase selection candidate: step 500,
  checkpoint SHA-256
  `76989a6282d5e56e7e32ed9974a0f82c64457cad4f24bec4d5080dc663d01c01`.
- TabUR selection candidate: step 1500, equal to the completed terminal checkpoint.

Evaluation receipt hashes:

- TabUBase: `478d809deadc14b9d7ad5202f747b816aef9b118b0b31f873f3b8a4dc40d9aa3`
- TabUR: `f52bb565faa88e58b6ecb0a71ac477df7062d932064454c1299ed1f4b53c90c5`

The consolidated result audit passed all snapshot, identity, recovery-chain,
finite-loss, scored-target, abstention, checkpoint, sidecar, receipt, and evaluation
hash checks. Its audit hash is
`073eab3d55f08679f13e443f445852e8945b7b1d36f3e588528e6aa3f4527ba1`;
the audit file SHA-256 is
`f1dd30d395eeac23f755964c4bc1eea85a78f081e6d16fbbdea35b4afc60d054`.

## Decision on old-checkpoint warm start

Do not reinterpret the old checkpoint as an exact resume. The data and recipe hashes
changed, so that operation is invalid by contract.

Do not spend another two full runs on warm start in this milestone. Fresh scratch
already demonstrates stable optimization under the new prior. A warm-start run would
now answer a narrower sample-efficiency question, not whether v3.1 can train. If that
question becomes important, run it as a separate paired arm with:

- explicit `StateProjection`;
- a new run identity and cold-vs-warm initialization label;
- identical v3.1 snapshot, seeds, budget, and held-out addresses;
- no inherited evidence status.

## Remaining boundary

The GB10 host still reports about 101 GiB used and 18 GiB available after the old
global OOM, even while no compute process is active. The v3.1 cap trained safely under
that condition and no new OOM appeared, but unrelated large-memory work should not be
co-scheduled until the host's stale post-OOM memory state is reset or diagnosed.
