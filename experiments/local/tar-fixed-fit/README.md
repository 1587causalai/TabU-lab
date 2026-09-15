> Historical fixed-episode diagnostic. On 2026-09-06 the owner clarified that
> every training update must rebuild row roles and nominal codebooks. These runs
> do not establish that capability. Their preregistration, data and completed
> receipts remain unchanged. The active runner now rejects this old protocol;
> reproduce it only from its archived source snapshot. Continue with
> [the current Small protocol](../tar-small-validation/README.md).

# TAR fixed-episode fitting diagnostic

This local diagnostic asks whether the full 54,071,520-parameter TAR realization
can fit receiver-only target cells. It does not train a foundation model. Its
preregistration, source snapshot and results remain `local_unissued`; no formal
issuance, catalog promotion or pretrained pointer is created.

The data files were prepared from locally cached scikit-learn 1.8.0 bundled
Diabetes and Iris files, without downloading private data. Source file hashes,
full numeric data, externally declared classes and exact row splits are recorded.
Diabetes uses raw covariates; preprocessing is fitted on input-visible values by
TAR. Splits precede model preprocessing. Context/fit/held-out IDs are disjoint.

| Dataset | Context rows | Fixed fit targets | Held-out rows |
|---|---:|---:|---:|
| Diabetes | 32 | 16 | 8 |
| Iris | 30 | 15 | 9 |

Each training episode contains context facts and fit-row covariates. Fit labels
are loss-only sidecars. This is supervised parameter fitting of a fixed episode,
not evidence of in-context learning. Held-out evaluation uses context plus **one**
held-out row per episode. Neither fit labels nor other held-out rows enter that
forward input. Only initial and final checkpoints are evaluated; held-out scores
never select a checkpoint or an optimizer setting.

Seeds 1729, 2718 and 31415 vary initialization with fixed data/splits/codebooks.
Both gates remain 1, single-axis inducing stays enabled, and S=128 even at small N.
Each lane uses at most 1200 updates or 600 training seconds. Diagnostic overrides
are batch size 1, no warmup, and constant learning rate 1e-4. These are explicit
fit-test choices; they do not replace the model's proposed pretraining defaults.
Initial-model and context-constant baselines accompany each result. Strict fit
criteria are declared in the preregistration; exhausting the wall-time budget
without passing is inconclusive about representational capacity.

From a prepared environment in the repository root:

```sh
uv run --no-sync tabu-lab tar fit \
  --preregistration experiments/local/tar-fixed-fit/preregistration.yaml \
  --dataset diabetes --seed 1729 --device cuda:0 \
  --output-root runs-local/tar-diabetes-1729
```

Run each dataset/seed sequentially with a new output directory. At startup the
single-GPU launcher requires no other compute processes, GPU utilization below
20%, temperature below 80 C, and at least 16 GiB of available host memory. It
records `blocked_resources` and returns exit code 3 before model allocation when
those conditions fail. It never stops another service or falls back to CPU.
Runtime/gradient failures retain a failure result and any existing curve records.

`--device cpu --smoke` is a two-update, reduced-model plumbing check, reported as
`smoke_completed`. It checks the runner, scorer and saving path only; it does not
answer the full TAR fitting question. The full-model CUDA run remains a separate
attempt. Successful lanes write config, exact source/data hashes, every training
update, initial/final metrics, optimizer checkpoint and file checksums. Results
are not automatically published or synchronized to W&B.

本仓库保留协议与输入作为可复现配置。原实验的结果与 checkpoint 继续绑定原始归档；当前集成的来源字段改变 checkpoint source identity，新的运行会产生新身份。
