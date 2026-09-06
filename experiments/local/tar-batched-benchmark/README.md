# TAR batched axes benchmark

This local diagnostic compares the frozen serial implementation with batched,
episode-masked axis operators. Both arms use the exact same initial weights,
fixed episode/codebook, full 54,071,520-parameter model and FP32 backbone / FP64
terminal. TF32 is disabled. No optimizer update occurs in the timing comparison.

The dataset files are byte-identical to the resampled fitting diagnostic.
The reference implementation is hashed in `preregistration.yaml`; it lives in
`tests/unit/tar/serial_reference.py` solely as a correctness/performance oracle.
No new model ID or mathematical variant is introduced.

The CLI first compares full carriers, responses, predictions, loss and all
parameter gradients under declared FP32 tolerances. Each arm then has two warmup
steps and five CUDA-synchronized forward + loss + backward timings per dataset.
One additional CPU/CUDA profiler trace per arm records actual kernel counts.
The timer includes episode compilation and scoring. It excludes optimizer updates,
new episode sampling, audit hashing and evaluation; it is not whole-run speedup.

```bash
uv run --no-sync tabu-lab tar benchmark \
  --preregistration experiments/local/tar-batched-benchmark/preregistration.yaml \
  --reference tests/unit/tar/serial_reference.py \
  --device cuda:0 --output-root /tmp/tar-batched-benchmark-new
```

Outputs include resolved preregistration, environment, matched episode IDs, errors,
individual timings, median speedup, peak allocated memory, profiler traces and
checksums. The output directory must be new. Results remain `local_unissued`.
Large-table scalability and other devices require separate measurements.

Implementation: OMAB supports independent leading batch axes and dense masks.
Column processing uses `collect_column_chunk=4`; row processing uses
`receiver_chunk_rows=256`. Inner source/receiver chunks still preserve a global
softmax. Empty evidence deletes inducing residuals and attention output bias;
Null remains exactly zero. Python loops now schedule chunks, rather than every
individual row/column. No cross-episode evidence sharing or fixed codebook cache
is introduced.

本仓库保留协议与输入作为可复现配置。原实验的结果与 checkpoint 继续绑定原始归档；当前集成的来源字段改变 checkpoint source identity，新的运行会产生新身份。
