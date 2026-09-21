# V5.4 single-table fit panel — device-local readout (bd9a42d)

This is the fresh first-wave single-table fit panel after commit `bd9a42d` (`perf(v54): keep MPS readout device local`). The previous 2026-09-21 panel was stopped before this panel and is retained only as historical evidence.

Each table receives three independent replicates. Each replicate has exactly **16,384 optimizer updates**, giving an initial budget of **16,384 × 3 per table**. Replicates have independent model/order/mask/code/window/evaluation seeds and are never strict continuations of one another.

The baseline uses the current V5.4 default supervised-row Query masking with fraction 0.25, composition codec `constant_weight_composition_v1`, and V5.4 Small. `center_chunk_size=128` is explicit in every manifest. `random_cell` remains the first competing episode and is reserved for the next comparison after this baseline.

Host allocation:

- `dgx2` → `discoscm_095` → CUDA / FP64
- `deepthought` → `scm_mixed_v1_001` → CUDA / FP64
- `gongqian-mini` → `scm_mixed_v1_017` → MPS / FP32
- `zichao-mini` → `scm_mixed_v1_040` → MPS / FP32
- `dustinstudio` → `discoscm_076` → MPS / FP32

MPS runs require `PYTORCH_ENABLE_MPS_FALLBACK=0` and `PYTORCH_MPS_FAST_MATH=0`. The MPS shared-LL Cholesky and solve are device-local; a CPU fallback is an error. Each replicate uses a fresh output directory. A completed fit receipt establishes only single-table fit evidence; held-out final-test evaluation and generalization remain separate gates.
