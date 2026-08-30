# TabUBase synthetic-pretraining → real-task fine-tuning: OpenML regression-8

Date: 2026-08-30  
Evidence status: `local_unissued`  
Model: `tabu.cell.base@0.2.0`  
Profile: `supervised.label_broadcast.v1`  
Tokenizer: `cell-tokenizer.v2`, source-scoped frozen codebook, `B=100`, seed `1729`

## Outcome

This is a separate non-frozen downstream fine-tuning lane.  The synthetic
long-context PT-S1 checkpoint is used as the initialization for one arm; a
same-seed scratch model is trained with the same real-task episodes.  The
eight cached OpenML regression tables are evaluated with all held-out test
rows (no 512-row cap).  The run contains 8 datasets × 3 seeds × 2 TabUBase
arms = 48 fine-tune evaluations, plus 48 MLP/XGBoost baseline fits.

The pretrained initialization reduces mean scaled RMSE versus scratch on 7/8
datasets.  The largest gains are `cars` (46.3%), `cpu_activity` (41.9%), and
`space_ga` (10.7%).  `energy_efficiency` is the exception (-1.3%, effectively
slightly worse than scratch).  This verifies a useful initialization signal,
not broad superiority over supervised baselines.

## Fine-tuning semantics

| Field | Value |
|---|---|
| Real source | OpenML CTR23 cached ARFF snapshot, no network fetch |
| Split | deterministic 60/20/20 train/validation/test, seed `20260829` |
| Fine-tune labels | 128 train rows per dataset/seed (`label_budget=128`) |
| Training episodes | 400 updates, AdamW, learning rate `3e-4`, weight decay `1e-4` |
| Episode sampling | disjoint context/query samples from the 128-row label subset; context ≤64, query ≤32 |
| Evaluation | all held-out test rows; same test rows for TabUBase, MLP, and XGBoost |
| TabUBase arms | `pretrained` (synthetic PT-S1 init) and `scratch` |
| Baselines | MLP `(64,64)` and XGBoost, fitted on the same 128 train rows |
| Validation/tuning | none in this quick exploratory run |

This differs from the previous frozen full-context panel: frozen ICL exposes
all train-partition rows as labeled context and creates no optimizer; this
fine-tuning lane intentionally creates AdamW and updates model weights.

## Pooled test metrics

Values are means over the three seeds.  `scaled_*` uses the response mean and
scale fitted on the 128-row label subset; raw `RMSE`/`MAE` are in the original
target units.  Lower RMSE/MAE is better and higher $R^2$ is better.

| Dataset | Arm | RMSE | MAE | Scaled RMSE | Scaled MAE | $R^2$ |
|---|---|---:|---:|---:|---:|---:|
| white_wine | TabU pretrained | 0.8330 | 0.6581 | 0.9111 | 0.7199 | 0.0633 |
| white_wine | TabU scratch | 0.8646 | 0.6785 | 0.9463 | 0.7427 | -0.0130 |
| white_wine | MLP | 0.9329 | 0.7232 | 1.0201 | 0.7912 | -0.1743 |
| white_wine | XGBoost | 0.7667 | 0.5971 | 0.8383 | 0.6531 | 0.2070 |
| red_wine | TabU pretrained | 0.7446 | 0.5637 | 0.9895 | 0.7493 | 0.1592 |
| red_wine | TabU scratch | 0.8005 | 0.6273 | 1.0572 | 0.8298 | 0.0257 |
| red_wine | MLP | 0.8289 | 0.6191 | 1.0978 | 0.8204 | -0.0433 |
| red_wine | XGBoost | 0.7080 | 0.5193 | 0.9440 | 0.6927 | 0.2398 |
| cpu_activity | TabU pretrained | 3.0900 | 2.2300 | 0.1719 | 0.1241 | 0.9712 |
| cpu_activity | TabU scratch | 5.2852 | 3.5659 | 0.2957 | 0.1989 | 0.9139 |
| cpu_activity | MLP | 12.9295 | 4.9857 | 0.7216 | 0.2769 | 0.4604 |
| cpu_activity | XGBoost | 7.0789 | 3.2684 | 0.3968 | 0.1827 | 0.8434 |
| kin8nm | TabU pretrained | 0.2190 | 0.1763 | 0.8356 | 0.6727 | 0.3196 |
| kin8nm | TabU scratch | 0.2345 | 0.1901 | 0.8940 | 0.7248 | 0.2199 |
| kin8nm | MLP | 0.1837 | 0.1396 | 0.7009 | 0.5327 | 0.5215 |
| kin8nm | XGBoost | 0.2076 | 0.1668 | 0.7922 | 0.6363 | 0.3887 |
| pumadyn32nh | TabU pretrained | 0.0342 | 0.0274 | 0.9662 | 0.7748 | 0.0464 |
| pumadyn32nh | TabU scratch | 0.0375 | 0.0300 | 1.0569 | 0.8481 | -0.1423 |
| pumadyn32nh | MLP | 0.0400 | 0.0316 | 1.1288 | 0.8940 | -0.3008 |
| pumadyn32nh | XGBoost | 0.0333 | 0.0267 | 0.9410 | 0.7541 | 0.0964 |
| energy_efficiency | TabU pretrained | 1.4206 | 0.9547 | 0.1430 | 0.0961 | 0.9809 |
| energy_efficiency | TabU scratch | 1.4000 | 1.0364 | 0.1412 | 0.1046 | 0.9816 |
| energy_efficiency | MLP | 1.8932 | 1.2641 | 0.1909 | 0.1274 | 0.9663 |
| energy_efficiency | XGBoost | 1.3740 | 0.8645 | 0.1387 | 0.0873 | 0.9823 |
| cars | TabU pretrained | 2853.8217 | 2112.9222 | 0.2733 | 0.2023 | 0.9131 |
| cars | TabU scratch | 5290.4897 | 3966.7606 | 0.5086 | 0.3820 | 0.6997 |
| cars | MLP | 2801.1339 | 2090.7266 | 0.2670 | 0.1994 | 0.9168 |
| cars | XGBoost | 2903.6483 | 2065.6230 | 0.2778 | 0.1978 | 0.9110 |
| space_ga | TabU pretrained | 0.1487 | 0.1134 | 0.7977 | 0.6080 | 0.4133 |
| space_ga | TabU scratch | 0.1666 | 0.1266 | 0.8937 | 0.6790 | 0.2598 |
| space_ga | MLP | 0.1324 | 0.1002 | 0.7103 | 0.5374 | 0.5332 |
| space_ga | XGBoost | 0.1367 | 0.1030 | 0.7332 | 0.5526 | 0.5042 |

## Interpretation

The cleanest transfer signal is initialization benefit over the matched
scratch model: 7/8 mean scaled-RMSE improvements, with all three seeds better
on `cpu_activity`, `kin8nm`, `pumadyn32nh`, and `space_ga`.  Against fitted
baselines, TabUBase is competitive on `cpu_activity` and `cars`, but XGBoost
or MLP remains better on most tasks.  The pretrained and baseline arms do not
share inference semantics: TabUBase is fine-tuned through dense episodes,
whereas MLP/XGBoost are ordinary inductive fits.

## Receipt and provenance

- Receipt: [`real-finetune-openml8-pt-s1-3x3.json`](../../.local-runs/tabubase-codebook-b100-v1/real-finetune-openml8-2026-08-30/real-finetune-openml8-pt-s1-3x3.json), SHA-256 `4b06f4d1a12e10376f14b07f52fd2e7b6faf57b34eae880c69ff5ee7fa655d38`
- Remote run root: `/home/cms/tabubase-eval/20260830-real-finetune-openml8-v1`
- Remote physical host: `dgx2` / `spark-b5b3`
- Runtime: Docker `wehub/ml-gpu:20260712`, Python 3.12.3, torch `2.12.0.dev20260322+cu130`, scikit-learn `1.8.0`, XGBoost `3.3.0`, CUDA device
- Remote source snapshot SHA recorded in receipt: `9527494510ac789f0ec158b627cdf41e2332a80bc3c7a8c161415b892a8cc49b`

## Claim boundary

This is `local_unissued` exploratory evidence.  It supports a matched
synthetic-pretraining initialization comparison on eight cached OpenML
regression tables.  It does not establish SOTA, a formal benchmark, a
foundation-model claim, causal identification, or broad real-data
generalization.
