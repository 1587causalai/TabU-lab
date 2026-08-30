# TabUBase cached-OpenML regression-8 frozen full-context exploration

Date: 2026-08-30
Evidence status: `local_unissued`
Model: `tabu.cell.base@0.2.0`
Profile: `supervised.label_broadcast.v1`
Tokenizer: `cell-tokenizer.v2`, source-scoped frozen codebook, `B=100`, seed `1729`

## Outcome

An independent exploratory panel of eight additional numeric-only OpenML CTR23
tables was evaluated on dgx2 (`spark-b5b3`). The frozen TabUBase arm sees every
row in the train partition with its label and every held-out predictor in one
transductive episode. Held-out responses are scorer-only values. No optimizer is
created and every frozen arm has byte-identical before/after parameter hashes.

The panel contains 8 datasets × 3 checkpoint seeds × 3 split seeds × 3 frozen
arms = 216 frozen records. MLP and XGBoost were fitted on the same train rows
and scored on the same complete held-out rows (8 × 3 × 2 = 48 baseline fits).
The strict comparison receipt verifies dataset hashes, split manifests,
context rows, query rows, feature indices, and all frozen gates.

This is a deliberately greedy data expansion, not a replacement for the prior
12-dataset panel. It is regression-only; the existing old6/new6 panels remain
the source for classification coverage.

## Frozen semantics and provenance

| Field | Value |
|---|---|
| Panel | `tabubase-real-full-context-cached-openml-regression-8-v1` |
| Source cache | `openml-ctr23-tabuf-v0-cache` (machine-local path omitted) |
| Parser | `scipy.io.arff.loadarff` / `liac-arff` |
| Context | all train-partition rows and labels |
| Query | all held-out predictors, one transductive episode |
| Query readout | 64-row response-readout chunks only |
| Checkpoints | PT-S1 long-context seeds `1729,2718,31415` |
| Splits | `1729,2718,31415`, train fraction `0.7` |
| Frozen optimizer gates | `optimizer_created=false`, `frozen_arm_optimizer_created=false` |
| Hash gate | `all_frozen_arm_parameter_hashes_unchanged=true` |
| Runtime | dgx2 / `spark-b5b3`, Docker `wehub/ml-gpu:20260712`, Python 3.12.3, torch `2.12.0.dev20260322+cu130`, CUDA 13.0 |

## Common held-out metrics

Rows are descriptive means over the three checkpoint × split units for TabUBase
and over the three split fits for each baseline. Lower RMSE/MAE is better;
higher $R^2$ is better.

| Dataset | Arm | RMSE | MAE | Scaled RMSE | Scaled MAE | $R^2$ |
|---|---|---:|---:|---:|---:|---:|
| cars | TabU frozen | 4881.4006 | 3729.2945 | 0.4942 | 0.3775 | 0.7518 |
| cars | MLP | 2092.5370 | 1443.2156 | 0.2118 | 0.1461 | 0.9549 |
| cars | XGBoost | 2276.0486 | 1562.9862 | 0.2304 | 0.1582 | 0.9467 |
| cpu_activity | TabU frozen | 12.8843 | 7.8474 | 0.6886 | 0.4194 | 0.4393 |
| cpu_activity | MLP | 2.7556 | 1.9225 | 0.1472 | 0.1027 | 0.9754 |
| cpu_activity | XGBoost | 2.2619 | 1.5636 | 0.1209 | 0.0836 | 0.9836 |
| energy_efficiency | TabU frozen | 3.0870 | 2.4165 | 0.3066 | 0.2400 | 0.9026 |
| energy_efficiency | MLP | 1.0333 | 0.7164 | 0.1026 | 0.0711 | 0.9893 |
| energy_efficiency | XGBoost | 0.5639 | 0.3503 | 0.0560 | 0.0348 | 0.9968 |
| kin8nm | TabU frozen | 0.2084 | 0.1678 | 0.7890 | 0.6353 | 0.3675 |
| kin8nm | MLP | 0.0738 | 0.0578 | 0.2794 | 0.2187 | 0.9208 |
| kin8nm | XGBoost | 0.1269 | 0.0990 | 0.4805 | 0.3748 | 0.7657 |
| pumadyn32nh | TabU frozen | 0.0342 | 0.0273 | 0.9571 | 0.7641 | 0.1227 |
| pumadyn32nh | MLP | 0.0337 | 0.0269 | 0.9442 | 0.7523 | 0.1448 |
| pumadyn32nh | XGBoost | 0.0220 | 0.0175 | 0.6160 | 0.4913 | 0.6366 |
| red_wine | TabU frozen | 0.7106 | 0.5624 | 0.8809 | 0.6972 | 0.2228 |
| red_wine | MLP | 0.7733 | 0.5444 | 0.9582 | 0.6745 | 0.0773 |
| red_wine | XGBoost | 0.5862 | 0.4172 | 0.7266 | 0.5170 | 0.4710 |
| space_ga | TabU frozen | 0.1482 | 0.1063 | 0.7506 | 0.5381 | 0.4444 |
| space_ga | MLP | 0.1016 | 0.0756 | 0.5144 | 0.3829 | 0.7386 |
| space_ga | XGBoost | 0.1126 | 0.0788 | 0.5706 | 0.3989 | 0.6789 |
| white_wine | TabU frozen | 0.8244 | 0.6507 | 0.9282 | 0.7326 | 0.1188 |
| white_wine | MLP | 0.7939 | 0.5603 | 0.8936 | 0.6307 | 0.1807 |
| white_wine | XGBoost | 0.6325 | 0.4667 | 0.7121 | 0.5254 | 0.4818 |

On pooled $R^2$, TabUBase exceeds MLP on red_wine only (1/8) and does not
exceed XGBoost on any of the eight datasets. On pooled scaled RMSE it is below
MLP on red_wine only (1/8) and below XGBoost on none. Those are descriptive
comparisons between different inference semantics: TabUBase is transductive
over held-out predictors, while MLP/XGBoost are inductive fitted comparators.

## Receipts and hashes

- Frozen receipt ID: `cached-openml-frozen-full-context-3x3`, SHA-256 `3f8aa78f134d3f75d9b1cc5b96fa18dac81e3030a5b419cc751a95bc159bdd67`
- Baseline receipt ID: `cached-openml-baselines-full-context-3x3`, SHA-256 `078b3a1c33ac3454e1bf57b3823c27432e96233984500a51fa569e86bd6b2c4d`
- Strict comparison ID: `cached-openml-comparison-full-context-3x3`, SHA-256 `66e97a1fe1b694acf9d9c3f46a2978408d49e6b3b420669e9dc0428df9e0507f`
- Artifact manifest ID: `cached-openml-full-context-manifest`, SHA-256 `f2194d4c4508b47598802f4b0b866d86228806e13205308f859c331b393a7d2b`

These logical locators are also recorded in the
[portable local-artifact index](local-artifact-index.json); artifacts are not
bundled with this repository.

Remote source-tree SHA recorded by both run receipts: `4e77c1dd59f2e6d294b119acb08c5615b3c100425dd48808f85378d74e050893`.
The local worktree was dirty before this run and remains preserved; the local
experiment source is the current `codex/tabubase-eval-chain` checkout at
`3502fdd80539f2a8b9703cc4e4546fd01f3826ce` plus the panel patch.

## Claim boundary

This is `local_unissued` exploratory evidence. The cached OpenML files and CTR23
task records are provenance inputs, not independently reviewed formal data
authority. The panel demonstrates that the frozen/no-optimizer path and exact
held-out comparison can be scaled to eight more real regression tables. It does
not establish a benchmark, SOTA, foundation-model, causal, or broad real-data
generalization claim, and it does not authorize another pretraining phase.
