# TabUBase real full-context frozen ICL with exact-split baselines

Date: 2026-08-30
Evidence status: `local_unissued`
Model contract: `tabu.cell.base@0.2.0`
Profile: `supervised.label_broadcast.v1`
Tokenizer: `cell-tokenizer.v2`, source-scoped frozen nominal codebook,
`B=100`, seed `1729`

## Outcome

The corrected full-context evaluation is complete on 12 real datasets, three
checkpoint seeds, and three split seeds. For every TabUBase split, the model
receives **all train-partition rows and their labels at once** as context and
predicts **every row in the complete held-out partition**. No optimizer is
created, no parameter is updated, and every recorded frozen-arm parameter hash
is identical before and after evaluation.

The expanded-synthetic `v4` checkpoints improve the descriptive pooled primary
metric over the original PT-S1 checkpoints on all 12 datasets: classification
accuracy increases and normalized NLL decreases on all seven classification
datasets; regression $R^2$ increases and scaled RMSE decreases on all five
regression datasets. This is a broad within-model improvement, but it does not
make TabUBase broadly competitive with the fitted baselines:

- MLP and XGBoost remain ahead on every classification dataset. Digits is still
  especially weak at 17.36% pooled accuracy versus 97.71% for MLP and 96.97%
  for XGBoost.
- On regression, expanded TabUBase beats the fixed MLP in RMSE and $R^2$ on
  Diabetes and QSAR Fish Toxicity. It also slightly beats XGBoost on Diabetes
  RMSE and $R^2$, while XGBoost retains slightly lower Diabetes MAE. XGBoost is
  ahead on the other four regression datasets.
- The result is checkpoint-sensitive. In particular, two Spambase checkpoints
  collapse to the majority-class operating point and Segment does not improve
  uniformly by checkpoint.

These are descriptive `local_unissued` diagnostics. They are not a formal
benchmark, SOTA result, public receipt, release, or foundation-model claim.

## Registered estimand

For split $s$, let $T_s$ be the complete train partition and $Q_s$ the complete
held-out partition. The TabUBase evidence episode is

$$
E_s=
\{(x_i,y_i):i\in T_s\}
\cup
\{(x_j,\bot):j\in Q_s\}.
$$

The run enforces:

- `context_policy: full_train`: every row and label in $T_s$ is context;
- `query_limit: null`: no held-out row is sampled away or truncated;
- all $Q_s$ predictors enter one shared-dynamics, transductive episode;
- `query_readout_chunk_rows: 64` chunks only target response readout after the
  shared dynamics and therefore does not create multiple evidence episodes;
- model parameters have `requires_grad=False`, evaluation uses inference mode,
  no optimizer is constructed, and before/after hashes must match.

The target-only response path encodes and runs dynamics over the complete
context-plus-query table once, then evaluates only query target cells against
the labeled response supports. This removes the prior all-cell dense routing
wall without changing the evidence available to any query.

The MLP and XGBoost arms use the exact same train indices, held-out indices, and
selected feature indices, verified through the shared split-manifest SHA-256.
They are conventional inductive estimators: fit on $T_s$, then predict $Q_s$,
with no interaction among held-out rows. Accordingly, their metrics are useful
side-by-side reference values, but TabUBase and the baselines do not have
identical inference semantics.

## Datasets and complete split sizes

Each dataset uses split seeds 1729, 2718, and 31415. The sizes below are the
complete row counts used on every split; classification splitting is
stratified. Digits has 64 source predictors and uses a train-only variance
projection to the model limit of 63 predictors.

| Dataset | Task | Classes | Train/context rows | Held-out/query rows | Predictors |
|---|---|---:|---:|---:|---:|
| Iris | Classification | 3 | 105 | 45 | 4 |
| Wine | Classification | 3 | 125 | 53 | 13 |
| Breast Cancer | Classification | 2 | 398 | 171 | 30 |
| Digits | Classification | 10 | 1,258 | 539 | 63 |
| Diabetes | Regression | — | 309 | 133 | 10 |
| California Housing | Regression | — | 14,448 | 6,192 | 8 |
| Banknote Authentication | Classification | 2 | 960 | 412 | 4 |
| Segment | Classification | 7 | 1,617 | 693 | 19 |
| Spambase | Classification | 2 | 3,221 | 1,380 | 57 |
| Airfoil Self Noise | Regression | — | 1,052 | 451 | 5 |
| Concrete Compressive Strength | Regression | — | 721 | 309 | 8 |
| QSAR Fish Toxicity | Regression | — | 636 | 272 | 6 |

Every full-context size is outside the synthetic pretraining curriculum, whose
largest context was $K=64$. The smallest real context is 105 and the largest is
14,448. This evaluation is therefore an explicit context-length extrapolation,
not evidence that the model was pretrained at these lengths. The architecture's
four inducing slots remain a separate capacity limitation.

## Common metrics and baseline contract

Classification metrics are computed from the same probability matrix for every
arm: accuracy, balanced accuracy, macro-F1, log loss, normalized NLL, and
one-vs-rest macro ROC-AUC. Here

$$
\operatorname{normalized\ NLL}
=\frac{\operatorname{log\ loss}}{\log C},
$$

where $C$ is the number of classes. Lower log loss and normalized NLL are
better; higher values are better for the other classification metrics.

Regression reports RMSE, MAE, train-scale-normalized RMSE and MAE, and held-out
$R^2$. The scale-normalized errors divide the raw error by the target standard
deviation estimated from the train partition only. Lower errors and higher
$R^2$ are better.

The fixed baseline contract is:

- MLP: two hidden layers `(64, 64)`, ReLU, Adam, `alpha=1e-4`,
  `learning_rate_init=1e-3`, `max_iter=500`, `tol=1e-4`, no early stopping;
  predictors use a train-only `StandardScaler`; regression targets use a
  train-only standardization followed by inverse transformation.
- XGBoost: 300 trees, depth 6, learning rate 0.05, subsample 0.8,
  column sample 0.8, `tree_method=hist`, `n_jobs=8`; classification uses log
  loss and regression uses squared-error/RMSE.
- Neither baseline is tuned on these held-out partitions. For both,
  `estimator_seed = split_seed`.

The locked runtime was Python 3.12.3, scikit-learn 1.8.0, and XGBoost 3.3.0.

## Effect of expanded synthetic pretraining

`PT-S1` and `v4` values below are descriptive means over three checkpoint seeds
and three split seeds, for nine equally weighted checkpoint-by-split units per
dataset. The table shows the primary common metric pair for each task.

| Dataset | Task | PT-S1 primary | Expanded `v4` primary | Change | PT-S1 loss | Expanded `v4` loss | Change |
|---|---|---:|---:|---:|---:|---:|---:|
| Breast Cancer | Classification | Accuracy 0.5932 | Accuracy 0.8817 | +0.2885 | Norm. NLL 0.8949 | Norm. NLL 0.5249 | -0.3700 |
| Digits | Classification | Accuracy 0.0975 | Accuracy 0.1736 | +0.0761 | Norm. NLL 0.9994 | Norm. NLL 0.9781 | -0.0213 |
| Iris | Classification | Accuracy 0.4420 | Accuracy 0.8346 | +0.3926 | Norm. NLL 0.8707 | Norm. NLL 0.4145 | -0.4562 |
| Wine | Classification | Accuracy 0.6268 | Accuracy 0.8512 | +0.2243 | Norm. NLL 0.8206 | Norm. NLL 0.4802 | -0.3404 |
| Banknote Authentication | Classification | Accuracy 0.5736 | Accuracy 0.7589 | +0.1853 | Norm. NLL 0.9573 | Norm. NLL 0.7365 | -0.2209 |
| Segment | Classification | Accuracy 0.3516 | Accuracy 0.4662 | +0.1146 | Norm. NLL 0.9324 | Norm. NLL 0.8608 | -0.0716 |
| Spambase | Classification | Accuracy 0.6058 | Accuracy 0.6349 | +0.0291 | Norm. NLL 0.9719 | Norm. NLL 0.9378 | -0.0341 |
| California Housing | Regression | $R^2$ 0.0251 | $R^2$ 0.3829 | +0.3577 | Scaled RMSE 0.9875 | Scaled RMSE 0.7901 | -0.1974 |
| Diabetes | Regression | $R^2$ 0.1775 | $R^2$ 0.4031 | +0.2256 | Scaled RMSE 0.8812 | Scaled RMSE 0.7524 | -0.1288 |
| Airfoil Self Noise | Regression | $R^2$ -0.5661 | $R^2$ 0.2413 | +0.8074 | Scaled RMSE 1.2489 | Scaled RMSE 0.8723 | -0.3767 |
| Concrete Compressive Strength | Regression | $R^2$ 0.2049 | $R^2$ 0.5767 | +0.3718 | Scaled RMSE 0.8762 | Scaled RMSE 0.6394 | -0.2367 |
| QSAR Fish Toxicity | Regression | $R^2$ 0.2095 | $R^2$ 0.5326 | +0.3230 | Scaled RMSE 0.9273 | Scaled RMSE 0.7136 | -0.2137 |

The pooled direction is favorable on all 12 datasets, but it is not uniform by
checkpoint. Segment accuracy decreases slightly for expanded seed 1729 and
2718 relative to their matched original checkpoints, while seed 31415 supplies
most of the pooled gain. On Spambase only seed 1729 improves accuracy; seeds
2718 and 31415 remain at the same majority-class accuracy. A pooled improvement
therefore does not authorize a per-seed robustness claim.

## Classification performance

Each TabUBase checkpoint row is the mean over three split seeds. `TabU v4
pooled` is the descriptive mean over all nine checkpoint-by-split units. MLP and
XGBoost rows are means over the same three split seeds.

| Dataset | Model | Accuracy | Balanced acc. | Macro-F1 | Log loss | Norm. NLL | ROC-AUC OvR macro |
|---|---|---:|---:|---:|---:|---:|---:|
| Breast Cancer | TabU v4 seed 1729 | 0.8519 | 0.8105 | 0.8283 | 0.4230 | 0.6102 | 0.9308 |
| Breast Cancer | TabU v4 seed 2718 | 0.9025 | 0.8876 | 0.8939 | 0.2929 | 0.4226 | 0.9625 |
| Breast Cancer | TabU v4 seed 31415 | 0.8908 | 0.8615 | 0.8772 | 0.3756 | 0.5419 | 0.9667 |
| Breast Cancer | **TabU v4 pooled** | **0.8817** | **0.8532** | **0.8665** | **0.3638** | **0.5249** | **0.9533** |
| Breast Cancer | MLP | 0.9688 | 0.9636 | 0.9665 | 0.1057 | 0.1525 | 0.9963 |
| Breast Cancer | XGBoost | 0.9630 | 0.9568 | 0.9601 | 0.0972 | 0.1403 | 0.9935 |
| Digits | TabU v4 seed 1729 | 0.2424 | 0.2407 | 0.2187 | 2.2300 | 0.9685 | 0.6583 |
| Digits | TabU v4 seed 2718 | 0.1633 | 0.1633 | 0.1252 | 2.2465 | 0.9756 | 0.6629 |
| Digits | TabU v4 seed 31415 | 0.1150 | 0.1131 | 0.0625 | 2.2798 | 0.9901 | 0.6099 |
| Digits | **TabU v4 pooled** | **0.1736** | **0.1724** | **0.1355** | **2.2521** | **0.9781** | **0.6437** |
| Digits | MLP | 0.9771 | 0.9771 | 0.9770 | 0.0935 | 0.0406 | 0.9989 |
| Digits | XGBoost | 0.9697 | 0.9697 | 0.9696 | 0.1032 | 0.0448 | 0.9994 |
| Iris | TabU v4 seed 1729 | 0.8000 | 0.8000 | 0.7928 | 0.3882 | 0.3534 | 0.9501 |
| Iris | TabU v4 seed 2718 | 0.7926 | 0.7926 | 0.7857 | 0.5677 | 0.5167 | 0.9526 |
| Iris | TabU v4 seed 31415 | 0.9111 | 0.9111 | 0.9101 | 0.4103 | 0.3735 | 0.9832 |
| Iris | **TabU v4 pooled** | **0.8346** | **0.8346** | **0.8295** | **0.4554** | **0.4145** | **0.9620** |
| Iris | MLP | 0.9556 | 0.9556 | 0.9555 | 0.1215 | 0.1106 | 0.9960 |
| Iris | XGBoost | 0.9333 | 0.9333 | 0.9312 | 0.2404 | 0.2188 | 0.9840 |
| Wine | TabU v4 seed 1729 | 0.8679 | 0.8792 | 0.8729 | 0.4421 | 0.4024 | 0.9711 |
| Wine | TabU v4 seed 2718 | 0.8491 | 0.8466 | 0.8499 | 0.6602 | 0.6009 | 0.9459 |
| Wine | TabU v4 seed 31415 | 0.8365 | 0.8519 | 0.8369 | 0.4804 | 0.4373 | 0.9679 |
| Wine | **TabU v4 pooled** | **0.8512** | **0.8592** | **0.8532** | **0.5275** | **0.4802** | **0.9616** |
| Wine | MLP | 0.9811 | 0.9806 | 0.9810 | 0.0629 | 0.0573 | 0.9991 |
| Wine | XGBoost | 0.9811 | 0.9841 | 0.9819 | 0.0780 | 0.0710 | 0.9998 |
| Banknote Authentication | TabU v4 seed 1729 | 0.7807 | 0.7665 | 0.7703 | 0.5434 | 0.7840 | 0.9009 |
| Banknote Authentication | TabU v4 seed 2718 | 0.6400 | 0.5947 | 0.5366 | 0.5922 | 0.8544 | 0.9627 |
| Banknote Authentication | TabU v4 seed 31415 | 0.8560 | 0.8556 | 0.8546 | 0.3958 | 0.5710 | 0.9466 |
| Banknote Authentication | **TabU v4 pooled** | **0.7589** | **0.7390** | **0.7205** | **0.5105** | **0.7365** | **0.9367** |
| Banknote Authentication | MLP | 1.0000 | 1.0000 | 1.0000 | 0.0014 | 0.0020 | 1.0000 |
| Banknote Authentication | XGBoost | 0.9919 | 0.9922 | 0.9918 | 0.0191 | 0.0275 | 0.9999 |
| Segment | TabU v4 seed 1729 | 0.5296 | 0.5296 | 0.5183 | 1.7836 | 0.9166 | 0.8058 |
| Segment | TabU v4 seed 2718 | 0.4319 | 0.4319 | 0.4085 | 1.6995 | 0.8734 | 0.7986 |
| Segment | TabU v4 seed 31415 | 0.4372 | 0.4372 | 0.3898 | 1.5420 | 0.7924 | 0.8579 |
| Segment | **TabU v4 pooled** | **0.4662** | **0.4662** | **0.4389** | **1.6750** | **0.8608** | **0.8208** |
| Segment | MLP | 0.9779 | 0.9779 | 0.9779 | 0.1095 | 0.0562 | 0.9984 |
| Segment | XGBoost | 0.9798 | 0.9798 | 0.9798 | 0.0582 | 0.0299 | 0.9994 |
| Spambase | TabU v4 seed 1729 | 0.6932 | 0.6117 | 0.5824 | 0.6049 | 0.8728 | 0.9451 |
| Spambase | TabU v4 seed 2718 | 0.6058 | 0.5000 | 0.3773 | 0.6805 | 0.9817 | 0.4000 |
| Spambase | TabU v4 seed 31415 | 0.6058 | 0.5000 | 0.3773 | 0.6647 | 0.9590 | 0.6005 |
| Spambase | **TabU v4 pooled** | **0.6349** | **0.5372** | **0.4456** | **0.6501** | **0.9378** | **0.6485** |
| Spambase | MLP | 0.9411 | 0.9362 | 0.9380 | 0.4298 | 0.6201 | 0.9774 |
| Spambase | XGBoost | 0.9524 | 0.9490 | 0.9501 | 0.1283 | 0.1851 | 0.9882 |

## Regression performance

The aggregation units are the same as in the classification table.

| Dataset | Model | RMSE | MAE | Scaled RMSE | Scaled MAE | $R^2$ |
|---|---|---:|---:|---:|---:|---:|
| California Housing | TabU v4 seed 1729 | 0.8953 | 0.6525 | 0.7772 | 0.5665 | 0.4029 |
| California Housing | TabU v4 seed 2718 | 0.9279 | 0.7023 | 0.8056 | 0.6097 | 0.3586 |
| California Housing | TabU v4 seed 31415 | 0.9071 | 0.7233 | 0.7875 | 0.6279 | 0.3871 |
| California Housing | **TabU v4 pooled** | **0.9101** | **0.6927** | **0.7901** | **0.6014** | **0.3829** |
| California Housing | MLP | 0.5261 | 0.3504 | 0.4567 | 0.3042 | 0.7939 |
| California Housing | XGBoost | 0.4543 | 0.3010 | 0.3944 | 0.2613 | 0.8463 |
| Diabetes | TabU v4 seed 1729 | 58.9380 | 47.9961 | 0.7602 | 0.6190 | 0.3908 |
| Diabetes | TabU v4 seed 2718 | 57.6845 | 47.2613 | 0.7441 | 0.6096 | 0.4168 |
| Diabetes | TabU v4 seed 31415 | 58.3779 | 48.5333 | 0.7529 | 0.6260 | 0.4016 |
| Diabetes | **TabU v4 pooled** | **58.3335** | **47.9303** | **0.7524** | **0.6182** | **0.4031** |
| Diabetes | MLP | 73.0113 | 56.1961 | 0.9411 | 0.7243 | 0.0557 |
| Diabetes | XGBoost | 59.3925 | 47.2236 | 0.7658 | 0.6087 | 0.3774 |
| Airfoil Self Noise | TabU v4 seed 1729 | 5.8447 | 4.5611 | 0.8483 | 0.6619 | 0.2826 |
| Airfoil Self Noise | TabU v4 seed 2718 | 6.1672 | 4.9448 | 0.8951 | 0.7177 | 0.2015 |
| Airfoil Self Noise | TabU v4 seed 31415 | 6.0180 | 4.7480 | 0.8734 | 0.6891 | 0.2398 |
| Airfoil Self Noise | **TabU v4 pooled** | **6.0100** | **4.7513** | **0.8723** | **0.6895** | **0.2413** |
| Airfoil Self Noise | MLP | 1.9174 | 1.3110 | 0.2782 | 0.1902 | 0.9221 |
| Airfoil Self Noise | XGBoost | 1.7859 | 1.1792 | 0.2592 | 0.1711 | 0.9326 |
| Concrete Compressive Strength | TabU v4 seed 1729 | 9.8386 | 7.6610 | 0.5878 | 0.4577 | 0.6435 |
| Concrete Compressive Strength | TabU v4 seed 2718 | 11.4095 | 9.2481 | 0.6819 | 0.5527 | 0.5217 |
| Concrete Compressive Strength | TabU v4 seed 31415 | 10.8610 | 8.9719 | 0.6487 | 0.5358 | 0.5650 |
| Concrete Compressive Strength | **TabU v4 pooled** | **10.7031** | **8.6270** | **0.6394** | **0.5154** | **0.5767** |
| Concrete Compressive Strength | MLP | 5.3123 | 3.6845 | 0.3172 | 0.2201 | 0.8952 |
| Concrete Compressive Strength | XGBoost | 4.2444 | 2.7851 | 0.2534 | 0.1664 | 0.9332 |
| QSAR Fish Toxicity | TabU v4 seed 1729 | 1.0001 | 0.7434 | 0.6971 | 0.5182 | 0.5538 |
| QSAR Fish Toxicity | TabU v4 seed 2718 | 1.0123 | 0.7430 | 0.7057 | 0.5181 | 0.5432 |
| QSAR Fish Toxicity | TabU v4 seed 31415 | 1.0586 | 0.8244 | 0.7381 | 0.5749 | 0.5007 |
| QSAR Fish Toxicity | **TabU v4 pooled** | **1.0237** | **0.7703** | **0.7136** | **0.5371** | **0.5326** |
| QSAR Fish Toxicity | MLP | 1.0620 | 0.7153 | 0.7399 | 0.4983 | 0.4929 |
| QSAR Fish Toxicity | XGBoost | 0.9016 | 0.6384 | 0.6282 | 0.4447 | 0.6364 |

## Frozen and compatibility gates

Four complete machine-readable `local_unissued` full-context frozen result
panels were produced: original PT-S1 and expanded `v4`, each on old6 and new6.
Every panel contains 162 metric records:
six datasets by three checkpoints by three splits by three frozen arms
(`pretrained_frozen`, `random_init_frozen`, and `pretrained_shuffled`). Across
the four panels:

- all 36 checkpoint-by-arm before/after parameter-hash pairs are exactly equal;
- both `optimizer_created` and `frozen_arm_optimizer_created` are `false`;
- all shared dataset hashes and split-manifest hashes match the corresponding
  MLP/XGBoost panels;
- all train rows, held-out query rows, and selected features are identical
  between the compared arms;
- the four generated comparison artifacts pass every compatibility and frozen
  gate.

Whole-source-tree equality was not one of those generated comparison gates.
The expanded `v4` and baseline panels record source-tree SHA-256
`c291d71a...`, while the original PT-S1 panels record `92c220ee...`. A
post-run checksum dry-run over `src/tabu_lab/**/*.py`, excluding bytecode,
found exactly one difference: the PT-S1 source tree contains the newly added
`tabubase_full_context_comparison.py`; every pre-existing Python file,
including the evaluator path, is byte-identical to the archived `c291d71a`
tree. The numerical comparison remains descriptive, and this audit does not
retroactively turn whole-tree source equality into a preregistered gate.

Hash scope is one complete panel invocation for each checkpoint and arm: the
hash is taken before the dataset/split loop and again after that loop. The
receipt does not misrepresent these as separate adjacent hashes for every
dataset-split record.

The real full-context implementation is also covered by exact dense-versus-
target-only parity at small size, query-readout chunk invariance, all-query
coverage, no-optimizer tripwires, parameter-hash checks, exact split-sharing
tests, metric tests, and baseline preprocessing/probability-alignment tests.

## Execution and resource evidence

The live project path and Git top level both resolved to the same historical
donor workspace. Its machine-local path is intentionally omitted from this
portable report.

The inspected branch was `codex/tabubase-eval-chain` at
`3502fdd80539f2a8b9703cc4e4546fd01f3826ce`. The pre-existing worktree was
dirty; 405 dirty/untracked entries were observed before this implementation.
No unrelated user change was cleaned, reset, or committed.

The `dgx2` alias resolved to physical host `spark-b5b3`. Before execution it had
no TabUBase process and no GPU compute process. The historical PT-S1 root was
left intact and isolated execution ID `tabubase-real-full-context-v1` was used;
the machine-local root is intentionally omitted.

The post-run check again found zero TabUBase jobs. The expanded `v4` and
baseline source snapshot has tree SHA-256
`c291d71a14f4701b94a01fd2be3050947943e21e061ad9b4539741b5738c51d2`;
its archive SHA-256 is
`3a75d1b03cd4709d38906ea7a9443edead5d47c7878e5a12384ca5b10909451e`.
The two PT-S1 panels record source-tree SHA-256
`92c220ee79b6220e74177a98378e6bc54fa20f67aa39ae6b28c9c174d338ddb1`;
the corresponding source tree, including the comparator source, is preserved as
`source-snapshot-92c220ee79b6220e.tar.gz` with archive SHA-256
`1ecf8ba2e598a03ceb72a5c28bd1325b7f98521739a4880c172c977adec0bf32`.
The one-file difference and non-gate status are disclosed above.

Long-table smokes included Spambase with 3,221 context rows and 1,380 held-out
rows, and California Housing with 14,448 context rows and 6,192 held-out rows.
Observed peak CUDA allocation was approximately 1.91 GB and 0.52 GB,
respectively. Both smokes kept hashes unchanged. The prior dense all-cell path
would have required approximately 18.3 GiB and 57.1 GiB just for the dominant
routing difference tensor; the target-only path avoids that allocation.

## Artifact index

The machine-readable artifact ID `real-full-context-v4-manifest` records the
environment, source snapshot, gates, logical artifact names, and hashes. It is
registered in the [portable local-artifact index](local-artifact-index.json) and
is not bundled with this repository. The manifest SHA-256 is
`9448bbf5010077a5316295fe0d06aeed0607e1dc229a64ad4be1fd02ed98b690`.

Main result hashes are:

| Artifact | SHA-256 |
|---|---|
| `real-full-context-pts1-old6-3x3.json` | `a629bf91360eb385160ea10323238083db86462eba293baa0377dccca2b81faf` |
| `real-full-context-pts1-new6-3x3.json` | `22eb106fe636221c8a620fecefd2e20e0d3352aade57c53b0724de751990a9ce` |
| `real-full-context-v4-old6-3x3.json` | `f7c8ada9f2e513436edaf3e71da3aa08ee9b6aa91a7628894df060c0516f6686` |
| `real-full-context-v4-new6-3x3.json` | `4b5673d63f43d67343576b1bb20b876e7ab64bd560f90901d8b06bc4a535ddf0` |
| `baselines-full-context-old6-3splits.json` | `f0d3399de25d00a75d4faf50928d41e27aeef390f567a9fc42647d1974cbddf4` |
| `baselines-full-context-new6-3splits.json` | `85d131d18cb3bfc8a5f285fa8d0b1ed1639ec4440c2294d9aa1fef6df2ac57bc` |
| `comparison-full-context-pts1-old6.json` | `42d713a4dfde3c08041f74e1c1a19826599debb9b1424658ebc3dd1f6952bad3` |
| `comparison-full-context-pts1-new6.json` | `c782910c33ecab145fe60fc82632b4bef583b856c5d76f8a1bd5a5c5fdfc4920` |
| `comparison-full-context-v4-old6.json` | `105896310486bfa02037818ae365bb0d203ae8c2e4c7158b5c2a054669be9890` |
| `comparison-full-context-v4-new6.json` | `7b545494a5ade966512ecc105ecf4490d7d6d59b9d03917706d14959902a4cb8` |

The preregistered protocols are
[`real-full-context-frozen-icl.yaml`](../../experiments/transfer-base-v2/real-full-context-frozen-icl.yaml)
and
[`real-full-context-frozen-icl-openml-new6.yaml`](../../experiments/transfer-base-v2/real-full-context-frozen-icl-openml-new6.yaml).
They preregister the full-context estimand, data/split rules, frozen gates,
common metrics, and baseline protocol. They do not hash-bind both concrete
checkpoint lines: old6 names PT-S1 and its seeds, while new6 describes the
expanded checkpoints without registering the final `v4` checkpoint hashes.
This report therefore does not claim that every concrete checkpoint artifact
was individually hash-preregistered.

## Claim boundary and next gate

This run establishes a concrete optimizer-free real-data ICL mechanism at full
train-partition context length and shows that expanded synthetic pretraining
substantially improves its descriptive pooled performance across the present
12-dataset panel. It does not establish uniform checkpoint robustness or broad
baseline competitiveness.

The most discriminating next pretraining gate is not simply more updates. It is
to widen the context-length curriculum beyond $K=64$ while preserving the `v4`
law diversity, and to test whether improvements survive by checkpoint on the
hard cases: Digits, Segment, and Spambase. Any such run should remain separate
from this frozen receipt set and should retain the same exact-split baseline
panel for comparability.
