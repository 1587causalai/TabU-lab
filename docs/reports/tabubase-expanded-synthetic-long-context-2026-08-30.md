# TabUBase expanded synthetic long-context pretraining and frozen full-context ICL

Date: 2026-08-30
Evidence status: `local_unissued`
Model contract: `tabu.cell.base@0.2.0`
Profile: `supervised.label_broadcast.v1`
Tokenizer: `cell-tokenizer.v2`, source-scoped frozen nominal codebook,
`B=100`, seed `1729`

## Outcome

The bounded long-context intervention is complete for three PT-S1 checkpoint
seeds. It holds the expanded-synthetic `v4` world-law distribution and model
architecture fixed, extends the training curriculum from

$$
(2,4,8,16,32,64)
$$

to

$$
(2,4,8,16,32,64,128,256,512),
$$

and replaces the dense all-cell training terminal with a differentiable
query-response-only path. All three 20,000-world, 20,000-update runs pass the
finite, exact-resume, and fixed-validation-improvement gates.

The frozen evaluation requested for downstream tasks is also complete:

- every train-partition row and label is exposed as context;
- every held-out predictor row is included in one complete transductive
  evidence episode;
- no optimizer is created and no parameter is updated in any frozen arm;
- all before/after parameter hashes are exactly equal;
- 12 datasets, three checkpoints, and three split seeds are evaluated;
- MLP and XGBoost use the exact same train, held-out, and feature indices.

The result establishes a materially stronger optimizer-free ICL mechanism than
the original PT-S1 line, but not broad parity with fitted tabular learners. All
seven classification datasets remain behind MLP and XGBoost in pooled accuracy.
On regression, TabUBase descriptively beats both baselines on Diabetes by
pooled $R^2$, and beats MLP but not XGBoost on QSAR Fish Toxicity.

Relative to the direct expanded-`v4` predecessor, long-context training improves
pooled accuracy or $R^2$ on 10 of 12 datasets. The important negative result is
that classification normalized NLL worsens on Breast Cancer, Iris, Wine, and
Segment even when accuracy improves on three of those four datasets. Added
context training therefore helped decision quality more consistently than
probability calibration.

This is not a formal receipt, public benchmark, SOTA claim, released checkpoint,
or foundation-model designation. It also does not close the registered
$K=1024,2048,4096,8192$ runtime ladder or authorize PT-S2.

## Intervention and held-constant fields

| Field | Value |
|---|---|
| Generator | `tabubase.expanded-synthetic.v4` |
| Protocol | `tabubase.expanded-synthetic-long-context.v1` |
| Training $K$ | `2,4,8,16,32,64,128,256,512` |
| Frozen context candidate bank | 512 rows |
| Query rows per world | 64 |
| Query readout chunk | 64 rows |
| Training forward mode | `query_response_only_v1` |
| Loss normalization | per target cell |
| World/update budget | 20,000 / 20,000 per PT-S1 seed |
| Model architecture | unchanged |
| World-law distribution | unchanged from expanded `v4` |

Query response truth remains outside model forward and is used only by the
loss-side truth object. The query-response path computes only the required query
target cells after shared table dynamics; query readout chunking never truncates
context or partitions the evidence episode.

## Preflight and training gates

The same frozen training snapshot passed:

- dense-versus-query-response loss and parameter-gradient parity at small $K$;
- query-chunk invariance and a tripwire proving the dense terminal is not called;
- 45 focused unit tests locally and in the remote source snapshot;
- 12/12 CUDA forward/backward cases across four modalities and
  $K\in\{128,256,512\}$;
- peak CUDA allocation 415,290,368 bytes and reservation 448,790,528 bytes;
- a complete 20,000-world Stage-A compile with zero failures.

PT-S0 seed 1729 passes with validation loss
$8.41073\rightarrow8.31042$. PT-S1 results are:

| Seed | Initial validation | Final validation | Finite | Exact resume | Improved | Checkpoint-20000 SHA-256 |
|---:|---:|---:|---|---|---|---|
| 1729 | 8.41073 | 8.02498 | pass | pass | pass | `71a4746a...05e4` |
| 2718 | 10.21009 | 9.64268 | pass | pass | pass | `2f01dc32...4a0c` |
| 31415 | 23.78970 | 23.54601 | pass | pass | pass | `b3aa08db...f224` |

The exact-resume probe remains the registered single-step in-memory restoration
of model, optimizer, and RNG state. It is not misrepresented as a persisted
checkpoint interruption/restart trial.

## Synthetic frozen low-shot preservation

The synthetic held-out panel retains the historical nested grid
$K\in\{0,1,2,4,8,16,32\}$. It is a low-shot preservation gate, not a direct
$K=128\ldots512$ utilization curve.

| Modality/control | Original PT-S1 gain | Long-context gain | Long-context 95% CI |
|---|---:|---:|---:|
| Classification, pretrained vs random | 0.02291 | 0.04970 | [0.04468, 0.05476] |
| Classification, normal vs shuffled | 0.07961 | 0.10142 | [0.09218, 0.11123] |
| Regression, pretrained vs random | 0.09874 | 0.34568 | [0.32674, 0.36681] |
| Regression, normal vs shuffled | 0.23216 | 0.25554 | [0.24185, 0.26940] |

All three checkpoint receipts declare
`frozen_arm_optimizer_created: false`; all pretrained, random-init, and
label-shuffled before/after state hashes are equal. The world-clustered aggregate
gate passes for classification and regression.

## Real full-context estimand

For split $s$, the complete evidence episode is

$$
E_s=
\{(x_i,y_i):i\in T_s\}
\cup
\{(x_j,\bot):j\in Q_s\},
$$

where $T_s$ is the entire downstream train partition and $Q_s$ is the entire
held-out partition. The run records `context_policy: full_train`,
`query_limit: null`, and `query_policy: all_heldout_rows`.

The two accepted frozen receipts each contain 162 records:

$$
6\ \text{datasets}
\times3\ \text{checkpoints}
\times3\ \text{splits}
\times3\ \text{frozen arms}.
$$

Across both receipts, `optimizer_created` and
`frozen_arm_optimizer_created` are false, and every checkpoint-by-arm hash is
unchanged. The strict comparison receipts additionally verify identical dataset
hashes, train indices, held-out indices, feature indices, context rows, and query
rows against the baseline receipts.

The real context sizes range from 105 to 14,448 labeled rows, and the held-out
query sizes range from 45 to 6,192 rows. Values above 512 are inference-time
context extrapolation; they are not evidence that those lengths occurred in
pretraining.

## Classification metrics

TabUBase rows are descriptive means over nine checkpoint-by-split units. MLP and
XGBoost rows are means over the same three split seeds. Higher is better except
normalized NLL.

| Dataset | Model | Accuracy | Balanced acc. | Macro-F1 | Norm. NLL | ROC-AUC OvR macro |
|---|---|---:|---:|---:|---:|---:|
| Breast Cancer | TabUBase frozen | 0.8317 | 0.7762 | 0.7913 | 0.6182 | 0.9707 |
| Breast Cancer | MLP | 0.9688 | 0.9636 | 0.9665 | 0.1525 | 0.9963 |
| Breast Cancer | XGBoost | 0.9630 | 0.9568 | 0.9601 | 0.1403 | 0.9935 |
| Digits | TabUBase frozen | 0.1861 | 0.1857 | 0.1452 | 0.9745 | 0.6747 |
| Digits | MLP | 0.9771 | 0.9771 | 0.9770 | 0.0406 | 0.9989 |
| Digits | XGBoost | 0.9697 | 0.9697 | 0.9696 | 0.0448 | 0.9994 |
| Iris | TabUBase frozen | 0.8519 | 0.8519 | 0.8436 | 0.4585 | 0.9627 |
| Iris | MLP | 0.9556 | 0.9556 | 0.9555 | 0.1106 | 0.9960 |
| Iris | XGBoost | 0.9333 | 0.9333 | 0.9312 | 0.2188 | 0.9840 |
| Wine | TabUBase frozen | 0.8826 | 0.8683 | 0.8583 | 0.5345 | 0.9606 |
| Wine | MLP | 0.9811 | 0.9806 | 0.9810 | 0.0573 | 0.9991 |
| Wine | XGBoost | 0.9811 | 0.9841 | 0.9819 | 0.0710 | 0.9998 |
| Banknote Authentication | TabUBase frozen | 0.8269 | 0.8178 | 0.8189 | 0.6737 | 0.9274 |
| Banknote Authentication | MLP | 1.0000 | 1.0000 | 1.0000 | 0.0020 | 1.0000 |
| Banknote Authentication | XGBoost | 0.9919 | 0.9922 | 0.9918 | 0.0275 | 0.9999 |
| Segment | TabUBase frozen | 0.5232 | 0.5232 | 0.4826 | 0.8755 | 0.8385 |
| Segment | MLP | 0.9779 | 0.9779 | 0.9779 | 0.0562 | 0.9984 |
| Segment | XGBoost | 0.9798 | 0.9798 | 0.9798 | 0.0299 | 0.9994 |
| Spambase | TabUBase frozen | 0.6687 | 0.5808 | 0.5030 | 0.8864 | 0.8180 |
| Spambase | MLP | 0.9411 | 0.9362 | 0.9380 | 0.6201 | 0.9774 |
| Spambase | XGBoost | 0.9524 | 0.9490 | 0.9501 | 0.1851 | 0.9882 |

Digits remains the clearest hard failure. Checkpoint variation is also material:
Wine accuracy is 0.7484, 0.9686, and 0.9308 for seeds 1729, 2718, and 31415;
Digits is 0.1200, 0.3061, and 0.1323. On Spambase, seeds 2718 and 31415 both
remain at 0.6058 accuracy, while seed 1729 reaches 0.7944.

## Regression metrics

Lower error is better; higher $R^2$ is better.

| Dataset | Model | RMSE | MAE | Scaled RMSE | Scaled MAE | $R^2$ |
|---|---|---:|---:|---:|---:|---:|
| California Housing | TabUBase frozen | 0.8880 | 0.6868 | 0.7710 | 0.5963 | 0.4108 |
| California Housing | MLP | 0.5261 | 0.3504 | 0.4567 | 0.3042 | 0.7939 |
| California Housing | XGBoost | 0.4543 | 0.3010 | 0.3944 | 0.2613 | 0.8463 |
| Diabetes | TabUBase frozen | 55.8888 | 45.4344 | 0.7208 | 0.5859 | 0.4518 |
| Diabetes | MLP | 73.0113 | 56.1961 | 0.9411 | 0.7243 | 0.0557 |
| Diabetes | XGBoost | 59.3925 | 47.2236 | 0.7658 | 0.6087 | 0.3774 |
| Airfoil Self Noise | TabUBase frozen | 5.1560 | 3.9802 | 0.7484 | 0.5777 | 0.4372 |
| Airfoil Self Noise | MLP | 1.9174 | 1.3110 | 0.2782 | 0.1902 | 0.9221 |
| Airfoil Self Noise | XGBoost | 1.7859 | 1.1792 | 0.2592 | 0.1711 | 0.9326 |
| Concrete Compressive Strength | TabUBase frozen | 10.6669 | 8.4690 | 0.6373 | 0.5060 | 0.5766 |
| Concrete Compressive Strength | MLP | 5.3123 | 3.6845 | 0.3172 | 0.2201 | 0.8952 |
| Concrete Compressive Strength | XGBoost | 4.2444 | 2.7851 | 0.2534 | 0.1664 | 0.9332 |
| QSAR Fish Toxicity | TabUBase frozen | 0.9841 | 0.7383 | 0.6861 | 0.5147 | 0.5678 |
| QSAR Fish Toxicity | MLP | 1.0620 | 0.7153 | 0.7399 | 0.4983 | 0.4929 |
| QSAR Fish Toxicity | XGBoost | 0.9016 | 0.6384 | 0.6282 | 0.4447 | 0.6364 |

Diabetes is the one dataset where TabUBase has both lower pooled RMSE and higher
pooled $R^2$ than both fitted baselines. Its MAE is also lower than both. QSAR
Fish Toxicity has better RMSE and $R^2$ than MLP, while XGBoost remains ahead.

## Direct effect relative to expanded `v4`

The `v4` and long-context receipts use identical dataset hashes, checkpoint and
split seed panels, complete-context/query policies, and split manifests. The
table is descriptive; `primary delta` is long-context minus `v4` accuracy or
$R^2$. `Loss gain` is `v4` normalized NLL/scaled RMSE minus long-context loss,
so positive is better.

| Dataset | Primary metric | Expanded `v4` | Long-context | Primary delta | Loss gain |
|---|---|---:|---:|---:|---:|
| Breast Cancer | Accuracy | 0.8817 | 0.8317 | -0.0500 | -0.0933 |
| Digits | Accuracy | 0.1736 | 0.1861 | +0.0126 | +0.0036 |
| Iris | Accuracy | 0.8346 | 0.8519 | +0.0173 | -0.0440 |
| Wine | Accuracy | 0.8512 | 0.8826 | +0.0314 | -0.0543 |
| Banknote Authentication | Accuracy | 0.7589 | 0.8269 | +0.0680 | +0.0628 |
| Segment | Accuracy | 0.4662 | 0.5232 | +0.0569 | -0.0147 |
| Spambase | Accuracy | 0.6349 | 0.6687 | +0.0337 | +0.0514 |
| California Housing | $R^2$ | 0.3829 | 0.4108 | +0.0279 | +0.0191 |
| Diabetes | $R^2$ | 0.4031 | 0.4518 | +0.0487 | +0.0316 |
| Airfoil Self Noise | $R^2$ | 0.2413 | 0.4372 | +0.1960 | +0.1239 |
| Concrete Compressive Strength | $R^2$ | 0.5767 | 0.5766 | -0.0002 | +0.0022 |
| QSAR Fish Toxicity | $R^2$ | 0.5326 | 0.5678 | +0.0353 | +0.0276 |

Long-context improves all five regression scaled-RMSE values, but does not
produce a uniform classification likelihood improvement. That distinction is
more informative than the single statement that 10/12 decision metrics improve.

Relative to the still earlier original PT-S1 line, long-context improves pooled
accuracy or $R^2$ and the corresponding primary loss on all 12 datasets. That
larger gain combines expanded-law `v4` and long-context effects and therefore
cannot be attributed to context extension alone.

## Real-data mechanism controls

The pooled pretrained-minus-random and pretrained-minus-label-shuffled direction
is favorable on all 12 datasets. Eleven datasets have strict primary-metric wins
on all 9 checkpoint-by-split pairs against both controls. Spambase has a positive
pooled gain but only 3/9 strict accuracy wins against each control. This is real
optimizer-free context use, but still checkpoint-sensitive rather than universal.

## Runtime and provenance

The live workspace and Git top level resolved to the same historical donor
workspace; its machine-local path is intentionally omitted.
The inspected branch is `codex/tabubase-eval-chain` at commit
`3502fdd80539f2a8b9703cc4e4546fd01f3826ce`. The pre-existing dirty worktree was
preserved; no unrelated file was reset or cleaned.

The `dgx2` alias resolves to physical host `spark-b5b3`. The accepted runs used
isolated execution ID `tabubase-expanded-synthetic-long-context-v1`, runtime
image `wehub/ml-gpu:20260712`, Python 3.12.3, torch
`2.12.0.dev20260322+cu130`, CUDA 13.0, and an offline pinned
scikit-learn 1.8.0 wheel. The wheel SHA-256 is
`4496bb2cf7a43ce1a2d7524a79e40bc5da45cf598dbf9545b7e8316ccba47bb4`;
the installed dependency-tree SHA-256 is
`8448de978bf54fd5ae56c2c44a394606e8aeb6a123703706c37826e92a98341c`.

Failed infrastructure/provenance attempts are retained in the remote log tree.
They include a missing pytest module in the production image, missing sklearn in
the bare ML layer, an omitted California cache mount, a duplicated checkpoint
suffix, and a non-semantic OpenML cache-path mismatch. None produced an accepted
model result. The final new6 receipt uses the original absolute cache path so
the strict source-manifest hashes match the baseline without weakening the
comparator.

The post-run audit found no TabUBase or GPU compute process on dgx2.
Final local validation passes 58/58 focused tests and ruff on every changed
Python file.

## Artifact index

The machine-readable artifact ID
`expanded-synthetic-long-context-3checkpoints-manifest` is registered in the
[portable local-artifact index](local-artifact-index.json), is not bundled with
this repository, and has SHA-256
`d8dee96db081e4b5aa8597069f9c18d6c6d3aacdfdcdbdfdab4338a7a72cd0ac`.

| Artifact | SHA-256 |
|---|---|
| PT-S1 seed 1729 result | `7ab120e97c0b3e84d0f533d9a1f68f940650d5110c120fdc43d7ec127fe3796f` |
| PT-S1 seed 2718 result | `a6c1a73decf65eb9e62b3142f3c2b2d02fbafeef732fa8e32779109d12a4eb19` |
| PT-S1 seed 31415 result | `db2ac63e4f32d038f6f77e62e162de49b5df1650355fc9f1d0b25f1ca02f4b45` |
| Synthetic three-checkpoint aggregate | `02cdb7a4de12ff7ca87048798287304f05af3f6e12e6db1ac42af666e563f065` |
| Real full-context old6 | `ce632d106748b727961b4866b019ffd0ccbc3c0bf4a34fdbc28a5ea144b46520` |
| Real full-context new6 | `20cda7c90ff7d057e92ef9c671bc9018abf62d9d877239462cc377b71cec98cf` |
| Exact-split baseline comparison old6 | `2d0f8e7c5c48a58692024a3b1d8dc0128c78e2e2229be5f5052faea8bada9441` |
| Exact-split baseline comparison new6 | `1ea42f6074cc2d63fd7cbb05dcff51dd784467c2bf8338c4f4da1d6db6004d3f` |

The frozen preregistration is
[`expanded-synthetic-long-context-v1.yaml`](../../experiments/transfer-base-v2/expanded-synthetic-long-context-v1.yaml).
It remains the pre-run registration artifact; this report does not rewrite its
historical `not_run` status after observing results.

## Recommended next gate

Do not launch PT-S2 merely because the bounded run passed. The next smallest
discriminating sequence is:

1. preregister and run a held-out synthetic nested-context curve through
   $K=64,128,256,512$, including label shuffle and context-tail removal;
2. add a classification calibration-focused pretraining ablation while keeping
   downstream weights frozen and without fitting a downstream temperature;
3. close CUDA runtime gates at $K=1024$ and $2048$ before considering those
   lengths in training;
4. decide separately whether to extend to $K=4096,8192$ or test a model-capacity
   variant, because an architecture change cannot inherit this checkpoint line.

PT-S2 or an order-of-magnitude world/update expansion should be considered only
after actual long-context utilization and the classification NLL regression are
resolved under a new preregistration.

## Skillify checkpoint

This workflow is repeatable, changes durable experiment state, and benefits from
fewer future prompts. The smallest durable assets are project-specific rather
than a new general skill: the preregistration, runtime audit script, fail-closed
comparators, focused tests, artifact manifest, and this report already encode the
sequence. The next trigger is any request to extend training beyond $K=512$ or to
rerun full-context frozen ICL; the agent should reuse these gates and create a
new isolated run root rather than reconstructing the protocol from chat history.
