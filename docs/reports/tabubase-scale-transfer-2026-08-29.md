# TabUBase 20k synthetic pretraining and real-transfer panel

Date: 2026-08-29
Evidence status: `local_unissued`
Contract: `tabu.cell.base@0.2.0`
Profile: `supervised.label_broadcast.v1`

## Outcome

The 2,048-world PT-S0 gate passed, followed by three independent
20,000-world/20,000-update PT-S1 runs. All three PT-S1 seeds passed finite,
exact-resume, and validation-improvement gates.

The selected 20k checkpoints were then fine-tuned on five real datasets under
one paired 128-label panel. Each dataset used three seeds, a matched scratch
TabUBase arm, XGBoost, and a two-layer MLP. Classification uses log loss as the
primary metric; regression uses response-scaled RMSE (`nrmse`). Lower is better
for both primary metrics. Results are reported per dataset and are not averaged
across tasks.

The first classification projection was invalidated after an evaluator defect
was found: the class-balanced labeled subset was sorted by source row ID and
then truncated to its first 64 rows for inference context. On Wine this produced
context class counts `{0:35, 1:29, 2:0}`. Because the categorical terminal can
only assign probability to context-supported classes, the resulting Wine log
loss was structurally confounded. The corrected run uses the complete labeled
budget as evaluation context and fails closed if any training class is absent.
Training context and query sampling is now stratified, disjoint, and class
covered. The original JSON remains preserved but its classification metrics
must not be interpreted as model evidence.

## Pretraining

| Phase / seed | Worlds | Updates | Initial validation loss | Final validation loss | Gates |
|---|---:|---:|---:|---:|---|
| PT-S0 / 1729 | 2,048 | 2,000 | 1.1675 | 0.6585 | 3/3 pass |
| PT-S1 / 1729 | 20,000 | 20,000 | 1.1675 | 0.3154 | 3/3 pass |
| PT-S1 / 2718 | 20,000 | 20,000 | 1.4306 | 0.3732 | 3/3 pass |
| PT-S1 / 31415 | 20,000 | 20,000 | 1.1062 | 0.3890 | 3/3 pass |

The three PT-S1 endpoint checkpoint hashes are:

- seed 1729: `5b0464750d0896814b76210756b628fa814b1cbcdec436f8f1910678234e2190`
- seed 2718: `b673a143e627c80b942e8ad8d7b2b68c4c3ef647176d5dac6e40d3a1cba59c26`
- seed 31415: `ebe18049e966973618572651489d25ad8d6cf032a3d91ad4237e2fd5cfb46fa3`

### PT-S2 scale extension (in progress)

At the user's request, an independent tenfold extension completed under a new
identity: `PT-S2 = 200,000 worlds / 200,000 updates`. Seed 1729 used checkpoints
at `0/20k/50k/100k/150k/200k`; it required the passed PT-S1 seed-1729 result and
wrote to `tabubase-pt-s2-seed-1729`. On `dgx2` (`spark-b5b3`) with CUDA, the
initial validation loss was `1.1675147` and the final loss was `0.1561162`
(86.6% lower). The finite, exact-resume, and validation-improvement gates all
passed. PT-S1 artifacts remain unchanged. This extension is exploratory and
has no transfer or public-claim status until downstream runs are complete.

The immutable local-unissued receipt ID is `pt-s2-seed-1729-result`
(`sha256:bbf22fda0ad00afd572d7c3172cce484194a0e3bf751139e0c748c47dd2f9c8d`).

### PT-S2 single-seed real-transfer diagnostic

Using the `checkpoint-200000` artifact, a one-seed (`1729`) exploratory panel
ran on five available sklearn datasets with 128 labels and 400 fine-tuning
updates. This is a checkpoint diagnostic, not the formal Adult/Diabetes
three-seed Link 6 panel.

| Dataset | Task | PT-S2 TabUBase | Scratch | XGBoost | MLP |
|---|---|---:|---:|---:|---:|
| Iris | classification log loss | 0.8843 | 0.2733 | 0.4029 | 0.1155 |
| Wine | classification log loss | 0.3852 | 0.0728 | 0.0528 | 0.0168 |
| Breast Cancer | classification log loss | 0.8407 | 0.2388 | 0.1461 | 0.2321 |
| Diabetes | regression NRMSE | 0.8929 | 0.7883 | 0.6220 | 0.7849 |
| California Housing | regression NRMSE | 0.7138 | 0.7165 | 0.6147 | 0.7218 |

The S2 initialization wins only on California Housing in this single seed;
it loses to scratch and classical baselines on the other four datasets. This
does not indicate that PT-S2 failed to fit synthetic data—the PT-S2 gates
passed—but it is an early negative-transfer signal for the current fine-tune
schedule and architecture. The result artifact ID is
`real-panel-pt-s2-seed-1729`
(`sha256:5b23e2a21cea0e1ea76613907818bd7a3dd0969328a54e9072a79f8d26f34494`).

As a schedule diagnostic, the same S2 checkpoint was rerun at the R0-selected
`1e-4` learning rate (still one seed, 128 labels, 400 updates):

| Dataset | PT-S2 at 1e-4 | Scratch | XGBoost | MLP |
|---|---:|---:|---:|---:|
| Iris | 0.6681 | 0.1723 | 0.4029 | 0.1155 |
| Wine | 0.2126 | 0.1855 | 0.0528 | 0.0168 |
| Breast Cancer | 0.5800 | 0.2292 | 0.1461 | 0.2321 |
| Diabetes | 0.7689 | 0.8796 | 0.6220 | 0.7849 |
| California Housing | 0.7289 | 0.8517 | 0.6147 | 0.7218 |

The gentler schedule improves all five PT-S2 values relative to `3e-4`, and
beats scratch on both regression tasks, but still trails XGBoost everywhere
and trails the MLP on all classification tasks. This separates a schedule
effect from the remaining classification transfer limitation. The diagnostic
artifact ID is `real-panel-pt-s2-seed-1729-lr1e-4`
(`sha256:b1265c27316cdc3b9d43e838cae1aae5431a0e52ab20bbebd51e9b00c5f9d7dd`).

### Runtime reduction benchmark

The training path now supports an explicit `emit_trace=False` fast mode. It is
used only inside the exploratory pretraining loop; public/evaluation forward
continues to emit the full truth-free trace and routing diagnostics. On the
same `dgx2` CUDA environment, a 35-update A/B benchmark measured:

| Training path | Mean wall time / update |
|---|---:|
| Full trace and routing diagnostics | 35.7 ms |
| Fast training path (`emit_trace=False`) | **22.2 ms** |

This is a 37.8% per-update reduction, primarily by avoiding per-step SHA-256,
CPU reductions, and CUDA synchronization. A second benchmark including episode
construction measured 22.8 ms/update synchronously versus 24.0 ms/update with
two prefetch threads (one and four workers were also slower). Episode
construction is therefore not the bottleneck for this episode size, and the
long PT-S2 run is launched single-process with prefetch disabled. The current
measured rate implies roughly 75 minutes for 200,000 updates, subject to host
load. Fast mode is content-equivalent for prediction tensors and is covered by
a unit test; its trace omission is recorded in the run identity.

## Real transfer: primary metrics

Three-seed means:

| Dataset | Task | TabU pretrained | TabU scratch | XGBoost | MLP | Pretrained seed wins vs scratch / XGB / MLP |
|---|---|---:|---:|---:|---:|---:|
| Iris | classification log loss | 0.3371 | 0.2419 | 0.4010 | 0.2378 | 1/3 · 2/3 · 1/3 |
| Wine | classification log loss | 0.1060 | 0.0803 | 0.0472 | 0.0122 | 2/3 · 2/3 · 1/3 |
| Breast Cancer | classification log loss | 0.7879 | 0.2155 | 0.1354 | 0.1796 | 0/3 · 0/3 · 0/3 |
| Diabetes | regression NRMSE | 0.8255 | 0.8562 | 0.7049 | 0.8980 | 2/3 · 0/3 · 2/3 |
| California Housing | regression NRMSE | 0.6974 | 0.8561 | 0.5952 | 0.7393 | 3/3 · 0/3 · 2/3 |

Secondary metrics make the modality split easier to see:

| Dataset | Metric | TabU pretrained | TabU scratch | XGBoost | MLP |
|---|---|---:|---:|---:|---:|
| Iris | accuracy | 0.9444 | 0.9444 | 0.9000 | 0.9667 |
| Wine | accuracy | 0.9905 | 0.9524 | 0.9905 | 1.0000 |
| Breast Cancer | accuracy | 0.9159 | 0.9275 | 0.9391 | 0.9333 |
| Diabetes | $R^2$ | 0.2788 | 0.2255 | 0.4756 | 0.1485 |
| California Housing | $R^2$ | 0.4305 | 0.1295 | 0.5858 | 0.3493 |

## Interpretation

1. The run is not a training failure. All three 20k pretraining runs are
   finite, exactly replayable at the probe boundary, and improve held-out
   synthetic validation substantially. The real-data fine-tune runs also
   remain finite and complete their update budgets.
2. The original Wine classification failure was an evaluation failure, not a
   demonstrated model failure. With full class support, pretrained TabUBase
   reaches mean accuracy 0.9905 and mean log loss 0.1060. The remaining gap to
   MLP and XGBoost is much smaller and seed-dependent.
3. Synthetic pretraining provides real transfer on regression. It beats the
   matched scratch arm on 2/3 Diabetes seeds and 3/3 California Housing seeds.
   It also beats the MLP mean on both regression datasets, with 2/3 seed wins
   on each.
4. The corrected classification result is mixed rather than uniformly weak.
   Iris and Wine have strong accuracy, and pretrained TabUBase beats XGBoost on
   mean Iris log loss. It does not beat the best conventional baseline on Wine
   or Breast Cancer. Breast Cancer remains a genuine negative transfer result:
   fine-tuning loss approaches zero while test log loss is 0.7879, indicating
   overconfidence/generalization failure rather than incomplete optimization.
5. Hyperparameters explain part, but not all, of the remaining negative result.
   The validation-only R0 study below selects `lr=1e-4`, 400 updates. This cuts
   pretrained Breast Cancer test log loss from 0.7879 to 0.4380, but the model
   still trails scratch, XGBoost, and MLP. The remaining gap is therefore not
   attributable solely to the original schedule or seed noise.

The evidence therefore supports: TabUBase has sufficient optimization capacity
to fit the synthetic prior, can fit class-covered Iris/Wine episodes, and learns
a transferable regression initialization. It does not support: TabUBase is
already a generally strong real-tabular model, or that scaled synthetic
pretraining makes it superior to boosted trees.

## Episode construction after correction

For labeled row universe $L$, update $t$ samples disjoint, class-covered sets
$C_t,Q_t\subset L$ for context and query. The model-facing evidence is

$$
E_t = \{(x_i,y_i):i\in C_t\}\cup
      \{(x_j,\mathrm{QUERY}):j\in Q_t\},
\qquad C_t\cap Q_t=\varnothing.
$$

Every query predictor $x_j$ is visible, but its response cell is physically
zeroed and marked `QUERY|TARGET`; $y_j$ exists only in the `TruthSidecar`.
Fine-tuning minimizes

$$
\mathcal L_t(\theta)=\frac{1}{|Q_t|}
\sum_{j\in Q_t}\ell\!\left(f_\theta(E_t)_j,y_j\right).
$$

For classification, both $C_t$ and $Q_t$ must contain every response class when
their budgets permit it. Evaluation uses all rows in $L$ as context, matching
the labeled budget used by XGBoost and MLP, and rejects an episode if a class is
unsupported. Train-only feature and response statistics remain estimated from
$L$; validation and test labels never enter tokenization, fine-tuning, or model
evidence.

## Classification R0 validation selection

R0 compared `lr={1e-4,3e-4,1e-3}` at update checkpoints `{400,1200}`. Every
candidate used the same three datasets, three seeds, pretrained/scratch arms,
splits, labeled rows, episode order, optimizer reset, and validation rows. No
test row was evaluated during selection. The objective first averages paired
seed/arm log loss within each dataset and then macro-averages the three dataset
means.

| Learning rate | Updates | Macro validation log loss |
|---:|---:|---:|
| **0.0001** | **400** | **0.4221** |
| 0.0001 | 1200 | 0.6174 |
| 0.0003 | 400 | 0.5928 |
| 0.0003 | 1200 | 0.6958 |
| 0.0010 | 400 | 0.5386 |
| 0.0010 | 1200 | 0.7294 |

The selected global schedule is therefore `lr=1e-4`, 400 updates. Extending to
1200 updates worsens the macro validation objective at every learning rate,
which supports over-training rather than training insufficiency.

Selected-schedule three-seed test means:

| Dataset | TabU pretrained log loss | TabU scratch | XGBoost | MLP | Pretrained accuracy |
|---|---:|---:|---:|---:|---:|
| Iris | 0.4254 | 0.1308 | 0.4010 | 0.2378 | 0.9222 |
| Wine | 0.1069 | 0.2252 | 0.0472 | 0.0122 | 0.9524 |
| Breast Cancer | 0.4380 | 0.2605 | 0.1354 | 0.1796 | 0.8899 |

The gentler schedule materially improves Breast Cancer calibration but does not
remove negative transfer. Wine retains a clear pretrained-vs-scratch advantage;
Iris and Breast Cancer do not. Because these test partitions had already been
opened by the earlier exploratory panel, this is diagnostic `local_unissued`
evidence rather than a sealed one-time test result.

## Artifacts and boundary

- Corrected classification panel:
  `real-panel-classification-corrected-v2`
  (`sha256:4c602faa3b3b099afd231ed351244c0e928dec0ffc13fd299f8b737d3ade2711`)
- R0 validation selection:
  `classification-r0-selection-v1`
  (`sha256:d8d37d7913f214e8167f0602624d14f78c43742250baed7887525f48f1bb27e8`)
- R0-selected classification test:
  `real-panel-classification-r0-selected-v1`
  (`sha256:b9d6b8fb340ed28d9cf057bedb294bc63f8adff20c568145b782d9f26f28d7a5`)
- Original panel retained for regression only:
  `real-panel-400`
  (`sha256:66a722e8ed69f881cc28e3835ba8a74bd96cf8e136b58abc5727954b7fdeecfd`)
- Portable locators: [local-artifact index](local-artifact-index.json)
- Execution ID: `tabubase-scale-v1` (machine-local root omitted)
- Primary host: `dgx2` / `spark-b5b3`, NVIDIA GB10
- Benchmark source-tree hash:
  `8b3dc490cea22c480fd8065b6051e7dcb9ffd4df6842ff194d3dab3b22547b9f`

This panel is exploratory `local_unissued` evidence. It is not a formal Link 5
or Link 6 receipt, a SOTA comparison, a public checkpoint release, or a
foundation-model claim.
