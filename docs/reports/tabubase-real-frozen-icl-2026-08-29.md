# TabUBase real-data low-shot frozen ICL diagnostic

Date: 2026-08-29  
Evidence status: `local_unissued`  
Contract: `tabu.cell.base@0.2.0`  
Profile: `supervised.label_broadcast.v1`

## Correction notice (2026-08-30)

This receipt evaluated only $K\in\{0,1,2,4,8,16,32\}$ labeled rows from each
train partition and limited the held-out query subset. It therefore measures a
**low-shot context-scaling diagnostic**, not the intended downstream
full-context estimand in which every train-partition row is visible as labeled
context. All numerical results and the immutable result SHA-256 below remain
valid for that narrower question, but they support no claim about performance
with the complete downstream training set in context.

The primary replacement protocol is
[`real-full-context-frozen-icl.yaml`](../../experiments/transfer-base-v2/real-full-context-frozen-icl.yaml).
No full-context empirical result existed when this correction was recorded.

## Estimand

This panel asks the narrower question of whether synthetic pretraining creates
optimizer-free **low-shot** ICL on real data. The model receives $K\leq32$
labeled context rows and visible query predictors, then predicts query labels
without any parameter update. None of the three arms creates an optimizer, and
all parameter hashes must be identical before and after evaluation.

This is distinct from the earlier 400-update real-data transfer panel. Those
fine-tuned results cannot answer the frozen-ICL question.

## Design

- Datasets: Iris, Wine, Breast Cancer, Digits, Diabetes, California Housing.
- Checkpoint seeds: 1729, 2718, 31415.
- Real split seeds: 1729, 2718, 31415.
- Context grid: $K\in\{0,1,2,4,8,16,32\}$.
- Arms: `pretrained_frozen`, `random_init_frozen`, and
  `pretrained_shuffled`.
- Classification primary metric: normalized NLL, where 1 is the uniform
  classifier and lower is better.
- Regression primary metric: RMSE divided by the train-split target standard
  deviation; lower is better. $R^2$ is secondary.
- AULC gain is `baseline AULC - pretrained AULC`; positive values favor the
  pretrained checkpoint.
- Classification AULC starts only when the context can contain every declared
  response class. Regression AULC uses $K\geq1$.

The frozen scale checkpoint permits 64 total columns. Digits therefore selects
63 predictors by train-only variance after splitting, then appends the response
column. All other datasets retain every predictor.

## Results

| Dataset | Primary AULC gain vs random (95% CI) | Paired wins | $K=32$ gain | Interpretation |
|---|---:|---:|---:|---|
| Iris | 0.0824 [-0.0277, 0.2040] | 6/9 | 0.1091 | positive mean, seed/split uncertain |
| Wine | 0.1258 [0.0550, 0.2103] | 9/9 | 0.1426 | robust positive frozen ICL |
| Breast Cancer | -0.2254 [-0.4502, 0.0256] | 3/9 | -0.0169 | negative NLL transfer |
| Digits | 0.00119 [0.00033, 0.00223] | 7/9 | 0.00171 | statistically positive but practically near chance |
| Diabetes | -0.0349 [-0.2460, 0.1619] | 5/9 | 0.1213 | unstable low-$K$, positive $K=32$ endpoint |
| California Housing | -0.0629 [-0.2668, 0.1201] | 4/9 | 0.0552 | unstable low-$K$, weak positive endpoint |

At $K=32$, Wine normalized NLL is 0.8446 for pretrained versus 0.9873
for random initialization. Iris is 0.8759 versus 0.9850. Digits is 1.0017
versus 1.0034 and its pretrained accuracy is 0.0959, so the tiny NLL gain is
not useful ten-class prediction.

Breast Cancer illustrates a calibration/ranking split. Pretrained accuracy at
$K=32$ is 0.5608 versus 0.5192 for random initialization, but normalized NLL is
worse, 1.0151 versus 0.9982. The model separates some cases while assigning
probabilities poorly.

For regression, Diabetes reaches scaled RMSE 0.9636 and $R^2=0.0123$ at
$K=32$, versus random-init 1.0850 and $R^2=-0.2511$. California Housing reaches
1.0975 and $R^2=-0.0435$, versus 1.1527 and $R^2=-0.1469$. Both endpoint gains
are real relative improvements, but California remains weak in absolute terms.
The negative full-curve AULC is caused mainly by unstable $K=4$ behavior.

## Shuffled-context control

Normal labels beat shuffled labels across the primary curve for Wine, Iris,
Breast Cancer, and Digits. For Breast Cancer this means the checkpoint uses the
context-label relationship, but the learned geometry is maladapted relative to
random initialization. The regression shuffled-control intervals cross zero,
so real-data regression context use is not yet robust across the complete
low-$K$ curve.

## Conclusion and boundary

The result establishes a real optimizer-free **low-shot** ICL signal on Wine
and a weaker/uncertain signal on Iris. It does not establish downstream
full-context performance. Within this low-shot diagnostic, Digits is
effectively chance, Breast Cancer has negative NLL transfer, and regression is
only encouraging at $K=32$.

XGBoost and MLP are not included in this frozen comparison because fitting
either model on the task is a different estimand. The appropriate next
algorithmic controls are context-only kNN and ridge/logistic learners, reported
separately from the pretrained-vs-random mechanism test.

Remote run root:
`/home/cms/tabubase-eval/20260829-codebook-b100-v1`

Local result:
`.local-runs/tabubase-codebook-b100-v1/real-icl/real-frozen-icl-6datasets-3x3-v2.json`

Result SHA-256:
`7e9e3aa4944f9efdf65211b8034a69377ebae18fb8657bd71699d7c7027c4b09`

Execution source-tree SHA-256:
`f502cc691d74fade0cee2afbaddd246d0b2251227301ef584f663c915bb336b7`
