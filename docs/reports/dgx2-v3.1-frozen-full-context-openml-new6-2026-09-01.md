# TabU v3.1 frozen full-context OpenML `new6` evaluation

## Status and claim boundary

- Status: `local_unissued`.
- This is an optimizer-free Grow-lane diagnostic, not formal evidence or an
  accepted capability claim.
- Question: whether either selected current checkpoint beats a matched Linear
  model under frozen full-train-context ICL on all six pinned OpenML datasets.
- Answer under the preregistered primary metrics: **no**. Neither checkpoint
  wins any of the six datasets against Linear.

## Bound identities

- Host: `dgx2` (`spark-b5b3`), direct SSH execution; no agent or bot surface.
- Runtime: `wehub/ml-gpu:20260712`, Python 3.12.3, PyTorch
  `2.12.0.dev20260322+cu130`, scikit-learn 1.8.0, NVIDIA GB10.
- Evaluation source revision:
  `f624011943e01b9085e191065677937cd6c79bbb`.
- Evaluation source archive SHA-256:
  `f320b2bea5c82944e5f12f6ba1d9b559be17d6f683c4a5988f1b7242345aa7eb`.
- Evolution repository hash:
  `64336ec9dc4c5ba8021964f09765cd8c2de7934b4aa0deacfcd2aa05954760af`.
- Data panel hash:
  `050eb99f88c540082e3bea7b54aaa3ea47bb60f3fcfad10b22abe9079b975380`.
- TabUBase: `tabu.pretraining.query-base@1.3.0`,
  `tabu.query.base@0.1.0`, selected step 500, checkpoint SHA-256
  `76989a6282d5e56e7e32ed9974a0f82c64457cad4f24bec4d5080dc663d01c01`.
- TabUR: `tabu.pretraining.query-row@1.3.0`,
  `tabu.query.row@0.2.0`, selected step 1500, checkpoint SHA-256
  `b1919cc959f9358794d886580c1916a99838982882caf2d891da668be15f7ac4`.

## Frozen protocol

- Split seeds: 1729, 2718, and 31415.
- Each split exposes the complete 70% train partition as labeled context and
  scores every row in the held-out 30% partition.
- Feature selection, when needed, is train-only variance selection. No `new6`
  dataset exceeds the 63-predictor bound.
- Classification Linear arm: scikit-learn `LogisticRegression(C=1,
  max_iter=500, solver="lbfgs")`, no scaling.
- Regression Linear arm: scikit-learn `Ridge(alpha=1e-4)`, no scaling.
- Each dataset has its own non-overwriting receipt. The six receipts are bound
  into one aggregate panel receipt only after all identities and split coverage
  replay successfully.
- Both neural arms are frozen: no optimizer is created, trainable parameter
  count is zero, model-state hashes are unchanged, and changing
  `TruthSidecar` does not change forward predictions.

## Primary results

Lower is better for both metrics.

| Task macro | Metric | Linear | TabUBase | Base delta | TabUR | Row delta | Wins vs Linear |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classification | normalized NLL | 0.146666 | 0.988878 | +0.842212 | 0.987050 | +0.840384 | Base 0/3; Row 0/3 |
| Regression | scaled RMSE | 0.670068 | 1.088041 | +0.417972 | 1.062473 | +0.392405 | Base 0/3; Row 0/3 |

| Dataset | Task metric | Linear | TabUBase | TabUR |
|---|---|---:|---:|---:|
| banknote_authentication | normalized NLL | 0.030436 | 0.994282 | 0.991819 |
| segment | normalized NLL | 0.079826 | 1.000540 | 1.000506 |
| spambase | normalized NLL | 0.329737 | 0.971813 | 0.968827 |
| airfoil_self_noise | scaled RMSE | 0.701922 | 1.022194 | 1.017402 |
| concrete_compressive_strength | scaled RMSE | 0.627959 | 1.128117 | 1.081689 |
| qsar_fish_toxicity | scaled RMSE | 0.680324 | 1.113811 | 1.088328 |

Secondary metrics do not reverse the conclusion: Linear has higher mean
accuracy on all three classification datasets and higher mean $R^2$ on all
three regression datasets. TabUR is slightly better than TabUBase on all six
primary comparisons, but it remains behind Linear on every dataset.

The receipts preserve estimator warnings. `segment` and `spambase` reach the
frozen LogisticRegression 500-iteration limit, and `airfoil_self_noise` emits
an ill-conditioned Ridge solve warning. These are weaknesses of the Linear
arm, not advantages; because even these frozen, imperfect Linear fits win by a
large margin, they do not weaken the negative answer for the neural arms. A
future stronger scaled/converged Linear baseline should be added as a new
request rather than overwriting this one.

## Interpretation

This result rejects the narrow proposition that the two currently selected
synthetic-pretrained checkpoints already outperform matched Linear models in
frozen full-context real-data ICL. It does not show that the architecture
cannot transfer, and it does not isolate whether the dominant issue is the
synthetic world prior, pretraining scale, response geometry, or calibration.

The next diagnostic should separate representation gain from transfer
mismatch before spending on another large run: add matched random-initialized
frozen siblings to the same receipts, then inspect prediction dispersion and
calibration by task. If pretrained and random-initialized arms are nearly
identical, the pretraining objective/data route is the first suspect; if
pretraining improves substantially but remains below Linear, expand the
synthetic-to-real coverage and calibration tests without changing this frozen
panel.

## Receipts

- Run root:
  `/home/cms/tabubase-runs/20260901-v3.1-frozen-full-context-new6-v1`.
- Aggregate receipt:
  `/home/cms/tabubase-runs/20260901-v3.1-frozen-full-context-new6-v1/receipts/panel.json`.
- Aggregate receipt self-hash:
  `a99f163f609cd8e617da8c2ab2861f06496b8590e2832c62683a636a9c9ec649`.
- Dataset receipts are in the sibling `receipts/` directory; execution logs
  are in `logs/`.
