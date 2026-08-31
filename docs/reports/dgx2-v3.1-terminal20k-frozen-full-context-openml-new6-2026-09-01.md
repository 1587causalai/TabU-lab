# dgx2 v3.1 terminal-20k frozen full-context OpenML `new6`

Date: 2026-09-01  
Host: `dgx2` (`spark-b5b3`)  
Evidence status: `local_unissued`

## Result

The preregistered 20k-cumulative terminal checkpoints for TabUBase and
TabUR/query-row do **not** beat the matched Linear models on the frozen OpenML
`new6` panel. Both arms have zero per-dataset wins and fail the panel success
rule.

Lower is better for every value below. Classification uses normalized NLL;
regression uses target-scaled RMSE. Each dataset value is the mean of frozen
split seeds `1729`, `2718`, and `31415`.

| Dataset | Task | Linear | TabUBase 20k | TabUR/row 20k |
|---|---|---:|---:|---:|
| Banknote Authentication | classification | 0.030436 | 0.990152 | 1.230342 |
| Segment | classification | 0.079826 | 1.000343 | 1.098144 |
| Spambase | classification | 0.329737 | 0.970798 | 1.096131 |
| Airfoil Self Noise | regression | 0.701922 | 1.017703 | 1.025842 |
| Concrete Compressive Strength | regression | 0.627959 | 1.012417 | 1.011664 |
| QSAR Fish Toxicity | regression | 0.680324 | 1.033038 | 1.039791 |

Task-level dataset macros are:

| Task | Linear | TabUBase 20k | Delta vs Linear | TabUR/row 20k | Delta vs Linear |
|---|---:|---:|---:|---:|---:|
| Classification normalized NLL | 0.146666 | 0.987098 | +0.840431 | 1.141539 | +0.994873 |
| Regression scaled RMSE | 0.670068 | 1.021053 | +0.350984 | 1.025766 | +0.355697 |

The frozen panel success rule requires an arm to beat Linear on both task
macros. `tabu_base=false` and `tabu_row=false`.

## Change from the earlier frozen checkpoint panel

The comparator is the prior panel receipt at
`/home/cms/tabubase-runs/20260901-v3.1-frozen-full-context-new6-v1/receipts/panel.json`.
Negative deltas are improvements because lower is better.

| Task | Arm | Earlier checkpoint | Terminal 20k | Delta |
|---|---|---:|---:|---:|
| Classification | TabUBase | 0.988878 | 0.987098 | -0.001781 |
| Classification | TabUR/row | 0.987050 | 1.141539 | +0.154488 |
| Regression | TabUBase | 1.088041 | 1.021053 | -0.066988 |
| Regression | TabUR/row | 1.062473 | 1.025766 | -0.036708 |

Continued training therefore improved TabUBase regression and improved
TabUR/row regression in aggregate, while leaving TabUBase classification almost
flat and substantially degrading TabUR/row classification. This is a result
about these fixed terminal checkpoints and this frozen panel, not a broad model
family claim.

## Frozen execution identity

- accepted run root:
  `/home/cms/tabubase-runs/20260901-v3.1-terminal20k-frozen-full-context-new6-v6`;
- request hash:
  `205431164c77d58ff43f926e14bafe689946545c38163f95e3f91e217c54c850`;
- data panel hash:
  `050eb99f88c540082e3bea7b54aaa3ea47bb60f3fcfad10b22abe9079b975380`;
- evaluation source revision:
  `84db835b46b87226581f38ccea7d6ffc8fbd29ae`;
- evaluation source archive SHA-256:
  `9ff2aedd61c8a6a97c6f53ad690a7c287482ac154017582ba50100b0a73f82a9`;
- panel canonical receipt hash:
  `93522fde307b81ad6c8e99217734c1cd57686622ff76e21e46d404611e3528e9`;
- panel file SHA-256:
  `7b21a8b0e4b03759861c188984f4e0a5989ee1a51016a97d05f54f2c030dbd3c`.

The checkpoint filenames contain the continuation-local step `18500`; each
checkpoint has 20k cumulative training updates after the frozen warm-start.

| Arm | Checkpoint SHA-256 | Snapshot hash | Terminal selection receipt hash |
|---|---|---|---|
| TabUBase | `7344d8f52e84c22d5fb26f84fd904b30339007e02b4056d8b035a60ed3349f16` | `c5d71f1768c6dee70b121cea3ac246553ed4df59ba9789c4a479ef46620bd330` | `363be1dfae3bd170644aa9fbc27b141b1cf60e2b4e3f20288db403a35dec5223` |
| TabUR/row | `1dbf1ef08010b440064b170624539c1b2b51e780cb071ba3873ba80cdb3d5ea3` | `7eaf865dc1d009fd1506be123d26179ed25e5a143e2b49ebd2fab21d91b59b49` | `4fe23411844352320f95e04d82101f258328c653269ad22dc2253678384e430a` |

The exact offline OpenML cache is
`/home/cms/tabubase-eval/20260830-openml-new6-eval-v1/data-cache`.
All six pinned data IDs load from that cache. The runtime used Python 3.12.3,
torch `2.12.0.dev20260322+cu130`, CUDA 13.0, scikit-learn 1.8.0, and scipy
1.18.0.

## Conformance audit

All six dataset receipts satisfy the frozen-evaluation checks:

- both arms use the expected terminal checkpoint SHA-256 values;
- `optimizer_created=false` and `trainable_parameter_count=0`;
- model-state hashes are unchanged before and after inference;
- truth-substitution predictions are unchanged on the checked split;
- every receipt binds the same request hash and data cache;
- the controller and evaluation processes exited after aggregation.

Canonical dataset receipt hashes are:

| Dataset | Receipt hash |
|---|---|
| Banknote Authentication | `c49d18c54fdfec65f0742ac3cbcc1800dce52306c4a4ac6798c93407052b81a7` |
| Segment | `42e82ceb9c0660e8539108ca2194dc6aab17a07e3272f6efe51d1d123cbf17b9` |
| Spambase | `6f3672d5fb1e956b744a30357784abbb12455df786e2ab646aa9d7e6f3b7a107` |
| Airfoil Self Noise | `72167a0d30d69e5cedeef9c3c9565ba9c9212a7397d049084eb0159d18e7fb40` |
| Concrete Compressive Strength | `ecdbffda9c9d9dc50409388f70a08f16c10c96d9b24e5e7658450d7b4d3b7698` |
| QSAR Fish Toxicity | `1502098ca9a6aa91e3f6328189f881e6673096a4ff9f2e435860d9fdea1d959f` |

Attempts `v1` through `v5` are retained as failed infrastructure/preflight
history. They produced no dataset receipt: the issues were an incorrect cache
path, an incorrect repository-root argument, absent scikit-learn in the bare
GPU image, and shell argument expansion. Only `v6` is the complete panel run.

## Claim boundary

This is an optimizer-free Grow-lane comparison of preregistered terminal
checkpoints. It remains `local_unissued`; it is neither formal evidence nor an
accepted capability claim.
