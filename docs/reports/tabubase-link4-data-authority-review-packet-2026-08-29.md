# TabUBase Link 4 data-authority review packet

Status: candidate freeze complete; independent authority review pending.

This packet is the handoff for the two Base 0.2.0 real-data datasets. It does
not promote a dataset, issue an evaluation receipt, or authorize a public
claim. The private freeze roots contain raw bytes, labels, and evaluator truth;
they are not committed to this repository.

## Candidate freezes

| dataset | freeze id | freeze manifest SHA-256 | source authority |
| --- | --- | --- | --- |
| OpenML Adult v2 / task 7592 | `eval-data-freeze-6b72d3ab05f0959dc593ad4177af4590b83f9ff64f8e4b93363bb01ac3fdd2d3` | `e0075c9761afb34c280e6ae98ed1c7cf99e3219618c0cf08a6c6bc74c89a9a11` | data file `77aa1703717a29f0b5642e94c3ba1defd2486f0b34d4d8eccc1b37a5f7d226b0`; task split `dac4caf27b44e897f40a4c63c205f4748729db59b6575d4fdfc07d8fa9ebd437` |
| scikit-learn Diabetes 1.9.0 | `eval-data-freeze-88dd7260e5da5a9856f88f7de004a177801b7071dcc944adb24c8517fd247857` | `9449f13b4ae7b69de079ef766477c5cbb74cbf5766d96c787212026a25aa5e46` | raw feature `7fc0ded571454b1982210d3bb43f0aca44eae01a0b8654a3b24022bdb6b38009`; raw target `8e53f65eb811df43c206f3534bb3af0e5fed213bc37ed6ba36310157d6023803` |

Private replay roots for the reviewer are:

- Adult: `/private/tmp/tabubase-link4-authority-20260829/adult-freeze-v1`
- Diabetes: `/private/tmp/tabubase-link4-authority-20260829/diabetes-freeze-v1`

These roots are local retained-data artifacts and are deliberately not copied
into Git or the public catalog.

Adult uses task fold `0`, train-side validation carve seed `1729`, and the
explicit row-id semantics
`openml-task-rowid-zero-based-arff-data-order-v1`. Diabetes uses split seed
`1729`. Both completion masks use mask seed `1729`.

## Registered Base scenario snapshots

All four snapshots have passed the private `prepare` and public metadata-only
`register` checks. They are intentionally `self_consistent_unreviewed`, with
`publication_eligible=false` and no review ids.

| scenario | snapshot | authority-review subject SHA-256 |
| --- | --- | --- |
| `adult-v2-task-7592-classification-micro-base` | `openml-adult-v2-task-7592-adult-v2-task-7592-classification-micro-base-f15a8e858b23d5fc` | `7b9f1ce68f901bad84309c448b81ea5f07c2675d522609093b0af2b3c8b2b4e9` |
| `adult-v2-feature-completion-micro-base` | `openml-adult-v2-task-7592-adult-v2-feature-completion-micro-base-8ecfce76041b0110` | `2c978e5747bd07846cd24c9801844439a6633c5120c799fc07c03dfa2d27f347` |
| `sklearn-diabetes-regression-micro-base` | `sklearn-diabetes-sklearn-diabetes-regression-micro-base-6d91fe17958f67cb` | `9f03d0257ee68f2b5f0c10f2db49cd26cfbd30b61e401d5b1f252586384feecc` |
| `sklearn-diabetes-feature-completion-micro-base` | `sklearn-diabetes-sklearn-diabetes-feature-completion-micro-base-0577f05ca93db62a` | `8ed15731ff5df1d7fb5fb572f023a06255f69db2b797c1b6130b68788edfcd10` |

## Reviewer actions

1. Replay the private freeze integrity checks:

   ```bash
   uv run python scripts/freeze_eval_data_authority.py check \
     --output-root /private/tmp/tabubase-link4-authority-20260829/adult-freeze-v1
   uv run python scripts/freeze_eval_data_authority.py check \
     --output-root /private/tmp/tabubase-link4-authority-20260829/diabetes-freeze-v1
   ```

2. Recompute the two upstream Adult hashes and inspect the retained OpenML
   task split, including exhaustive/disjoint fold assignment and the declared
   zero-based ARFF row-id semantics.
3. Inspect the retained Adult license metadata and the exact copy bound as
   `evidence/adult-license-evidence.bin` in the private freeze.
4. Recompute the Diabetes 1.9.0 raw-file hashes and verify the BSD-3-Clause
   source/version binding.
5. Replay all four request-to-prepared-to-snapshot hashes from the private
   freeze roots; verify the Base v1 suite hashes and train-only preprocessing
   boundary.
6. If correct, create an independent cataloged review record whose subject is
   the exact authority-review subject SHA-256 above. Only then may the
   snapshots be promoted to `reviewed` and Link 4 be rerun formally.

The current diagnostic intentionally reports this as one remaining block:
`independent Adult/Diabetes dataset authority review and promotion are
pending`.
