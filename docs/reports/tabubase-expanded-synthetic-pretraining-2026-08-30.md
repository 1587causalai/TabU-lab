# TabUBase expanded synthetic pretraining and paired frozen ICL

Date: 2026-08-30  
Evidence status: `local_unissued`  
Contract: `tabu.cell.base@0.2.0`  
Profile: `supervised.label_broadcast.v1`  
Generator: `tabubase.expanded-synthetic.v4`  
Tokenizer: `cell-tokenizer.v2`, source-scoped frozen nominal codebook,
`B=100`, seed `1729`

## Real-evaluation correction

The real old6 and OpenML new6 receipts summarized below used only
$K\in\{0,1,2,4,8,16,32\}$ context rows. They are retained as optimizer-free
**low-shot diagnostics**, but they do not measure the downstream full-context
estimand where all train-partition rows are supplied as labeled context.
Consequently, every real-data AULC and $K=32$ statement below is bounded to
low-shot transfer. The synthetic held-out $K$ curves remain valid for their
registered context-length-scaling question.

The replacement full-context protocols are
[`real-full-context-frozen-icl.yaml`](../../experiments/transfer-base-v2/real-full-context-frozen-icl.yaml)
and
[`real-full-context-frozen-icl-openml-new6.yaml`](../../experiments/transfer-base-v2/real-full-context-frozen-icl-openml-new6.yaml).
They have now been executed on all 12 datasets with exact-split MLP/XGBoost
baselines. The successor result is recorded separately in
[`tabubase-real-full-context-frozen-icl-2026-08-30.md`](tabubase-real-full-context-frozen-icl-2026-08-30.md).
The low-shot sections below are retained as their original auxiliary evidence
and are not silently reinterpreted as full-context results.

## Outcome

The bounded `v4` rung passed its generator audit, PT-S0, and three-seed PT-S1
training gates. Each expanded checkpoint was compared directly with the old
variable-$K$ checkpoint carrying the same seed. The paired diagnostics found:

- positive synthetic held-out AULC gain for classification and regression at
  all three seeds, although seed 1729 still has a negative regression gain at
  $K=32$;
- positive real-data **low-shot** macro AULC gain on the existing six-dataset
  panel at all three seeds; 53 of 54 matched dataset-split pairs favor `v4`;
- positive regression **low-shot** macro AULC gain on all three seeds of the
  pinned OpenML `new6` panel, but mixed classification transfer: Banknote is
  stable, Segment is seed-sensitive, and Spambase is mostly negative;
- 18 evaluation receipts with no optimizer and 54 adjacent per-arm parameter
  hash pairs that are exactly unchanged.

This is three-seed, optimizer-free evidence that the expanded synthetic prior
improves the registered synthetic curves and several real-data low-shot
diagnostics. It is not evidence for full-training-set downstream ICL, a uniform
win across datasets or $K$, a broad benchmark, a formal receipt, a public
checkpoint, or a foundation-model claim. Seed 31415 also exposes a
training-stability risk despite passing the current training gates.

## Live workspace and execution audit

The pre-run audit resolved the local project path and Git top level to the same
non-symlinked directory:

`/Users/cms/.openclaw/workspace/projects/causal-superintelligence/TabU/tabu-lab`

The inspected branch was `codex/tabubase-eval-chain` at
`3502fdd80539f2a8b9703cc4e4546fd01f3826ce`. The worktree was already dirty:
219 tracked files were modified and 203 paths were untracked, with no staged,
deleted, or unmerged paths. These changes are not converted into a clean commit
or issued receipt by this report.

No relevant local TabUBase process was running. The `dgx2` host resolved to
`spark-b5b3`; it was reachable, had no TabUBase process, and had no GPU compute
process before the new run. The historical remote root
`/home/cms/tabubase-eval/20260829-codebook-b100-v1` was intact but was not a Git
checkout. It was treated as an artifact projection, not semantic source
authority. The new isolated execution root is:

`/home/cms/tabubase-eval/20260830-expanded-synthetic-v4`

The source snapshot used by the `v4` run has source-tree SHA-256:

`fe66094cea2a35f1e993613991eb4cc39d7ec84bd3859e271ff21dd6fe0bfe83`

The final post-run check again found no TabUBase Python process and no GPU
compute process on `dgx2`; the artifact filesystem had 977 GiB free. The local
worktree remains dirty at the same branch and HEAD, now with 219 tracked dirty
entries and 221 untracked entries, with no staged, deleted, or unmerged entry.

## Failed attempts retained

The path to `v4` was not a clean sequence of promoted runs. Each failed design
or execution attempt remains separate from the passed `v4` evidence.

| Version / attempt | Furthest point reached | Disposition and evidence |
|---|---|---|
| `v1` attempt 1 | PT-S0 reached update 2,000 | Training bytes were written, but final result construction raised `FileNotFoundError` because `git` was absent in the GPU container. This is an artifact/infrastructure failure, not a passed run. Failure receipt SHA-256: `ddaafb1939ca3e8a0d29b2fdcede664ea11f4789627402904f507897eb24f22c`. |
| `v1` attempt 2 | PT-S0 result written | Finite and exact-resume gates passed, but validation worsened from `6.360416` to `6.451323`; validation-improvement failed. Result SHA-256: `41e59e90b36795d63cfa65bc995127ffa35d1e2377af59ad7761b6d770fe3d95`. |
| `v2` | Stage-A audit only | The audit artifact passed its recorded generator gates, but pretraining was stopped before PT-S0. Review found that the categorical terminal cannot predict a declared class absent from context, making four-class $K=2$ structurally unsupported; the preflight also exposed context-schedule/parity coupling and insufficient runner strictness. Audit SHA-256: `7bc489a818f9be0519f37b80e38765b516de4442f929cd6db98c104dcab74406`. |
| `v3` | Stage A and PT-S0 passed; PT-S1 stopped | PT-S0 improved validation from `5.190759` to `5.020015` and passed all three training gates (`817ee8a1d4ed2482eaeead58217104f0c34f8f04c94db142a5d8412f07e22bd2`). PT-S1 failed closed at update 10,736 (`world_index=12194`, ordinal, $K=4$) because a 512-row candidate pool omitted one class. The root cause was a categorical predictor-probability permutation derived from the row-sampling seed, so calibration and episode banks did not share one world-level predictor distribution. No PT-S1 `result.json` exists; checkpoints through update 10,000 are retained. Stage-A SHA-256: `44356e1f0ce12ccbcf66e53a07a4d95e80838afd4e8a54475101bf3da61ae0ff`. |

`v4` moves the categorical predictor-probability permutation to stable
world-parameter state, separates row/noise sampling from response calibration,
uses query-blind class-balanced context selection, and fails closed for
support-impossible modality/$K$ pairs. Numeric and binary responses retain
$K\in\{2,4,8,16,32,64\}$; ordinal and categorical responses use
$K\in\{4,8,16,32,64\}$. Four-class $K=2$ is rejected rather than silently
compiled.

## `v4` generator and training gates

The Stage-A artifact passed `G-D0`, `G-D1`, `G-D2`, and `G-D2U`. It also records
a selected-world `G-D3` diagnostic, but that block explicitly says it is not
included in the Stage-A audit gate. The training-universe audit compiled all
20,000 world/support pairs with zero failures, observed every supported
modality/$K$ pair, and never required a context candidate pool above the frozen
size of 64. The 192 frozen validation worlds expand to 1,056 validation
episodes because each world is evaluated at every support-realizable $K$.

Stage-A audit SHA-256:
`040c98ad25f12e380e33b9e58b9e9da2a63e59d307584c72ce7ca2385a2ed664`.

| Phase | Seed | Worlds / updates | Initial validation | Final validation | Gates | Result SHA-256 |
|---|---:|---:|---:|---:|---|---|
| PT-S0 | 1729 | 2,048 / 2,000 | 13.165560 | 13.058431 | finite, exact-resume, validation-improvement pass | `316ef3968e1c3e2731367d2b8cf8542b696345c3b2bff6e917e0d7346771189a` |
| PT-S1 | 1729 | 20,000 / 20,000 | 13.165560 | 12.694250 | all three pass | `a1a6e77d5bb8e1db90ad84ef1c68fa3882eae38fdeb57dffec409c70d8ada9d1` |
| PT-S1 | 2718 | 20,000 / 20,000 | 12.815995 | 12.211020 | all three pass | `8c535529c287a1edd26c41bce4dc2347ad77938433b82450ceb9368b8fadb7fc` |
| PT-S1 | 31415 | 20,000 / 20,000 | 31.456700 | 31.367804 | all three pass | `00fdb2338ba81262bca3ad0b6dd04c30febc4d6acb57ac71ebb82223e3c48a01` |

The three selected `checkpoint-20000.safetensors` SHA-256 values are:

- seed 1729: `a9069920cfc1c3dbccfff8d65b334014e205e9693916e750de4e12632f765325`;
- seed 2718: `bb99f0605671f6d0318333007a6e4b6cfc2d29eace68131464e469069d904f81`;
- seed 31415: `f5a32ae33aaef018f7b361157723a1b9a12d35ab204b54a3628f3b830a3434fc`.

Seed 31415 passes the current recorded gates only narrowly: validation improves
by about 0.28%, while its recorded training loss reaches a maximum of
`9768.7441`. It is therefore a passed endpoint under the present contract, but
not evidence that the three training trajectories are equally stable.

## Frozen-ICL invariant

Both the old and expanded checkpoints were re-evaluated with the same hardened
evaluator source. Every evaluation creates three independent model arms:
`pretrained_frozen`, `random_init_frozen`, and `pretrained_shuffled`. Evaluation
runs in inference mode with model parameters frozen.

Across 18 unique evaluation receipts (old and expanded checkpoints for three
seeds on synthetic, real-old6, and real-new6 panels):

- the global frozen-arm optimizer flag is `false`;
- no optimizer is created for any frozen arm;
- each receipt records adjacent `before` and `after` parameter hashes for all
  three arms;
- all 54 adjacent per-arm pairs are byte-identical, and every receipt's
  aggregate unchanged gate is `true`.

The old paired checkpoints are the existing variable-$K$ PT-S1 artifacts with
the same seeds. The expanded checkpoints are the `v4` endpoints listed above.
There is no fine-tune arm in any comparison. Unit tests also fail if an
optimizer constructor is invoked inside the frozen evaluator.

## Paired synthetic held-out ICL

Each equal-seed pair uses the same 512 held-out worlds, split into 256
classification and 256 regression worlds, deterministic world index, context
grid $K\in\{0,1,2,4,8,16,32\}$, 32 query rows per world, and 2,000 paired
bootstrap replicates. Positive gain means old loss minus expanded loss.

| Seed | Classification AULC | Classification $K=32$ | Regression AULC | Regression $K=32$ |
|---:|---:|---:|---:|---:|
| 1729 | +0.0287336 | +0.1007996 | +0.0689533 | **-0.0600868** |
| 2718 | +0.0534979 | +0.0006017 | +0.4270519 | +0.3981705 |
| 31415 | +0.0473363 | +0.1565494 | +0.3208819 | +0.3452024 |
| Equal-seed descriptive mean | +0.0431893 | +0.0859836 | +0.2722957 | +0.2277620 |

The across-seed AULC ranges are `[+0.0287336,+0.0534979]` for classification
and `[+0.0689533,+0.4270519]` for regression. These are descriptive equal-seed
summaries, not a pooled estimate or a three-seed confidence interval. Seed 1729
still falsifies a uniform regression endpoint claim at $K=32$.

Paired comparison SHA-256 values are:

- seed 1729: `e76e0c6da82c8dc8c7bfc68e9c1ca4ce33d1e71b390e01bdd1922cfc9653a443`;
- seed 2718: `607a294cf76699e6913808a20eb2f988def62fe05876796500d8c60c48e48f8a`;
- seed 31415: `593bdaba64199b84e5470f133b6308116ade667f84e527d93ff5a04e3fa0b3a7`.

## Paired real low-shot frozen ICL on the existing six datasets

Each equal-seed comparison uses split seeds 1729, 2718, and 31415 on Iris, Wine,
Breast Cancer, Digits, Diabetes, and California Housing. Classification uses
normalized NLL and regression uses scaled RMSE.

| Checkpoint seed | Classification macro AULC gain | Regression macro AULC gain |
|---:|---:|---:|
| 1729 | +0.1122467 | +0.4452193 |
| 2718 | +0.3316897 | +0.2297607 |
| 31415 | +0.4111869 | +0.4014699 |
| Equal-seed descriptive mean | +0.2850411 | +0.3588166 |

Across the nine matched seed-split pairs per dataset, Breast Cancer, California
Housing, Diabetes, Iris, and Wine each have 9/9 positive AULC deltas. Digits has
8/9; its mean is positive at all three checkpoint seeds, but seed 31415 is only
`+0.000701`. Digits also remains weak in absolute terms. For example, the seed
1729 expanded checkpoint reaches only 9.8% to 16.8% accuracy at $K=32$ across
the three splits. Relative loss improvement is not useful ten-class
recognition.

Paired comparison SHA-256 values are:

- seed 1729: `f84a902f4ffa6860731e185b107485d497c26f2c6507449a4a1106e761854812`;
- seed 2718: `228e492f93f30e48a7db2ef0d3909ded303d2dc197f5f598f74df8fa91045604`;
- seed 31415: `f5ea51fcba1dda24fe548138c0f0d2e78a5c7e5d62eda860f15d51a517d89f0c`.

## OpenML `new6` low-shot status

The pinned numeric/no-missing panel in
[`real-frozen-icl-openml-new6.yaml`](../../experiments/transfer-base-v2/real-frozen-icl-openml-new6.yaml)
was materialized and evaluated without changing the old6 estimand. The panel is
Banknote Authentication, Segment, Spambase, Airfoil Self Noise, Concrete
Compressive Strength, and QSAR Fish Toxicity.

The `dgx2` container could not reach OpenML; the first attempt stopped before
any model arm or result receipt. Exact pinned data were then materialized on the
local host, validated against data ID, version, upstream MD5, license, target,
shape, and missingness, transferred as a 24-file cache, and revalidated on
`dgx2`. Key provenance digests are:

- evaluation source-tree SHA-256: `be724be9a405e3da405f3c212fd38ffdd1b63fce808ff4956e4b00f51734f922`;
- checked manifest file SHA-256: `86e31b9a94f09555881aaaf624b5f2c43849d70f1f52b2abf0b24e55114c9f20`;
- canonical manifest payload SHA-256: `9e4749956204e7446394638f9753f0e88aad920e55808c9008fbed304065031f`;
- local/remote cache manifest SHA-256: `ad08146a3422e568453fb8141f99cfbc258b3095ef57901e117fd30bf785cca4`;
- materialization manifest SHA-256: `6690f3515e622923d14dd7e2fff9ff5445344af69e71addab4db8b0cac34efc7`.

| Checkpoint seed | Classification macro AULC gain | Regression macro AULC gain |
|---:|---:|---:|
| 1729 | -0.0057160 | +0.3468559 |
| 2718 | +0.0532633 | +0.3835407 |
| 31415 | +0.0839849 | +0.3534428 |
| Equal-seed descriptive mean | +0.0438441 | +0.3612798 |

The classification range `[-0.0057160,+0.0839849]` crosses zero by seed;
regression is positive at all three seeds with range
`[+0.3468559,+0.3835407]`. Across nine matched seed-split pairs, Banknote and
all three regression datasets are 9/9 positive. Segment is 6/9 positive with
two of three seed means positive. Spambase is 1/9 positive with only one of
three seed means positive. This is expanded real **low-shot** evidence, but not
broad or uniform classification transfer.

Paired comparison SHA-256 values are:

- seed 1729: `f3d4347e7ecbe5944701524c690aff7082b0136ac639e16783905039b480c826`;
- seed 2718: `f3f8a6f12fba9b13aaa405a4dba086add02af81c9310ccd802ffd4301cc62921`;
- seed 31415: `468ab84ba14b496c16ad3ec485837bfe1e495d8367186c9504a5f146897ecdde`.

## Claim boundary and next evidence gates

The evidence in this report supports a bounded statement: across three
equal-seed comparisons, the `v4` expanded synthetic prior improves integrated
frozen-ICL loss on matched synthetic worlds and improves the registered
**low-shot** curves on the old6 real panel and new6 regression panel without
parameter updates. This low-shot result alone does not establish full-context
downstream performance. A separate successor run now supplies all train rows as
context and evaluates every held-out row; it is reported in
[`tabubase-real-full-context-frozen-icl-2026-08-30.md`](tabubase-real-full-context-frozen-icl-2026-08-30.md).
Within the low-shot evidence, synthetic regression at $K=32$ is negative for
seed 1729, Digits remains weak, Segment is seed-sensitive, and Spambase is
mostly negative.

It does not support:

- a generally strong real-tabular or Digits model;
- a uniform per-dataset, per-$K$, or classification transfer result;
- a stable-training conclusion for seed 31415;
- a long-context or order-of-magnitude Stage-E pretraining result;
- a broad benchmark, SOTA, causal-identification, foundation-model, release, or
  formal-receipt claim.

PT-S2 was not started. Before scaling again, the next gate is to diagnose seed
31415's loss spikes and strengthen the training-stability criterion, then
replay the most sensitive classification cases (Segment and Spambase) without
changing the frozen estimand. Independent review, immutable receipt issuance,
and any long-context experiment remain separate later gates.

## Artifact index

- v4 preregistration:
  [`expanded-synthetic-v4.yaml`](../../experiments/transfer-base-v2/expanded-synthetic-v4.yaml)
- v4 world schema:
  [`tabubase-synthetic-world-v4.schema.json`](../../schemas/tabubase-synthetic-world-v4.schema.json)
- generator and audit implementation:
  [`tabubase_expanded_synthetic.py`](../../src/tabu_lab/experiments/tabubase_expanded_synthetic.py),
  [`audit_tabubase_expanded_synthetic.py`](../../scripts/audit_tabubase_expanded_synthetic.py)
- paired comparator:
  [`tabubase_paired_frozen_icl.py`](../../src/tabu_lab/experiments/tabubase_paired_frozen_icl.py),
  [`compare_tabubase_paired_frozen_icl.py`](../../scripts/compare_tabubase_paired_frozen_icl.py)
- frozen evaluators:
  [`tabubase_icl.py`](../../src/tabu_lab/experiments/tabubase_icl.py),
  [`tabubase_real_icl.py`](../../src/tabu_lab/experiments/tabubase_real_icl.py)
- OpenML new6 loader and pinned panel:
  [`tabubase_openml_new6.py`](../../src/tabu_lab/experiments/tabubase_openml_new6.py),
  [`real-frozen-icl-openml-new6.yaml`](../../experiments/transfer-base-v2/real-frozen-icl-openml-new6.yaml)
- replacement real full-context protocols:
  [`real-full-context-frozen-icl.yaml`](../../experiments/transfer-base-v2/real-full-context-frozen-icl.yaml),
  [`real-full-context-frozen-icl-openml-new6.yaml`](../../experiments/transfer-base-v2/real-full-context-frozen-icl-openml-new6.yaml)
- local copy of the three-seed descriptive summary:
  [`.local-runs/tabubase-expanded-synthetic-v4/three-seed/expanded-v4-three-seed-summary-v1.json`](../../.local-runs/tabubase-expanded-synthetic-v4/three-seed/expanded-v4-three-seed-summary-v1.json),
  SHA-256 `40825efb2a9af554780d29ba5ba88cebe942bf530ef889ff2db62489623cf52c`
