# TabUR one-shot optimization execution guide — 2026-08-30

Status: one-shot implementation and experiment guidance  
Evidence status: `none` (`planning_only`)  
Applies to: `tabu.query.row@0.1.0`, `supervised.label_broadcast.v1`  
Baseline source: commit `709206987562e9920656957823f22464da156999` and
[`tabur-evaluation-ladder-2026-08-30.md`](../reports/tabur-evaluation-ladder-2026-08-30.md)  
Primary comparison target: TabUBase cached OpenML regression-8 protocol  
Retirement rule: mark this guide `completed` or `superseded` in place after the final
corrected comparison report is written. Do not move or delete it, because the final
report must preserve a stable back-link. Do not turn it into a permanent model
contract by accident.

This document tells the next implementing agent what to change, what to run, and when
to stop. It is not a preregistration, receipt, benchmark, accepted capability claim, or
authorization to rewrite existing results. Existing negative results remain evidence
and must not be overwritten.

## 中文执行摘要

这份文件只服务本轮 TabUR 优化，核心顺序是：

1. 保留当前 Stage 5/6 为历史 bounded diagnostic，不覆盖旧结果；
2. 先修 real-regression 的 context/task/raw 三层坐标，再修 exact same-init
   scratch 对照；
3. 用旧 prior 只跑 validation-side 的 `0/20/100/400` update 诊断，不提前查看
   OpenML8 final test；
4. 建立真正按 world 参数变化的 regression synthetic v2，覆盖多宽度、多 law 与
   variable-$K$；分类作为独立后续 lane，不能继续用 numeric-only checkpoint 解释
   Iris；
5. 依次跑 `512/1500 → 2048/6000 → 20000/20000`，只有 held-out synthetic 和
   real validation 同时过门才进入下一档；`100000/100000` 只是可选扩容；
6. 最终分别交付 full-train-context frozen ICL 表与 128-label/400-update fine-tune
   表，两表都含 Train mean、MLP、XGBoost，并保留各自不同的 split/budget 标签；
7. 所有 frozen arms 不得创建 optimizer，且逐 arm 记录前后完全相同的 parameter
   hash；所有 fine-tune arms 则记录 root、checkpoint、start 和 final hashes；
8. correctness、same-init、exact-resume、validation 或 hash 任一硬门失败，就停止扩大
   训练并保留负结果。

## 1. Optimization question

The one-shot objective is to answer one narrow question:

> After fixing the real-regression prediction coordinates and using an exact
> same-initialization control, can profile-compatible synthetic pretraining provide a
> reproducible frozen-ICL or fine-tuning benefit for TabUR on held-out synthetic worlds
> and the same eight real OpenML regression datasets used by TabUBase?

The required deliverable is not merely a lower training loss. It is a corrected,
paired comparison containing:

1. a frozen-weight ICL lane using every downstream train row as labeled context;
2. a separate real-task fine-tuning lane;
3. exact same-split MLP and XGBoost descriptive baselines;
4. per-arm source, checkpoint, parameter, data, split, and evaluator hashes;
5. all failures and negative transfer results.

Do not claim that TabUR is better or worse than TabUBase until this protocol has been
run. The architectures and current experiment budgets are not matched.

## 2. Why the current Stage 6 result is not an architecture verdict

The current large Stage 6 result is:

| Dataset | Scratch | Pretrained | Gain, scratch minus pretrained |
|---|---:|---:|---:|
| Iris log loss | 0.966266 | 24.029102 | -23.062836 |
| Diabetes RMSE | 69.360475 | 79.190333 | -9.829858 |

It should be retained, but interpreted as a failed bounded diagnostic. Four material
confounds remain:

- the synthetic supervised generator has four numeric predictors, a numeric response,
  fixed coefficients, three simple response laws, and fixed context size 12;
- Iris is a classification task, but the generator contains no classification worlds;
- downstream adaptation uses only 64 labels, 20 updates, one result per task, and at
  most 64 test rows;
- the scratch arm does not start from the same initial parameter state as the
  pretraining arm.

There is also a correctness issue in the regression path. The query model's public
`numeric` output is context-standardized, while the current real-task truth and metric
path treat it as if it were already in the task-level standardized coordinate. The
Diabetes RMSE therefore cannot be used as a clean transfer verdict until Gate R1 below
passes.

Stage 5 does show a useful synthetic frozen-ICL mechanism, but it uses a separate
`completion.artificial_mask.v1` checkpoint. It does not validate the supervised Stage
6 checkpoint.

## 3. Non-negotiable experiment semantics

### 3.1 Keep frozen ICL and fine-tuning separate

The two lanes answer different questions and must never share an arm label.

| Lane | Optimizer | Parameter update | Real labeled evidence | Required hash gate |
|---|---|---|---|---|
| `real_full_context_frozen_icl` | forbidden | forbidden | all train-partition rows | before hash equals after hash for every arm |
| `real_finetune_transfer` | required | expected | fixed 128-row label subset | initial, checkpoint, and final hashes all recorded |

For the frozen lane, the primary downstream episode for split $s$ is

$$
E_s=
\{(x_i,y_i):i\in T_s\}
\cup
\{(x_j,\bot):j\in Q_s\},
$$

where $T_s$ is the complete train partition and $Q_s$ is the complete held-out test
partition. Query-response readout may be chunked, but chunking must not truncate or
change $T_s$, omit held-out rows, create an optimizer, or update a parameter.

If the runtime cannot consume all of $T_s$, record the dataset/split as `blocked`.
Falling back silently to $K\leq64$ is forbidden.

### 3.2 Preserve truth and split authority

- Split before fitting any statistic, tokenizer state, imputer, or codebook.
- Query response truth exists only in the loss/scoring sidecar.
- MLP, XGBoost, pretrained TabUR, scratch TabUR, and frozen controls use the same
  feature and split manifest hashes.
- Classification and regression results stay separate; do not average them into a
  foundation-model score.
- Existing receipts are immutable. Corrections create new artifacts that link to the
  superseded diagnostic.

### 3.3 Separate execution success from capability success

Replace the overloaded Stage 6 `pass` meaning with two fields:

```text
execution_status = succeeded | failed | killed | blocked
capability_gate = passed | failed | not_applicable
```

Finite records mean only `execution_status=succeeded`. They do not imply positive
pretraining lift.

## 4. Ordered execution plan

Do the gates in order. No expensive GPU pretraining is allowed before R1 and R2 pass
on CPU.

| Gate | Intervention | Main output | Exit condition |
|---|---|---|---|
| R0 | Freeze current baseline and lineage | baseline manifest | old artifacts preserved and hashes recorded |
| R1 | Repair regression coordinates | scale adapter plus tests | train/eval coordinates agree exactly |
| R2 | Repair paired controls and status | exact same-init runner | pairing, hash, and status tests pass |
| R3 | Corrected rerun without expanding the prior | adaptation curve | clean diagnosis at 0/20/100/400 updates |
| R4 | Build a diverse TabUR supervised prior | generator v2 | replay, leakage, coverage, and oracle gates pass |
| R5 | Tune optimizer and scale by bounded rungs | selected checkpoints | held-out synthetic validation improves |
| R6 | Run final real frozen and fine-tune panels | comparison report | full OpenML8 table and receipts complete |

## 5. Gate R0 — preserve the baseline

Before editing behavior:

1. record the scoped Git commit and dirty state;
2. record hashes for the Stage 5 and Stage 6 JSON results and every checkpoint that
   will be reused;
3. copy no result into a new file by hand; consume the original artifact read-only;
4. create a new experiment ID and output root for the corrected run;
5. confirm live accelerator workload before starting or stopping anything.

Exit criteria:

- the current Stage 6 result remains addressable as the uncorrected diagnostic;
- no old result/checkpoint has been modified;
- every planned new result has a collision-resistant output name and refuses
  overwrite.

## 6. Gate R1 — repair the real-regression coordinate contract

### 6.1 Canonical coordinate map

Let the real-task response first be standardized from raw units:

$$
y^{(g)}=\frac{y^{(raw)}-\mu_g}{s_g}.
$$

For one episode, the query model internally standardizes visible context responses:

$$
z=\frac{y^{(g)}-\mu_c}{s_c}.
$$

The public TabUR `numeric` prediction is $\widehat z$. Therefore task-scale and raw
predictions are:

$$
\widehat y^{(g)}=\widehat z s_c+\mu_c,
$$

$$
\widehat y^{(raw)}=\widehat y^{(g)}s_g+\mu_g.
$$

For real fine-tuning, use exactly one of the following explicit contracts:

1. transform truth to $z$ and train against public `numeric`; or
2. use `numeric_raw_prediction` as $\widehat y^{(g)}$ and train against task-scale
   truth $y^{(g)}$.

Use option 2 for the one-shot correction because it keeps the current real-task
`TruthSidecar` in one stable task-level coordinate. Do not silently change the global
`Objective` contract for every TabU model. Add a query-row-specific real regression
adapter or objective boundary.

Evaluation must use `numeric_raw_prediction` and then apply the global inverse exactly
once. Never multiply public `numeric` directly by $s_g$.

### 6.2 Required tests

Add tests that establish all of the following:

- `numeric_raw_prediction == numeric * numeric_context_scale + numeric_context_mean`;
- a constructed oracle prediction yields zero real-task regression error;
- training loss and evaluation metric use the same conversion;
- applying $y'=ay+b$, $a>0$, changes raw predictions consistently while leaving
  scaled error invariant within a declared tolerance;
- context/query disjointness and truth-sidecar isolation remain intact;
- the classification path is byte- or tolerance-equivalent before and after the
  regression correction.

Exit criteria: all targeted tests pass on CPU float32, including at least two different
context subsets whose context means and scales differ. If the old and corrected
Diabetes predictions are identical despite different context statistics, stop and
inspect the test: it is not exercising the bug.

Primary files:

- [`src/tabu_lab/models/query_base.py`](../../src/tabu_lab/models/query_base.py)
- [`src/tabu_lab/experiments/query_row_real_benchmark.py`](../../src/tabu_lab/experiments/query_row_real_benchmark.py)
- [`src/tabu_lab/experiments/query_row_finetune_lift.py`](../../src/tabu_lab/experiments/query_row_finetune_lift.py)
- [`src/tabu_lab/training/objective.py`](../../src/tabu_lab/training/objective.py)

## 7. Gate R2 — exact paired controls, frozen hashes, and result identity

For each root seed $r\in\{1729,2718,31415\}$:

1. initialize one model state $\theta_0^{(r)}$;
2. clone it before any optimizer exists;
3. pretrain one clone to obtain $\theta_{PT}^{(r)}$;
4. initialize the scratch fine-tune arm from the exact bytes of $\theta_0^{(r)}$;
5. initialize the pretrained fine-tune arm from $\theta_{PT}^{(r)}$;
6. give the two arms identical real splits, episode indices, update counts, optimizer
   class, optimizer hyperparameters, and query order.

The receipt must satisfy:

```text
scratch_initial_parameter_sha256 == pretrain_initial_parameter_sha256
pretrained_initial_parameter_sha256 == pretrained_checkpoint_parameter_sha256
scratch_episode_schedule_sha256 == pretrained_episode_schedule_sha256
```

For frozen arms, additionally require:

```text
optimizer_created == false
requires_grad_update_attempted == false
parameter_sha256_before == parameter_sha256_after
```

Record model spec, profile, tokenizer, generator, source tree, checkpoint, dataset,
split, evaluator, package lock, and runtime-version identities. Loading must remain
strict and profile-bound.

Add contract tests that fail under the old `seed + 1000 + offset` scratch behavior.
The ladder check must also fail `capability_gate` when the preregistered lift criterion
is not met, even when all values are finite.

This worktree currently has no installed `tabu-lab` command entrypoint. R2 must add a
minimal `[project.scripts]` entrypoint and contract-tested `tabu-lab tabur optimize`
subcommand that accepts a preregistration path, device, and non-overwriting output
root. `--help`, resolved-config parity, immutable-output refusal, and exit-code
behavior are part of the R2 tests. Existing `python scripts/...` entrypoints remain
smoke/debug helpers only.

Exit criteria: exact same-init and frozen-hash tests pass on two repeated CPU runs, and
the serialized result is byte-equivalent apart from explicitly allow-listed elapsed
time/environment fields.

## 8. Gate R3 — corrected diagnosis using the existing prior

Do not expand synthetic data yet. First determine how much of the negative result came
from evaluator error, initialization noise, or insufficient adaptation.

Use the current supervised checkpoint only if its exact tensor hash and identity
sidecar are available. Otherwise reproduce it under the corrected R2 lineage and label
the new checkpoint as a reproduction, not the old artifact.

Run regression only at this gate:

- root seeds: `1729, 2718, 31415`;
- label budget: 128;
- fine-tune update checkpoints: `0, 20, 100, 400`;
- learning rate: `3e-4`, AdamW, weight decay `1e-4`;
- evaluation: complete validation partition only; final test rows stay sealed;
- datasets: Diabetes first as a smoke, then the fixed development subset
  `cpu_activity, kin8nm, pumadyn32nh, white_wine`;
- comparators: exact same-init scratch, MLP, and XGBoost using the same label rows.

Update 0 is an initialization/frozen diagnostic, not fine-tuning. Report it separately.

Interpretation matrix:

| Observation | Interpretation | Next action |
|---|---|---|
| negative at 20, positive by 400 | under-adaptation dominated | proceed to R4, retain 400 updates |
| negative at all updates and seeds | prior mismatch or capacity issue | proceed to R4, do not scale old prior |
| mixed signs across seeds | initialization variance dominates | retain three seeds; do not summarize one run |
| corrected RMSE changes materially | old evaluator contaminated the result | supersede interpretation, preserve old receipt |
| non-finite or coordinate test failure | evaluator/model error | stop; R4 is forbidden |

R3 has no positive-performance exit requirement. Its exit criterion is a complete,
correct, paired diagnosis. Do not inspect OpenML8 test metrics or use them to change
the generator, optimizer, capacity, or stopping rule.

## 9. Gate R4 — TabUR supervised synthetic prior v2

Reuse the principles in
[`tabubase-expanded-synthetic-pretraining-data.md`](tabubase-expanded-synthetic-pretraining-data.md),
but implement a separately versioned TabUR generator. Do not point two model families
at an ambiguous mutable generator ID.

### 9.1 Regression-first distribution

The first v2 run should target the eight numeric OpenML tasks without memorizing those
tables:

- sample predictor width rather than fixing four columns; cover 4 through 32 with
  log-scale anchors and jitter, while explicitly exercising the panel widths
  `6, 8, 9, 11, 17, 21, 32`;
- sample world-level coefficients, sparsity, noise, scales, tails, interactions,
  thresholds, latent factors, and subgroup laws;
- include sparse additive, sparse DAG/SCM, tree/threshold, latent-factor, polynomial
  interaction, periodic/saturating, and mixture/subgroup families;
- include Gaussian, heavy-tailed, skewed, mixture, quantized, and bounded numeric
  predictor regimes;
- sample response transforms and signal-to-noise ratio;
- permute feature order after generation;
- split family/template and world parameters before generating rows or compiling
  context/query episodes.

Increasing rows from one fixed response law does not create a new world. The manifest
must report empirical coverage, not only `world_count`.

### 9.2 Context curriculum

The fine-tune-transfer prior uses this required core schedule:

```text
short/medium anchors: 8, 16, 32, 64, 128, 256, 512
```

Keep short-context worlds throughout training. A separate frozen-full-context runtime
extension adds anchors `1024, 2048, 4096, 8192` only after query-only chunked execution
matches the dense reference at small $K$. Batch by a fixed target-row/cell budget so
long episodes are not silently under- or overweighted.

Failure at $K>512$ marks the affected frozen full-context dataset/split or the entire
frozen lane `blocked`; it does not invalidate a correct $K\leq512$ generator or block
the fine-tune-transfer lane. Conversely, a successful shape/runtime smoke at long $K$
does not establish context utilization.

### 9.3 Classification is a separate extension

Do not rerun Iris transfer until v2 includes binary/multiclass response worlds and a
stable source-scoped category codebook. When enabled, classification must report log
loss and calibration in addition to accuracy. A numeric-only checkpoint is not a
classification-pretrained checkpoint merely because the profile ID matches.

### 9.4 Generator exits

Before model optimization, require:

- deterministic replay from generator version, root seed, partition, and world ID;
- truth substitution leaves forward evidence unchanged;
- no statistic or category assignment observes query truth;
- declared minimum coverage in every family/width/noise/context bucket;
- known-reference recovery on selected worlds;
- dense/chunked prediction parity at small $K$;
- finite forward/backward and bounded peak memory through the required $K\leq512$
  core curriculum.

Failure of a core generator exit blocks all pretraining. The separate $K>512$ runtime
exit blocks only the frozen-full-context extension. Do not repair coverage by renaming
duplicated row samples as new worlds.

## 10. Gate R5 — optimizer selection and bounded scaling

The legacy `Adam, lr=1e-2` setting remains a control, not the default.

### 10.1 Train/validation selection

Use train worlds for optimization and never select on real test tasks. Screen the
following pretraining learning rates at a bounded pilot budget:

```text
AdamW, weight_decay=1e-4
learning_rate in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2 legacy control}
```

Select by held-out synthetic validation AULC and endpoint loss, subject to finite
gradients and no degradation of the short-context gate. Save the best validation
checkpoint; do not automatically use the final training step.

### 10.2 Scaling rungs

Run only one rung at a time:

| Rung | Worlds | Updates | Purpose |
|---|---:|---:|---|
| B0 | 512 | 1,500 | optimizer and correctness screen |
| B1 | 2,048 | 6,000 | compare directly with the old budget |
| B2 | 20,000 | 20,000 | three-seed main pretraining candidate |
| B3 | 100,000 | 100,000 | optional; only after B2 has positive held-out scaling slope |

Each rung requires three root seeds for a promoted candidate. B3 is forbidden if B2
does not improve held-out synthetic ICL over B1, if validation regresses, or if any
exact-resume/hash gate fails. B3 also requires that B2 does not regress the fixed R3
real-validation panel relative to B1.

B2 is allowed only when B1 passes the synthetic promotion gate and the corrected
400-update fine-tune protocol shows positive mean exact-scratch gain on at least three
of the four R3 development datasets, with no dataset degrading by more than 10% in
mean scaled RMSE. This development check uses validation rows only; it never opens
test truth. If B1 misses this gate, revise v2 or run the capacity-only ablation instead
of spending the B2 budget.

Required synthetic arms:

- `pretrained_frozen`;
- `same_init_random_frozen`;
- `pretrained_label_shuffled`;
- `pretrained_context_row_shuffled` where legal;
- context-tail removal at long $K$ when evaluating the separate long-context extension.

Primary synthetic promotion gate:

- pretrained aggregate loss is lower than same-init random and shuffled controls;
- the result is positive in at least two of three root seeds;
- a root-seed-to-world hierarchical bootstrap 95% interval for aggregate gain excludes
  zero; a world-only interval may be reported only as conditional on one checkpoint;
- larger context improves tasks constructed to require more evidence;
- short-context performance does not regress beyond the preregistered tolerance.

### 10.3 Capacity is the last ablation

Keep the current 1,947-parameter TabUR configuration fixed through B1. If v2 is
correct and diverse but transfer remains uniformly negative, introduce the following
capacity-only ablation:

| Preset | `d_model` | heads | `d_ff` | blocks | inducing slots | row tokens |
|---|---:|---:|---:|---:|---:|---:|
| tiny/current | 8 | 2 | 16 | 1 | 2 | 4 |
| medium | 16 | 4 | 32 | 2 | 4 | 4 |
| base-scale | 32 | 4 | 64 | 2 | 4 | 4 |

Change only capacity in that ablation; do not simultaneously change generator,
tokenizer, optimizer, or downstream budget. Keep row tokens fixed at four until a
separate row-token ablation is justified.

## 11. Gate R6 — final real-data comparison

Use the fixed cached panel:

```text
white_wine, red_wine, cpu_activity, kin8nm,
pumadyn32nh, energy_efficiency, cars, space_ga
```

The two real lanes intentionally use different existing split authorities and must
emit different split-manifest identities:

- frozen full-context follows the committed
  [`real-full-context-cached-openml-regression-8.yaml`](../../experiments/transfer-base-v2/real-full-context-cached-openml-regression-8.yaml)
  70/30 protocol summarized in
  [`tabubase-cached-openml-regression-8-full-context-2026-08-30.md`](../reports/tabubase-cached-openml-regression-8-full-context-2026-08-30.md);
- fine-tune transfer follows the fixed seed `20260829` 60/20/20 split and 128-label
  protocol summarized in
  [`tabubase-cached-openml-regression-8-finetune-2026-08-30.md`](../reports/tabubase-cached-openml-regression-8-finetune-2026-08-30.md).

Do not relabel either split as the other or reuse one manifest hash for both lanes.

### 11.1 Frozen full-context lane

- checkpoint seeds: `1729, 2718, 31415`;
- split seeds: `1729, 2718, 31415`;
- split: deterministic 70/30 train/test under each split seed;
- context: every train-partition predictor and label;
- query: every held-out test predictor;
- no optimizer and no parameter update;
- TabUR parameter hash before/after every dataset, checkpoint, split, and control arm;
- MLP/XGBoost fit on the same complete train partition as descriptive inductive
  references.
- a train-mean predictor is retained as the minimum regression sanity baseline.

The pretrained TabUR frozen arm must also be compared with same-init random frozen and
label-shuffled frozen controls. MLP/XGBoost are not substitutes for these mechanism
controls. TabUR frozen aggregation has `3 checkpoint seeds × 3 split seeds` units per
dataset; each fitted baseline has three split-seed units and is not replicated by
checkpoint seed.

### 11.2 Fine-tune transfer lane

- root seeds: `1729, 2718, 31415`;
- split: deterministic 60/20/20 using fixed split seed `20260829`;
- root seed controls checkpoint/init and the 128-row subset sampled from the train
  partition; validation supports R3/R5 selection and test remains sealed until R6;
- 128 labeled train rows selected identically for all arms;
- 400 AdamW updates at the frozen selected learning rate;
- exact same-init scratch versus pretrained initialization;
- all held-out test rows;
- MLP/XGBoost fit on the same 128 labeled rows.
- the train-mean sanity baseline uses the same 128 labels.

### 11.3 Required metrics and table

For every dataset and arm report:

- RMSE;
- MAE;
- scaled RMSE / NRMSE;
- scaled MAE;
- held-out $R^2$;
- seed mean and standard deviation;
- paired scratch-minus-pretrained gain and win count.

Define positive transfer consistently as

$$
g_{d,r}=
\operatorname{NRMSE}_{scratch,d,r}
-
\operatorname{NRMSE}_{pretrained,d,r},
$$

so $g_{d,r}>0$ is better for pretrained initialization. Report per-dataset means and
the dataset-to-root-seed hierarchical confidence interval defined below. Do not hide a
negative dataset in one pooled number.

The frozen full-context table must contain at least these rows per dataset:

```text
TabUR pretrained frozen
TabUR same-init random frozen
TabUR pretrained label-shuffled frozen
Train mean full-train
MLP full-train
XGBoost full-train
```

The fine-tune-transfer table must contain at least these rows per dataset:

```text
TabUR pretrained fine-tuned
TabUR exact same-init scratch fine-tuned
Train mean 128-label
MLP 128-label
XGBoost 128-label
```

The two TabUR lanes must remain visibly labeled and use their own aggregation units;
never put a fine-tuned result under a frozen heading or collapse full-train and
128-label baselines into one row.

### 11.4 Exploratory promotion threshold

The corrected TabUR transfer candidate is called promising only if:

- mean fine-tune gain is positive on at least 6 of 8 datasets;
- at least 16 of 24 dataset-seed pairs are wins;
- a dataset-to-root-seed hierarchical bootstrap 95% interval for mean NRMSE gain has a
  positive lower bound;
- no dataset's mean scaled RMSE degrades by more than 10% relative to exact scratch;
- no data, scale, checkpoint, same-init, finite, or hash gate fails.

This threshold authorizes only the wording “promising local transfer candidate.” It
does not authorize SOTA, benchmark, supported-model, foundation-model, or causal
claims. Beating MLP/XGBoost is reported descriptively and is not required to establish
that synthetic pretraining improved TabUR over its own exact scratch control.

## 12. Stop and branch rules

| Evidence | Decision |
|---|---|
| R1 or R2 fails | stop all scaling; fix correctness |
| core $K\leq512$ synthetic frozen ICL fails after v2 | do not run B2/B3; inspect generator/readout |
| only the $K>512$ runtime/context gate fails | block the frozen full-context extension; fine-tune may continue |
| synthetic succeeds, real frozen and fine-tune both fail | generator-to-real mismatch or capacity remains |
| fine-tune succeeds, frozen fails | useful initialization exists; optimizer-free ICL is not established |
| frozen succeeds, MLP/XGBoost remain better | ICL mechanism exists but is not broadly competitive |
| only larger capacity succeeds | record a capacity interaction; do not credit prior alone |
| signs change across seeds | report instability and expand replication before claims |
| B2 does not improve held-out B1 | kill B3; more worlds are not justified |

## 13. Required artifact bundle

Every completed rung should emit one non-overwriting bundle containing:

```text
preregistration.yaml
resolved-config.json
source-identity.json
generator-manifest.json
world-partition-manifest.json
dataset-and-split-manifest.json
checkpoint.safetensors
checkpoint.identity.json
training-metrics.jsonl
validation-metrics.json
per-example-predictions.jsonl
comparison.json
receipt.json
checksums.sha256
```

At minimum, the receipt records:

- exact canonical CLI invocation;
- root seed and derived seed map;
- source, lock/runtime, generator, data, split, model, tokenizer, checkpoint, and
  evaluator hashes;
- optimizer creation flags by lane;
- parameter hashes before and after every frozen arm;
- exact-resume result;
- all pass, fail, kill, and blocked reasons;
- peak memory, elapsed time, and rows/target cells processed.

Evidence-producing commands must use the R2 command entrypoint, for example `uv run
tabu-lab tabur optimize ...`, before a promoted run. The final receipt records the
fully resolved invocation rather than this abbreviated example. Existing ad-hoc
scripts may remain smoke/debug helpers, but their output alone does not become a
formal receipt.

## 14. Completion checklist

The one-shot optimization task is complete only when all boxes are true:

- [ ] Old Stage 5/6 artifacts remain immutable and linked.
- [ ] Regression task/context/raw coordinate tests pass.
- [ ] Scratch is exact same-init, not merely another random seed.
- [ ] Frozen arms create no optimizer and preserve parameter hashes.
- [ ] Execution and capability statuses are separate.
- [ ] Generator v2 has replay, leakage, coverage, and known-reference gates.
- [ ] Checkpoint selection uses held-out synthetic validation.
- [ ] Three PT seeds pass finite and exact-resume checks.
- [ ] Frozen full-context OpenML8 evaluates all train and test rows or records blocked.
- [ ] Fine-tune OpenML8 uses 128 labels, 400 updates, and all test rows.
- [ ] MLP/XGBoost use the exact corresponding label/split manifest.
- [ ] Final report includes every dataset, seed, metric, negative result, and claim
      boundary.
- [ ] Independent non-developer review is complete before any public submission.
- [ ] Gong approval is obtained before PR, merge, release, checkpoint publication, or
      accepted claim.

After completion, write a dated final report under `docs/reports/`, link this guide as
the superseded execution plan, mark it `completed_positive`, `completed_negative`,
`stopped_protocol_failure`, or `stopped_alignment_failure` in place, and stop using it
as an active instruction. Do not move or delete this file.
