# TabUBase expanded synthetic pretraining distribution

Status: design proposal v0.1 (`design_open`)
Date: 2026-08-30
Scope: `tabu.cell.base@0.2.0` / `supervised.label_broadcast.v1`
Implementation status: a bounded `v4` Stage-A/PT-S0/three-seed PT-S1 run and a
separate, preregistered `K<=512` query-response-only long-context PT-S1 arm now
exist as `local_unissued` evidence. Both use corrected full-train-context frozen
evaluation on 12 real datasets: every held-out row is evaluated, exact split
manifests are shared with fixed MLP/XGBoost baselines, no optimizer is created,
and before/after hashes are unchanged for every frozen arm. See
[`tabubase-real-full-context-frozen-icl-2026-08-30.md`](../reports/tabubase-real-full-context-frozen-icl-2026-08-30.md)
for the `v4` line and
[`tabubase-expanded-synthetic-long-context-2026-08-30.md`](../reports/tabubase-expanded-synthetic-long-context-2026-08-30.md)
for the long-context arm. Relative to `v4`, long-context training improves
pooled accuracy or $R^2$ on 10 of 12 datasets and all five regression scaled
RMSE values, but worsens normalized NLL on four of seven classification
datasets. It does not close the $K=1024\ldots8192$ runtime or context-utilization
ladder, authorize PT-S2, issue a formal receipt, or create a foundation-model
claim.

## 1. Purpose

This document defines the design space for the next TabUBase synthetic
pretraining distribution. It separates four changes which must not be conflated:

1. more sampled worlds;
2. more diverse world-generating laws;
3. a wider distribution of context sizes;
4. a runtime path that can actually consume long context without materializing a
   dense quadratic routing tensor.

The goal is not to make a foundation-model or benchmark claim. The goal is to
construct a falsifiable pretraining prior for frozen-weight in-context learning
(ICL), including real-table ICL with no optimizer and no parameter update.

This proposal is downstream of the six-link evaluation contract in
[`eval-chain.md`](eval-chain.md). It does not replace a future pretraining
manifest, preregistration, immutable receipt, or independent review.

## 2. Motivation from the current evidence

The current generator uses eight predictors, one response, three fixed score
families, and four response types. Its structural diversity is therefore much
closer to twelve fixed templates with different random samples than to a broad
prior over table-generating mechanisms.

The first fixed-context checkpoint was trained with exactly 64 context labels on
every update. It used the label/context relationship, but its frozen
classification ICL result was worse than random initialization at low $K$. A
separate intervention changed only the context curriculum to

$$
(2,4,8,16,32,64),
$$

and restored positive aggregate classification and regression ICL signals. The
diagnostic is recorded in
[`tabubase-nominal-codebook-b100-2026-08-29.md`](../reports/tabubase-nominal-codebook-b100-2026-08-29.md).

This supports a narrow mechanism conclusion: context-size distribution is a
pretraining variable. It does not establish that the present generator is broad
enough, that longer context is already supported efficiently, or that TabUBase
generally transfers to real tables.

## 3. Required semantic boundaries

The expanded distribution must preserve the following boundaries:

- one world is a task-generating mechanism, not merely one batch of rows;
- world split occurs before episode compilation and before context/query sampling;
- query response truth is absent from model forward and available only to the
  loss-side `TruthSidecar`;
- numeric statistics, category thresholds, imputers, and other estimated state
  use context/train evidence only;
- natural missingness, artificial masks, query markers, and semantic nulls remain
  distinct;
- the tokenizer codebook is representation state, not a source of target truth;
- frozen ICL creates no optimizer and preserves the parameter hash;
- generator coverage, synthetic fitting, frozen ICL, real fine-tuning, and
  comparison with XGBoost/MLP remain separate evidence questions.

## 4. Hierarchical world definition

The new generator should sample a world-level object

$$
\omega=
(M,d_z,G,\mathcal F,\beta,\Sigma,\pi,\nu,\rho,\mathcal T),
\qquad
\omega\sim p_{\Omega},
$$

where:

- $M$ is the predictor count;
- $d_z$ is the latent dimension;
- $G$ is a dependency graph or another declared dependence structure;
- $\mathcal F$ selects structural functions;
- $\beta$ contains effect parameters;
- $\Sigma$ controls latent and observed dependence;
- $\pi$ contains categorical probabilities;
- $\nu$ controls noise and tail behavior;
- $\rho$ controls missingness when the family permits it;
- $\mathcal T$ declares the response modality and transformation.

Rows are then conditionally sampled inside the world:

$$
z_i\sim p_{\omega}(z),
$$

$$
x_{ij}
=
f_{\omega,j}
\left(
z_i,
x_{i,\operatorname{pa}_{G}(j)},
\epsilon_{ij}
\right),
$$

$$
s_i=h_{\omega}(x_i,z_i)+\epsilon_i^{(y)},
\qquad
y_i=\mathcal T_{\omega}(s_i).
$$

The distinction is important: increasing the number of sampled rows from one
fixed $h$ does not increase the diversity of task laws. Expanded pretraining must
sample both $\omega$ and rows conditional on $\omega$.

## 5. Proposed schema prior

The schema itself should vary across worlds. The following values are candidate
ranges, not yet frozen defaults:

| Variable | Candidate distribution |
|---|---|
| Predictor count $M$ | log-bucketed over $4,8,16,32,64$ |
| Latent dimension $d_z$ | $1$ to $\min(16,M)$ |
| Numeric share | sampled per world, with at least one numeric lane in mixed worlds |
| Ordinal share | sampled per world; cardinality $3$ to $20$ |
| Nominal share | sampled per world; cardinality $2$ to $100$ |
| Response columns | exactly one for `supervised.label_broadcast.v1` |
| Response type | numeric, binary, ordinal, or nominal |

The training distribution must include both homogeneous and mixed-type schemas.
Feature order is randomly permuted after generation so that family or response
law cannot be recovered from a fixed column position.

## 6. Candidate value distributions

### 6.1 Numeric predictors

Numeric base variables should be drawn from a mixture of distribution families:

$$
p(x)=
\lambda_1\mathcal N(\mu,\sigma^2)
+\lambda_2 t_{\nu}
+\lambda_3\operatorname{LogNormal}(\mu,\sigma^2)
+\lambda_4\sum_{k=1}^{L}\alpha_k\mathcal N(\mu_k,\sigma_k^2),
$$

with world-level sampling of location, scale, skew, tail weight, and mixture
separation. Candidate transformations include monotone transforms, clipping,
quantization, counts, and bounded ratios.

Every estimated standardizer remains context-only:

$$
\mu_j(C)=\frac{1}{|C|}\sum_{i\in C}x_{ij},
\qquad
\sigma_j(C)=
\max\left(
\sqrt{\frac{1}{|C|}\sum_{i\in C}(x_{ij}-\mu_j(C))^2},
\epsilon
\right).
$$

### 6.2 Ordinal predictors

An ordinal variable can be generated from a latent score $a_{ij}$ and ordered
world-level thresholds:

$$
o_{ij}=c
\iff
\tau_{j,c}<a_{ij}\le\tau_{j,c+1},
\qquad
\tau_{j,0}<\cdots<\tau_{j,C_j}.
$$

Thresholds must come from world parameters or legal context evidence, never query
truth.

### 6.3 Nominal predictors

Nominal class probabilities should include balanced and long-tail worlds:

$$
\pi_j\sim\operatorname{Dirichlet}(\alpha_j\mathbf 1),
\qquad
\log\alpha_j\sim
\operatorname{Uniform}(\log\alpha_{\min},\log\alpha_{\max}),
$$

$$
c_{ij}\sim\operatorname{Categorical}(\pi_j).
$$

This produces common, imbalanced, and rare-category regimes instead of always
using four nearly balanced classes.

## 7. Candidate response-law families

No single family should dominate the update budget. The initial candidate
training mixture is:

1. sparse generalized linear and generalized additive models;
2. randomly sampled sparse DAG/SCM structural equations;
3. tree, threshold, and piecewise-constant rules;
4. latent-factor and low-rank interaction models;
5. polynomial and multiplicative interactions;
6. periodic, saturating, and monotone nonlinearities;
7. categorical lookup tables and category-by-numeric interactions;
8. mixture-of-experts and subgroup-specific response laws.

Examples include:

### Sparse additive law

$$
s_i=\beta_0+
\sum_{j\in S_1}\beta_jx_{ij}
+\sum_{j\in S_2}\gamma_jg_j(x_{ij})
+\epsilon_i.
$$

### Interaction law

$$
s_i=\beta_0+
\sum_j\beta_jx_{ij}
+\sum_{(j,k)\in E_I}\gamma_{jk}x_{ij}x_{ik}
+\epsilon_i.
$$

### Tree/threshold law

$$
s_i=
\sum_{\ell=1}^{L}
a_{\ell}
\mathbf 1[x_i\in R_{\ell}]
+\epsilon_i.
$$

### Latent-factor law

$$
x_i=A z_i+\eta_i,
\qquad
s_i=b^{\top}z_i+z_i^{\top}Qz_i+\epsilon_i.
$$

### Subgroup-specific law

$$
g_i\sim\operatorname{Categorical}(\pi),
\qquad
s_i=h_{g_i}(x_i)+\epsilon_i.
$$

Coefficient scale, sparsity, signal-to-noise ratio, interaction order, tree
depth, and subgroup overlap are sampled per world. The current fixed coefficients
must not survive as the only realization of a named family.

## 8. Response modalities

Regression may use a direct score or a monotone output transform:

$$
y_i=a+b\,q(s_i)+\epsilon_i,
$$

where $q$ can be identity, bounded, positive-only, count-like, or heavy-tailed.

Binary classification may use a sampled link and bias:

$$
P(y_i=1\mid x_i,\omega)=
\sigma\left(\frac{s_i-b_{\omega}}{T_{\omega}}\right).
$$

Multiclass classification may use sampled class scores:

$$
P(y_i=c\mid x_i,\omega)=
\frac{
\exp(s_{ic}/T_{\omega})
}{
\sum_{c'=1}^{C_y}\exp(s_{ic'}/T_{\omega})
}.
$$

Ordinal response uses ordered thresholds. Class balance and temperature are
world parameters and must cover easy, ambiguous, and imbalanced tasks.

## 9. Fixed 100-code category representation

The vector inventory remains fixed:

$$
E=\{e_1,\ldots,e_{100}\},
\qquad
\lVert e_b\rVert_2=1.
$$

The proposed generator does not assign a global semantic meaning to code index
$b$. For each world and declared domain, it samples or deterministically derives
an injective assignment

$$
\pi_{\omega,j}:
\{1,\ldots,C_j\}
\hookrightarrow
\{1,\ldots,100\},
$$

and emits

$$
t(c_{ij})=e_{\pi_{\omega,j}(c_{ij})}.
$$

This keeps the geometry stable while preventing a code such as $e_7$ from
silently meaning the same real-world category across unrelated worlds. Domains
with more than 100 categories fail closed in this design. Whether assignment is
world-scoped, source-scoped, or a paired ablation remains an owner decision to
freeze before implementation.

## 10. Context and query distribution

Context length is an explicit random variable:

$$
K\sim p_K,
\qquad
\mathcal K=
\{2,4,8,16,32,64,128,256,512,1024,2048,4096,8192\}.
$$

The query count is a separate compute variable:

$$
Q_{\mathrm{train}}\in\{32,64\}.
$$

Large context does not require a correspondingly large query batch. Query rows
can be chunked while sharing the same context evidence.

### 10.1 Staged maximum context

For total update budget $U$, a candidate curriculum is

$$
K_{\max}(u)=
\begin{cases}
128, & u<0.15U,\\
512, & 0.15U\le u<0.40U,\\
2048, & 0.40U\le u<0.70U,\\
8192, & 0.70U\le u\le U.
\end{cases}
$$

Within the active range, the sampler first chooses a log-scale bucket and then
jitters the exact length inside it. This retains anchor lengths while preventing
the model from seeing only powers of two:

$$
j\sim\operatorname{Uniform}\{j_{\min},\ldots,j_{\max}(u)\},
$$

$$
K=\left\lfloor2^{j+\delta}\right\rfloor,
\qquad
\delta\sim\operatorname{Uniform}(0,1).
$$

The final sampler probabilities must be recorded in the pretraining manifest.
The values above are a candidate, not a frozen schedule.

### 10.2 Nested evaluation contexts

For evaluation, one held-out world should generate a fixed ordered context bank
$C_{\max}$ and query bank $Q$. Smaller contexts are nested prefixes:

$$
C_2\subset C_4\subset\cdots\subset C_{8192}\subset C_{\max}.
$$

This makes the comparison paired: changes in loss across $K$ are caused by added
evidence rather than different query rows or a different sampled world.

Unseen lengths such as $192,768,3072,6144,12288$, and $16384$ should be reserved
for interpolation and extrapolation checks.

## 11. Length-aware batching

Episodes with very different $K$ should not be padded together. The dataloader
should bucket compatible schemas and lengths, with an approximately fixed row
budget per optimizer step:

$$
B(K,Q)=
\max\left(
1,
\left\lfloor
\frac{T_{\mathrm{rows}}}{K+Q}
\right\rfloor
\right).
$$

Short-context buckets therefore contain more worlds per forward pass. Long
contexts use smaller batches and gradient accumulation. The optimizer-facing
normalization must declare whether loss is averaged per target cell, per episode,
or per world so that long episodes do not silently receive a different weight.

Required runtime records include:

- sampled world IDs and $K$ values;
- rows, target cells, and worlds per optimizer step;
- gradient accumulation count;
- padding and valid-cell masks;
- data-generation and device-wait time;
- peak allocated memory and tokens/rows per second.

## 12. Long-context runtime requirement

The current same-column terminal forms dense row-to-row routing. With
$N=K+Q$, $M$ features, and address dimension $d_a$, its dominant intermediate
has shape approximately

$$
[N,M,N,d_a],
$$

and therefore costs

$$
O(MN^2d_a)
$$

time and memory. At $K=8192$, this is not an acceptable pretraining path.

The required long-context implementation computes only query response cells
against legal context response supports:

$$
O(QKd_a),
$$

not all cells against all cells.

For Nadaraya--Watson, support chunks can be accumulated exactly:

$$
\widehat y_q=
\frac{\sum_{i\in C}w_{qi}y_i}
{\sum_{i\in C}w_{qi}},
\qquad
w_{qi}=\exp\left(-\frac{\lVert z_q-z_i\rVert^2}{h^2}\right).
$$

For local-linear readout, each support chunk updates sufficient statistics:

$$
S_0=\sum_iw_i,
\quad
S_x=\sum_iw_i\delta_i,
\quad
S_{xx}=\sum_iw_i\delta_i\delta_i^{\top},
$$

$$
S_y=\sum_iw_iy_i,
\qquad
S_{xy}=\sum_iw_i\delta_i y_i.
$$

The final solve uses the accumulated moments. The implementation must match the
dense terminal on small episodes within a preregistered tolerance before it is
used for long-context evidence. A memory-saving approximation is a distinct
terminal variant and cannot inherit exact-terminal receipts.

## 13. Worlds that make long context useful

Long context must add information, not merely repeat an easy pattern. The
generator should contain tasks where additional evidence is statistically useful:

- weak effects near the detection boundary;
- rare categories and rare subgroups;
- many nuisance or redundant predictors;
- noisy decision boundaries;
- high-order interactions;
- mixtures whose components become identifiable only with more rows;
- calibration and tail-estimation problems.

A weak-effect family can scale the effect size with the intended difficulty:

$$
y_i=\frac{c}{\sqrt{K_*}}x_{ij}+\epsilon_i,
$$

where $K_*$ is a world-level difficulty parameter, not the observed evaluation
context. This allows a world to require roughly $K_*$ examples without leaking
the evaluation value of $K$ into its law.

Rare-category worlds can draw

$$
P(c=c_{\mathrm{rare}})=p_{\mathrm{rare}},
\qquad
p_{\mathrm{rare}}\ll1.
$$

The generator must also retain easy and low-context worlds. Otherwise scaling to
long context would destroy the existing few-shot ICL target.

## 14. Train, validation, and held-out family split

Splitting world IDs after episode compilation is invalid. The order is:

$$
\text{family/template split}
\rightarrow
\text{world-parameter split}
\rightarrow
\text{row generation}
\rightarrow
\text{context/query compilation}.
$$

Evaluation should distinguish:

1. **seen-family/unseen-world**: new parameters and rows from trained families;
2. **held-out-family synthetic ICL**: at least one complete function family not
   used for optimization;
3. **held-out regime**: unseen noise, cardinality, width, or context length;
4. **real frozen ICL**: no fine-tuning and no optimizer;
5. **real fine-tune transfer**: a separate Link-6 comparison.

Candidate missingness policy:

- train on declared no-missing, MCAR, and bounded MAR regimes;
- hold out MNAR and at least one unseen MAR mechanism;
- never infer missingness masks from target truth.

This policy is not frozen; it changes the meaning of the held-out-family test and
must be preregistered.

## 15. Generator-level correctness gates

Before model training, the data system must pass independent generator gates.

### G-D0: deterministic replay

The tuple

$$
(\text{generator version},\text{root seed},\text{partition},\text{world ID})
$$

must reproduce byte-equivalent world parameters, rows, schema, roles, masks, and
truth-sidecar bytes on the same declared backend.

### G-D1: authority and leakage

- world partition precedes compilation;
- query target truth cannot affect tokenizer state, thresholds, statistics, or
  masks;
- context-only estimates match an independent reference;
- truth substitution leaves model inputs unchanged.

### G-D2: distribution coverage

The sampled corpus must report empirical coverage over:

- function families;
- response modalities;
- widths and type mixtures;
- category cardinalities and imbalance;
- SNR/noise buckets;
- missingness regimes;
- context-length buckets.

Coverage thresholds must be frozen before the run. A total world count alone is
not sufficient.

### G-D3: known-reference recovery

For selected worlds, an oracle or correctly specified classical learner must
recover the intended relationship as context grows. This checks that generator
noise and task difficulty are internally coherent.

### G-D4: dense/streaming parity

At small $K$, query-only chunked routing must match the current dense reference
within declared numeric tolerance, including support ledgers and empty-support
status.

### G-D5: long-context execution

At $K\in\{1024,2048,4096,8192\}$:

- forward and backward remain finite;
- peak memory remains within the declared budget;
- at least one active component group receives finite non-zero gradient;
- checkpoint resume remains exact under the declared determinism contract;
- no fallback to dense $N\times N$ routing occurs.

## 16. Pretraining and ICL evaluation

The primary frozen-ICL curves are evaluated per task modality:

$$
L_{\mathrm{cls}}(K)=
\frac{\operatorname{NLL}(K)}
{\operatorname{NLL}_{\mathrm{reference}}},
$$

$$
L_{\mathrm{reg}}(K)=
\frac{\operatorname{RMSE}(K)}
{\operatorname{scale}_{\mathrm{reference}}}.
$$

Context utilization is measured by the paired marginal gain

$$
\Delta(K)=L(K)-L(2K),
$$

and by AULC over $\log_2K$. A model is not credited for long-context ICL merely
because it accepts the tensor shape. Normal context must outperform:

- random-init frozen;
- label-shuffled context;
- predictor/context-row shuffled controls where applicable;
- a context-tail removal control for large $K$.

Metrics should include:

- AULC and endpoint with world-clustered confidence intervals;
- monotonicity-violation rate over nested contexts;
- gain per context doubling;
- performance at unseen interpolation and extrapolation lengths;
- calibration for classification;
- parameter hash before and after every frozen arm.

Classification and regression remain separate. Results are not averaged into one
foundation-model score.

### Real downstream frozen-ICL estimand

The $K$-grid above is primary for synthetic context-length scaling, but it is
only an auxiliary low-shot diagnostic on a real downstream dataset. For a real
split $s$, let $T_s$ be the complete train partition and $Q_s$ the complete
held-out partition. The primary downstream frozen-ICL context size is

$$
K_s = |T_s|,
$$

and the complete evidence episode is

$$
E_s=
\{(x_i,y_i):i\in T_s\}
\cup
\{(x_j,\bot):j\in Q_s\}.
$$

Thus every train response is visible, every query response is hidden in the
evidence tensor, and all held-out predictors pass through shared dynamics in
one transductive episode. A bounded $Q_{s,c}\subseteq Q_s$ may be used only to
chunk the query-response terminal after those shared dynamics; it must never
change $E_s$, truncate $T_s$, or omit rows from $Q_s$. The frozen gate is

$$
\theta_{\mathrm{after}}=\theta_{\mathrm{before}},
\qquad \mathrm{optimizer}=\varnothing.
$$

Classification reports accuracy, balanced accuracy, macro-F1, log loss,
normalized NLL, and one-vs-rest macro ROC-AUC. Regression reports RMSE, MAE,
scaled RMSE, scaled MAE, and conventional held-out $R^2$. If the runtime cannot
evaluate $|T_s|$ context rows, that dataset/split is blocked evidence rather
than an authorization to fall back to $K\leq32$.

MLP and XGBoost comparisons must use the exact same train indices, held-out
indices, and selected feature indices, verified by shared split-manifest
hashes. Their fitted inductive semantics remain distinct from TabUBase's one
all-query transductive frozen episode; the side-by-side metrics are therefore
descriptive references rather than a claim that the estimands are identical.

## 17. Proposed implementation ladder

This ordering minimizes the chance that a large training run measures a runtime
artifact instead of the intended prior.

### Stage A — Freeze the generator contract

- decide schema ranges, family weights, category assignment scope, and held-out
  families;
- define a versioned world manifest and canonical hash;
- implement generator-only correctness and coverage tests.

Exit: deterministic generator replay and no-leakage gates pass. No model result is
implied.

### Stage B — Remove the quadratic long-context wall

- add query-response-only terminal execution;
- add exact chunked NW/local-linear accumulation;
- prove dense/streaming parity at small $K$;
- record peak-memory scaling.

Exit: $K=8192$ forward/backward smoke passes without dense row-pair materialization.

### Stage C — Bounded curriculum experiment

- train only up to $K=512$;
- compare fixed-$K$, current variable-$K$, and expanded-law variable-$K$ arms;
- verify that added law diversity does not destroy low-$K$ ICL.

Exit: predefined synthetic held-out ICL and context-shuffle gates pass.

### Stage D — Long-context curriculum

- introduce $K=1024$ and $2048$;
- then introduce $K=4096$ and $8192$;
- use length-aware batches and a fixed optimizer-facing target-cell budget;
- stop if loss, memory, or throughput crosses preregistered kill conditions.

Exit: positive context utilization is observed on tasks constructed to need more
evidence, not merely successful execution.

### Stage E — Expanded pretraining

Only after Stages A--D are reviewed should the world/update budget be enlarged by
an order of magnitude. The expanded run must retain failed attempts, exact
commands, manifests, checkpoints, and immutable local/formal receipts according
to the research operating system.

## 18. Decisions still open

The following choices are intentionally not frozen by this proposal:

1. exact family mixture weights;
2. maximum predictor width within or beyond the current `max_features=64`;
3. world-scoped versus source-scoped assignment into the fixed 100-code inventory;
4. the proportion of missingness families used for training versus held-out ICL;
5. exact $K$ sampling probabilities and stage boundaries;
6. row/target budget per optimizer step;
7. query count and support chunk size;
8. whether inducing-slot count remains fixed or becomes a separately identified
   architecture variant;
9. which families are completely held out;
10. the scale rung after the bounded curriculum passes.

Each choice that changes task semantics, model composition, or the statistical
estimand must enter the future manifest and identity. It cannot remain an
unrecorded CLI convenience.

## 19. Explicit non-goals

This proposal does not authorize or claim:

- public checkpoint publication;
- SOTA or superiority over XGBoost, MLP, TabPFN, or LimiX;
- causal identification from observational synthetic tables;
- unlimited context support;
- a successful PT-S2/S3 run;
- a formal Link-5 or Link-6 receipt;
- a foundation-model designation.

The immediate deliverable is a reviewable data-and-runtime contract. Training
begins only after the corresponding generator version, preregistration, budgets,
controls, and kill conditions are frozen.

## 20. Exploratory real-data expansion (2026-08-30)

To stress the frozen-transfer question without changing the existing new6
receipts, an independent cached-OpenML panel was added:

`white_wine`, `red_wine`, `cpu_activity`, `kin8nm`, `pumadyn32nh`,
`energy_efficiency`, `cars`, and `space_ga`.

All eight tables are numeric-only, have zero missing cells in the pinned ARFF
snapshot, and fit the current 63-predictor model limit. The panel is evaluated
with all train rows as labeled context and all held-out predictors in one
transductive episode, then compared with fixed MLP/XGBoost fits on exactly the
same splits. It is an exploratory `local_unissued` data-coverage result, not a
new pretraining phase and not evidence for a broad benchmark claim. The
receipt, source hashes, and common-metric table live in
[`tabubase-cached-openml-regression-8-full-context-2026-08-30.md`](../reports/tabubase-cached-openml-regression-8-full-context-2026-08-30.md).

## 21. Exploratory synthetic-pretraining → real-task fine-tuning (2026-08-30)

The same eight cached OpenML regression tables were also run through a
separate non-frozen transfer lane. A PT-S1 long-context checkpoint initialized
one arm, while a same-seed scratch model formed the paired control. Both arms
used 400 AdamW updates on 128 real train labels and were scored on every
held-out test row; MLP and XGBoost used the identical train/test row sets.

Pretrained initialization lowered mean scaled RMSE against scratch on 7/8
datasets, with the largest improvements on `cars` and `cpu_activity`.
`energy_efficiency` was the only small regression. This is initialization
evidence under a 128-label exploratory protocol, not a claim that TabUBase
beats fitted baselines or that the frozen ICL estimand changed. Full metrics
and the receipt are in
[`tabubase-cached-openml-regression-8-finetune-2026-08-30.md`](../reports/tabubase-cached-openml-regression-8-finetune-2026-08-30.md).
