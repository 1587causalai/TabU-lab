# V8 completion: first executable branch

## Mainline integration boundary (2026-10-11)

The default remains the **64-coordinate V8 baseline**. The opt-in V8.1 mode
below adds a learned feature codebook with the same 64-coordinate value interface.
The default is one completion round and Gaussian
column-shared local-linear recovery. The positive/signed attention readouts
remain opt-in, paused research alternatives; merging their code does not resume
their exploration or promote any checkpoint.

From a clean checkout, run the synthetic example with no private data or donor:

```bash
uv sync --frozen --extra dev
uv run python examples/completion_v8_smoke.py
uv run python examples/completion_v8_smoke.py --device mps  # when available
uv run pytest tests/unit/completion_v8 -q
```

The example performs one FP32 update and exact model-weight save/reload replay.
It is not an optimizer-resume or model-quality benchmark. See the
[integration record](../../../../docs/reviews/v8-integration-20261011/README.md)
for the imported scope, regression evidence and reproducibility limits.

The dated `preparations/...` and `latex/...` references below live in the separate
outer TabU research workspace, **not this Git repository**. They identify
historical receipts, not files available after cloning this repository. Historical
words such as "latest" describe the corresponding run and do not select today's
training parent. Real checkpoints, original datasets and full optimizer/RNG
artifacts are not distributed in this integration. Current research has stopped
the V8 baseline exploration; V8.1 engineering checks are separate from model-quality results.

## V8.2: presence-preserving linear row attention (2026-10-12, opt-in)

### Current supervised attention policy

The current supervised design is implemented as
`attention_policy="support_rows_null_v1"`. It requires one target column and
each row to be either a labeled Support or a target Query. Across rows, the
encoder (forward and inverse), column collect/read and optional Unit stack
use only **Support rows** as sources, including when Query features are known.
Within each row, every real column participates, including the Query target.

There is **no per-Cell source mask** on this path. Source sets are physical row
or column slices. Exact-zero Null carriers contribute zero through the existing
presence gates and bias-free dynamics; intermediate layers do not reset them
with a Null mask. Visible/Query roles remain at the input/codec/phi boundary and
for the supervised readout and loss. Unit/Feature remain structural receivers.

The mean divisor on this path is the table's real column count `M`, including
Null addresses, never an active-Cell count or sum of learned presence. Removing
a column therefore changes the mean, even if its carrier was zero. This is an
explicit protocol change from the historical active-source mean.

```python
from tabu_lab.models.completion_v8 import with_supervised_attention, configure_memory

model = with_supervised_attention(v82_donor)  # exact weights-only copy, new config
configure_memory(model, attention_mib=128, checkpoint_activations=True)
# For a fresh V8Config, set attention_policy="support_rows_null_v1" explicitly.
```

Serialization records the policy. Old configurations without the field retain
`legacy_cell_sources`; loading an old checkpoint never silently changes its
graph. The old `inherited.query_source` toggle has no effect on the new policy.
No parameters are added and no optimizer/RNG state is implicitly transferred.

```bash
uv run python examples/completion_v8_smoke.py --codebook-mode learned_feature64_v1 --row-attention presence_linear_mean_v1 --attention-policy support_rows_null_v1
```

Communication tests check Query-row isolation and gradients at encoder/backbone
boundaries, same-row communication, Null closure, forward/inverse consistency,
FP32 CPU/MPS loss/backward, checkpoint replay and activation recomputation.
FP64 is used only as an algebraic reference for forced tiny batch chunks.
FP32 chunking can change rounding, amplified by Gaussian LL on poorly
conditioned small episodes; bitwise equivalence across batch shapes is not
claimed. Preprocessing and LL's joint centers remain unchanged, so the complete
prediction is **not claimed independent of Query grouping**. No new quality
result or training budget follows from these engineering checks.

### Historical row operator and compatibility

V8.2 addresses the quadratic **column-count** cost of Step 3's within-row
attention. It retains V8.1's learned 64D feature pool, reversible encoder,
column inducing attention, FFN, Unit receivers, Gaussian LL and completion loss.
It does not replace the recovery readout with linear attention. The current
candidate is `presence_linear_mean_v1`; `presence_linear_v1` retains the raw
sum formulation as a diagnostic comparison. Neither selects a project-wide
default or a trained checkpoint.

```python
from tabu_lab.models.completion_v8 import (
    V8Config, V8CompletionModel, from_v81, configure_memory,
)

config = V8Config(
    inherited_spec, codebook_mode="learned_feature64_v1",
    row_attention="presence_linear_mean_v1",
)
model = V8CompletionModel(config)  # all model arithmetic defaults to FP32
# Or copy every V8.1 model tensor, with no added parameters:
# model = from_v81(v81_donor)
configure_memory(model, attention_mib=128, checkpoint_activations=True)
```

For each row/head, let `n=max(1, number of role-eligible sources)` and
`rho(x)=||P_A x||²/(tau+||P_A x||²)`. Q/K/V project raw states; normalized
queries/keys are `q_hat=q/sqrt(||q||²+eps)` and similarly for `k_hat`.

```text
S = (1/n) sum_j rho(y_j) k_hat_j v_j^T
h_i = S^T q_hat_i / sqrt(head_width)
attention_update_i = rho(x_i) W_O concat_heads(h_i)
```

The existing residual and local FFN follow unchanged. Source presence occurs
once, before aggregation; receiver presence gates the output. The divisor counts
role eligibility, not presence mass. For fixed Q/K/V and masks, multiplying every
source gate by `c` multiplies the attention increment by `c`. No positive kernel,
softmax/weight-sum denominator or attention-output RMS cancels this attenuation.
A role-eligible zero-presence source contributes zero but still counts in `n`;
this avoids a discontinuous divisor at gate zero. Null/ineligible positions do
not count and are deleted before projection. Unit/Feature tokens remain receivers
only; Query source eligibility follows the inherited policy.

The original raw sum overflowed after direct migration of the latest V8.1
weights on all three real wide-table forwards. Q/K normalization removes their
unbounded dot-product scale; the role-count mean removes width-dependent sum
growth. These are explicit **new design choices**, not an algebraically equivalent
optimization of softmax. With normalized Q/K, each head satisfies
`||h_i|| <= max_j ||v_j|| / sqrt(head_width)` (before receiver gate/output
projection); this does not prove stability of the whole network or training.
The raw diagnostic mode uses unnormalized Q/K and no count divisor; failures
remain explicit. It is not the recommended migration mode.

Each head stores a `head_width × head_width` summary. With `N` rows, `M` columns,
width `D` and head width `d`, row aggregation changes from `O(N M² D)` to
`O(N M D d)`, plus unchanged projections/FFN. Columns, encoder, codec preparation
and Gaussian LL still cost time; narrow tables may see no speed benefit.

`from_v81` requires the current 64D Gaussian-LL donor with `row_slots=0` and
copies every tensor exactly. Changed attention changes predictions. Optimizer
and RNG are not implicitly migrated. Saved files use the separate
`completion_v82_weights_v1` schema and serialize the precise row mode. Old
configs without `row_attention` retain inherited attention and their old schema.
The explicit memory helper bounds **independent batches**, preserves all sources
for each receiver, and recomputes encoder stages/backbone layers during backward.
The workspace budget is an estimate, not an allocator limit; weights files do
not automatically enable memory policies after reload.

The source also carries forward the latest frozen V8.1 without-replacement
feature allocation correction from 2026-10-11. This matches the current trained
donor's execution source, not the earlier V8.1 sampling implementation at
commit `76eb4dd6`. Full recoding assigns distinct indices to at most 10000 columns;
numeric/token RNG is unchanged. Historical sampling reproduction must use its
recorded frozen source. Partial mixing of independently sampled reference/new
plans can still collide; the qualified current protocol uses full recoding.

```bash
uv run python examples/completion_v8_smoke.py --codebook-mode learned_feature64_v1 --row-attention presence_linear_mean_v1
uv run pytest tests/unit/completion_v8 -q
# Private donor/data optional; no optimizer updates:
uv run python examples/completion_v82_benchmark.py --checkpoint V81.pt --output speed.json --device cuda
```

Engineering evidence, exact source/runtime identities and actual speed results
are recorded in the outer research workspace's
`preparations/v82-presence-linear-20261012/REPORT.md`. Numerical qualification
does not establish retained predictive accuracy or unseen-table generalization.

## V8.1: learned feature vocabulary (opt-in, corrected to 64D)

```python
from tabu_lab.models.completion_v8 import V8Config, V8CompletionModel, from_v8

config = V8Config(inherited_spec, codebook_mode="learned_feature64_v1")
model = V8CompletionModel(config)
# Preserve EVERY donor model weight; only the feature bank is new:
# model = from_v8(v8_donor)  # optimizer migration is not implicit
```

Only `model.feature_codebook.weight` is the new learned vocabulary:
10000 by 64, 640000 parameters, iid Gaussian normalized to unit length **once**.
The feature index is sampled uniformly; the bank is not reset or renormalized.
Numeric/category directions remain constants from the unchanged C64/8 or G64
codec: same inputs and seed preserve the old draws and preprocessing exactly.
Feature-index sampling uses a separate generator and does not consume codec RNG.
A unit-length Gaussian feature and a weight-8 binary direction share 64D shape,
not norm/distribution: their initial norms are 1 and sqrt(8), respectively.

Code/query/phi widths remain 64; table-coupling branches remain 32+32, lift64→128.
The whole encoder, lift, Query seed, backbone, Unit/Feature seeds can be copied
without slicing, padding or reinitializing old interfaces. Changing the feature
origins can still change predictions; this is not prediction-equivalent resume.
Typed sums, category eligibility, ranks, decoding and loss are unchanged.
The 10000-entry feature vocabulary does not limit category count.
The historical `learned_pool_v1` mode **continues to mean 32D** so its checkpoints
remain readable; request it explicitly in `from_v8(..., codebook_mode=...)` only
for historical reproduction. Do not continue the failed 32D run as the 64D donor.

Preparation caches feature **indices**, random directions and fitted statistics.
`episode.observed` is `None` in this mode: every forward binds a fresh
differentiable codec to the current bank. `output.codec` contains that forward's
snapshot for probabilities/decoding and `completion_loss`; pass the same episode
to forward and loss, then update the optimizer. Reusing the prepared episode
across optimizer steps is supported; caching learned observed codes is not.

```python
episode = prepare_completion(
    visible_only_inputs, model.config,
    recode_columns=selected_columns, reference_seed=0,
)
output = model(episode)
loss = completion_loss(output, episode, truth_sidecar)
loss.backward()
optimizer.step()
```

Omit `recode_columns` to draw the entire encoding from `inputs.code_seed`.
For partial recoding, unselected columns keep reference feature indices **and**
random directions; they still read the current bank. The caller controls the
selected column count and each input seed, so this option changes neither
training exposure nor replay proportions. Seeds determine draws with a local
generator; no hidden global sampling cursor is introduced.

`save_weights` records the bank and mode with schema
`completion_v81_weights_v1`; `load_weights` checks schema/mode agreement.
Old configs with no mode keep the original serialization and initialization.
These are weights-only files. A full training runner must separately preserve
optimizer, sampling state/input seeds, reference seed and recoding selection
to resume exactly. No old experimental runner or saved run is rewritten.

```bash
uv run python examples/completion_v8_smoke.py --codebook-mode learned_feature64_v1
uv run python examples/completion_v8_smoke.py --codebook-mode learned_feature64_v1 --device mps
uv run pytest tests/unit/completion_v8 -q
```

## 当前探索精度默认（2026-10-10 owner更新）

速度优先：**FP32参数、attention/presence、Gaussian核与LL矩阵累积/Cholesky/求解**。`V8CompletionModel(config)`及`load_weights(fp32_path)`自动使用此默认，无需调用方记得额外配置。显式改变参数dtype时计算策略随之调整；纯设备迁移保留已选策略。FP64旧权重加载时保留原dtype，开始新探索可显式转换为FP32并记录协议。

```python
from tabu_lab.models.completion_v8 import load_weights, configure_precision
model = load_weights(checkpoint, device="cuda")  # FP32权重默认快速策略v2
policy = model.execution_precision              # 写入本次运行协议
# 历史FP64计算重放：load_weights(checkpoint, precision="reference")
# 复现上一版混合精度，或经证据支持的局部例外：
# configure_precision(model, mode="fp32_attention_fp64_ll", reason="具体必要性")
```

设置按模型实例隔离，V7等历史代码的默认不改。新保存的权重附带执行精度回执；加载默认遵循当前策略，严格重放需显式传历史策略。旧冻结源里的`fast_fp32`是仅注意力FP32；当前策略版本2同时覆盖LL，因此必须同时记录源码hash和`model.execution_precision`，不能只写模式名。数值失败会显式报错，不静默重试双精度。

必要的窄范围例外：可见数值统计/归一化保留稳定精度，避免大而近似相等的数变成同一个值；确定性码本生成/编解码沿用原协议，避免这次执行调整同时改变表示；评估指标独立复算保留审计精度。这些不意味着模型主干和LL继续双精度。

FP32数值核验入口与结果：`preparations/v8-fp32-default-20261010/`。当前1024分支最新实验为`preparations/v8-arena48-window1024-continue15m-20261010/`：48张非宽表继续15分钟，全部51表评测；其`common.py`、`memory_overlay.py`、`activation_overlay.py`及协议冻结源沿用上一轮资格路径。最新/本轮验证选中均终点445e91d9…（配套AdamW/RNG a08db00c…，Arena累计1946步，下一window-step1512、下一48表序号1160）。均值改善而整体Elo下降，保留候选，不自动替换全项目默认；上一轮51表终点2eda2fe3…保留对照。历史源码快照和旧运行结果未重写；表池/精度/训练窗口切换需记录新协议，不自动扩大训练预算。

后续splice专用5分钟探针见`preparations/v8-splice-window1024-ft5m-20261010/`：终点217更新63ee8066…（AdamW/RNG 4909e38a…，累计2163步、下一window-step1729）验证/测试Logloss均退化，验证选回父共享点445e91d9…。终点保留诊断，不替换上面的共享候选或改变其48表续接位置。

固定1000条拟合探针与基线见`preparations/v8-splice-fixed1000-fit5m-20261010/baselines/REPORT.md`：splice固定编码/8遮罩、fresh AdamW lr1e-4五分钟后训练96.5%/全1000上下文测试91.41%；同本轮数据MLP测试92.19%、CatBoost94.53%、XGBoost97.66%。学生学业表五分钟训练97.2%但测试69.53%（起点75%）。这证明两表上固定样本能明显拟合，不证明普遍泛化或架构容量上界。两个终点均仅为诊断，不替换上面的共享候选或续接游标。第三张SDSS17按owner先看基线再继续的顺序仅准备，零更新。

1024最宽表训练已核验：仅OMAB分块时正向逆编码触到64GiB；运行入口增加encoder正逆与逐backbone层的非重入激活重算后，完整1024行/1755列反向峰值40.22GiB。小型真实样本输出/损失及132组梯度逐值一致，146步共享微调完成。重算仅改变中间激活保存方式，保持全部行列和FP32；核心构造器不会隐式启用这个运行级叠加。

This opt-in prototype implements the matrix-completion manuscript in
`latex/model-factory/TabU_8th/TabU_Matrix_completion/`. It does not change V7
factories or defaults. The first implementation is intended to expose interface
and numerical errors, not establish predictive superiority.

```python
import torch
from tabu_lab.models.completion_v8 import (
    from_v7_checkpoint,
    prepare_completion,
    completion_loss,
    save_weights,
)

model = from_v7_checkpoint(donor_path)  # exact donor precision, weights only
model = model.to(device="cuda", dtype=torch.float32)  # explicit new-run conversion
episode = prepare_completion(visible_only_inputs, model.config)
output = model(episode)
loss = completion_loss(output, episode, truth_sidecar)
loss.backward()
save_weights(model, output_path)  # write once; no optimizer-resume claim
```

## Baseline V8 choices and inheritance boundaries

1. Reuse the visible-fitted V7 codec, declared column schemas and input sanitation.
   Natural missing Query may be completed; only `TruthSidecar.states == 1` Query
   are scored. The default objective is terminal-round, unweighted per-cell MSE
   averaged over 64 coordinates. An explicit nonempty scoring subset is supported.
2. One new shared, trainable 64-coordinate Query seed replaces type-specific seeds
   and donor initialization. Default is one round; optional rounds share all
   parameters and carry gradients through simultaneous Query writes.
3. Per-cell invertible phi, then split 32+32. Four independently parameterized
   coupling blocks mix columns within rows, rows within columns, columns, rows.
   The attention follows LN, SiLU Q/K/V, Q/K L2, summed K-transpose-V, scale
   1/sqrt(32), RMSNorm, receiver presence, and output projection. No source presence,
   softmax denominator, delta update, decay, or positional ordering is introduced.
   OFFN reuses the established 32→64→32 gated RMS/GELU implementation. Null storage
   is canonical zero and is outside the active-coordinate bijection. New blocks
   use ordinary random linear initialization; they are not identity-initialized.
4. Keep the donor's complete 128-wide Token Dynamics, optional Unit stack, source
   policy, Unit/Feature seeds, and strict-finite behavior. Compatible phi/lift are
   copied too. The first API deliberately requires width 128 and code width 64.
   The full donor spec is serialized as `inherited`; its round count, old loss,
   Query initialization, value encoder, and gradient-checkpointing switches do
   not control this branch. V8 owns those explicitly; the core constructor does
   not enable activation checkpointing. The qualified 1024 training wrapper
   described above applies and records recomputation separately.
5. Reuse V7's Gaussian Unit kernel and differentiable column-shared ridge solver;
   regression features are the full Cell token (`P_G = I`). This is an explicit
   first recovery choice, not an implementation of the manuscript's illustrative
   learned column-conditioned dot-product kernel. Centers are only the active
   rows of that column, supports only its actual visible cells. CUDA/CPU solver
   arithmetic defaults to FP32 in V8 exploration; explicit historical reference
   mode uses FP64 on CUDA/CPU and FP32 on MPS.
6. Assemble all predicted Query embeddings into one latent table, preserving
   non-Query embeddings. Invert the whole table once, then clamp visible **codes**.
   Clamping latent observations before/after a per-column inverse is not equivalent.

`from_v7` copies all keys of the selected backbone with strict loading and exact
value checks. Unshared donors require an explicit round selection. Dual-stream
donors lack phi/lift and require explicit `copy_value_interface=False`; their
Token Dynamics and seeds can still be copied. Old contextual encoders, old Query
seeds, optimizer and RNG are never silently inherited. The migration receipt
enumerates copied and newly initialized keys. Copying weights does not preserve
full-model predictions because the value representation and Query protocol change.

## Verification and limits

Tests cover explicit pairwise versus factored attention, full-table inversion and
gradients, axis order/permutations, hidden-truth isolation, mixed missing/observed
Query scoring, simultaneous latent assembly, actual parameter updates, strict
module inheritance, checkpoint reload, and MPS forward/backward.

The old120 probe uses the historical trained phi donor at step 7580, not a new
adopted default. Its fixed data/sampler, frozen source manifest, CUDA preflight,
short training, saves and terminal replay live in
`preparations/v8-completion-old120-probe-20261009/` at the outer TabU root.
Its single-target tasks exercise an important specialization, not arbitrary
missing-pattern training or unseen-table generalization. Multi-column Query and
natural missingness are currently covered by small integration tests.

An algebraic inverse need not be well-conditioned. Nonfinite intermediates fail
explicitly; broad numerical stability and convergence remain open. The short
probe's clipped-gradient frequency must be inspected, not interpreted as proof
of stability merely because all updates complete.

## Post-probe numerical correction

The frozen five-minute run predates a stability patch. Direct squaring in Q/K
L2 and output/OFFN RMS normalization could overflow on huge finite tensors and
silently return zero. `stable_normalize` rescales the input and epsilon together
and uses vector norm/hypot; this preserves the mathematical operator without
the dangerous square. The old Token Dynamics remain unchanged. The fix passed
CPU/MPS tests and CUDA forward/backward, zero-gradient and extreme-value probes.
Normal-range terminal predictions differ by up to 5.2e-5 on three checks, so the
old run's bitwise replay applies only to its frozen source, not across the patch.

A separate design boundary is now measured: visible-code clamping after inverse
generally changes re-encoded Query embeddings. This is a learned completion
update, not an exact joint-constraint solver. The attention readout alternative is now an implemented opt-in candidate,
without a training comparison or a default change.

## Optional positive linear readout

This earlier positive-normalized option is retained for comparison. Owner review
instead selected the signed option below to avoid the convex-combination limit.

Select `V8Config(spec, readout="positive_linear_attention")`, or pass the same
`readout` argument to `from_v7` / `from_v7_checkpoint`. Old serialized configs
continue to select `gaussian_ll`; the default state keys and initialization are
unchanged. The candidate adds shared bias-free 128→32 Q/K projections, applies
softplus + 1e-4, and uses visible-only 64D embeddings as unprojected values.
Per-column factored normalized aggregation avoids both a pairwise matrix and the
ridge solve. Common source scaling and per-query scaling cancel in the ratio;
no additive denominator epsilon, output normalization, or output projection
changes response coordinates. Empty support and nonfinite arithmetic fail.

The original LL uses both Unit geometry for its kernel and Cell regression
features. This candidate generates matching weights from Cell Q/K alone. In the
inherited implementation Unit and Feature tokens are receivers, not Cell sources,
and the extra Unit stack updates only Unit outputs. Perturbing those seeds and
Unit-only weights leaves candidate predictions bitwise identical in a CPU test.
They are therefore dispensable for this candidate's mathematical prediction,
and the separate `cell_dynamics.py` execution path now skips these receivers and
the Unit-only stack. Their stored keys remain for checkpoint compatibility; the
old LL execution path is intact. Cell outputs and input gradients match the full
inherited path within FP64 tolerance, with/without Query sources and row slots.
No whole-model speedup from this pruning has been measured. The remaining Cell attention weights
still participate, so this does not discard learned relational computation.

Tests cover explicit positive weights, constant/single-support responses, support
permutation, hidden-truth isolation, no LL invocation, finite end-to-end gradients,
and exact native checkpoint reload on CPU, MPS, and CUDA. This candidate has not been
trained; it lacks LL's explicit local linear extrapolation term. Existing LL probe
results do not establish its predictive quality.


## Optional signed linear readout (paused by owner)

2026-10-10: the owner paused exploration of all linear-attention readouts.
Preserve the implementations, checkpoints, and evidence; do not train or
redesign these candidates until the owner explicitly resumes this direction.
Current V8 training continues with the original Gaussian LL readout.

Select `V8Config(spec, readout="signed_linear_attention")`, or the same argument
to either V7 migration entry point. This adds a separate `SignedLinearReadout`;
the old LL default and positive-normalized candidate remain available.

Q/K use bias-free 128→32 projections and SiLU. **No LayerNorm or L2
normalization is added to this readout.** V is the
visible 64D value embedding without projection. Per-column 256-source chunks
accumulate K-transpose-E; Q reads the 32x64 summary, scaled by fixed 1/sqrt(32).
There is no softmax, positivity map, weight-sum denominator, source-count divisor,
output normalization, output projection, or presence gate. The Cell-only path
is shared with the earlier candidate; the original LL path is unchanged.

Coefficients may be negative and need not sum to one. There is no fixed bound on individual
coefficients; the result is not restricted to the support convex hull.
Constant responses are not guaranteed to be preserved, and duplicating supports
doubles the aggregate; fixed head-width scaling does not normalize source count.
These are explicit properties, not silently corrected by reintroducing convex
normalization. Nonfinite results and empty support fail explicitly.

All 35 V8 unit tests pass, including CPU/MPS forward/backward and exact reload;
the earlier CUDA and 100k-scale checks cover the historical L2 revision.
The current no-L2 candidate completed separate 5+10 minute CUDA stages with
qualification, migration/continuation, finite-state, and exact reload receipts.
Its fixed four-table R2 reached 0.4290, but the historical real12 benchmark
failed badly at larger support sizes. The signed sum has no source-count
calibration guarantee; preserve this as an experimental candidate.
See `preparations/v8-signed-readout-20261010/` at the outer TabU root.

每步随机码本诊断（2026-10-10）：`preparations/v8-splice-fixed1000-recode5m-20261010/RESULT.md`，原码本训练96.5%→82.2%、测试91.41%→82.03%；两个本轮未用码本平均测试59.38%→73.44%。224更新/300.425有效秒已停止；终点仅作诊断，不改变共享48表候选或其续训位置。

随机码本再续十五分钟（2026-10-10）：`preparations/v8-splice-fixed1000-recode15m-20261010/RESULT.md`，669更新/900.495有效秒，原码本训练83.3%/测试79.69%，两探针码本均值训练80.35%/测试82.81%。终点仅作已知表诊断；共享48表候选及其续训位置不变。

Muon直接探索五分钟（2026-10-10）：`preparations/v8-splice-muon-direct5m-20261010/REPORT.md`，219更新/300.972有效秒，原码本训练82.5%/测试77.34%，两探针均值81.7%/79.30%，未见整体突破。保存中点/终点与Muon+AdamW状态；仅诊断，不替换共享候选。四组对照按owner快速探索纠正提前停止。
