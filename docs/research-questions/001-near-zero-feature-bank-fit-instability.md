# 研究问题 001：近零 Query 初始化为什么会让 Feature response branch 发生突变？

**记录日期**：2026-09-05  
**状态**：Open research question；已有确定的实现错误修复，但理论解释尚未闭合  
**模型**：TabU-v2 / TabUR cell-as-query 结构  
**证据等级**：DGX2 `local_unissued` 直接拟合诊断；不是 ICL、泛化或 capability claim

## 1. 为什么记录这个问题

这个现象值得单独保存，因为它同时触及三层边界：

1. 一个看起来几乎为零的 Query 初始化，为什么会显著改变拟合能力？
2. 为什么把 `lambda_F` 从 0 直接切到 1 会导致拟合崩溃，而 warm-up 又能部分恢复？
3. 这到底是代码实现错误、优化器路径问题，还是 $F_a$ 不应该进入 response law 的理论问题？

当前不能把它简化成“初始化太大”或“Feature branch 不应该存在”。本记录的目标是把事实和推测分开，防止一个有趣但尚未闭合的现象被过早写成理论结论。

## 2. 一句话描述现象

在 diabetes 固定 episode 上，`lambda_F=lambda_U=1` 时，`q_std=3e-8`：

- 旧实现能够得到很低的 MSE，但 Feature/Unit states 实际被错误地清零；
- 修正 gate 后，Feature/Unit states 真正被激活，但拟合崩溃；
- 只开启 `lambda_F=1, lambda_U=0` 时，$F_a$ 单独就已经足以造成严重 seed 不稳定；
- 使用从 `lambda_F=0` 到 `1` 的 warm-up 后，拟合能力显著恢复。

因此，原来的“近零初始化成功”不是一个可靠的结构学习证据，而是由数值实现和优化路径共同造成的混合现象。

## 3. 固定实验协议

所有下列结果使用同一份固定数据和训练协议：

```text
dataset: sklearn.load_diabetes
data seed: 20260905
sample: 256 rows without replacement
context: first 128 rows
query: last 128 rows
predictors: all 10 features visible
response: context visible, query hidden and physically zeroed
model width: d_model=32, K=4, n_blocks=2
optimizer: Adam(lr=3e-3), gradient clipping=1.0
objective: MixedObjective(context_standardized)
training length: 240 steps unless warm-up is explicitly stated
metric: query raw-response MSE
```

这是一项 supervised masked-fit 诊断。query truth 只进入 `TruthSidecar` 和 objective，不进入 forward carrier。

## 4. 当前结构和实现

### 4.1 Query banks 如何进入 carrier

模型中有三组主要参数：

```python
unit_query = nn.Parameter(torch.empty(K, d_model))
feature_query = nn.Parameter(torch.empty(K, d_model))
response_base = nn.Parameter(torch.empty(K, d_model))
```

默认全部以 `std=0.02` 初始化。实验中只把 `feature_query` 和 `unit_query` 缩放到目标 `q_std`，没有缩放 `response_base` 或 Transformer 其他参数。

`feature_query` 沿 Feature Query rows broadcast，`unit_query` 沿 Unit Query columns broadcast。它们不是事实 source，只是 receiver states。

### 4.2 当前 response law

实现直接从最终 carrier 读取 Feature/Unit states：

```text
F_a = final feature-query carrier bank
U_r = final unit-query carrier bank
```

然后形成：

$$
A_{ra}=W+\lambda_F\left(F_a+\lambda_U U_r\right),
\qquad
\bm z_{ra}=A_{ra}\bm c_{ra}.
$$

对应实现位于 `src/tabu_lab/models/tabu_v2.py` 的 `_response_field`。这里采用的是设计文件中最小的 identity bank readout，没有额外的 Feature/Unit projection 或尺度匹配。

### 4.3 OMAB 的 normalization 边界

OMAB 对 attention 输入使用 `receiver_norm` 和 `source_norm`，对 FFN 输入使用 `ff_norm`，但最终状态仍保留 residual：

$$
H_{\mathrm{out}}
  =H_{\mathrm{in}}+
    \text{gated-attention update}+
    \text{gated-FFN update}.
$$

因此，Transformer 内部有 pre-norm，不等于最终读取出来的 $F_a/U_r$ 已经归一化。当前 response boundary 直接读取 raw final states。

## 5. 第一层确定原因：presence gate 的数值实现错误

旧实现使用：

$$
g(q)=1-\frac{\tau}{\tau+\lVert q\rVert^2}.
$$

理论上它等价于：

$$
g(q)=\frac{\lVert q\rVert^2}{\tau+\lVert q\rVert^2}.
$$

但当 $\lVert q\rVert^2\ll\tau$ 时，float32 中的减法会发生 catastrophic cancellation。

在 `q_std=3e-8` 时，输入 Query 的 RMS 确实约为 $3\times10^{-8}$，但旧代码计算出的 gate 是精确 `0`。第一层 OMAB 之后，$F/U$ state 也变成精确 `0`。

在 `q_std=5e-8` 时，gate 开始出现约 `1.19e-7` 的第一个 float32 有效台阶，$F/U$ 又恢复为非零状态。

这解释了旧扫描中非常尖锐的分界：

```text
q_std = 3e-8: branch 被错误清零，拟合看起来很好
q_std = 5e-8: branch 真正激活，拟合突然崩溃
```

修复位置：`src/tabu_lab/primitives/oattention.py` 的 `presence_gate`。

修复方式是直接计算 ratio，同时保留精确零和大值饱和处理。回归测试位于 `tests/unit/test_oattention.py`。

这部分已经不是假设，而是已复现、已定位、已修复的实现错误。

## 6. 第二层原因：小初始化不等于小更新

修正 gate 后，单个 seed 的训练轨迹如下：

```text
step 0:
  qF ≈ 2.97e-8, F ≈ 2.99e-8, ΔA ≈ 4.6e-8
  coordinate_rms ≈ 0.087

step 1:
  qF ≈ 3.00e-3, F ≈ 0.60, ΔA ≈ 0.90
  coordinate_rms ≈ 15.5

step 10:
  qF ≈ 1.1e-2, F ≈ 1.11, ΔA ≈ 1.98
  coordinate_rms ≈ 77

step 240:
  qF ≈ 2.9e-2, F ≈ 3.01, ΔA ≈ 4.96
  coordinate_rms ≈ 396
```

这里的关键是 Adam 的第一步近似为：

$$
q_1=q_0-\eta\frac{g_0}{|g_0|+\epsilon}.
$$

只要梯度非零，更新尺度主要由学习率决定，而不是由 $q_0$ 决定。因此 $3\times10^{-8}$ 的初始化在第一步就可以跳到约 $3\times10^{-3}$。

同时，Query state 通过 residual identity 直接进入 $F_a$，所以 attention gate 很小并不能保证 response law 的梯度也很小。

当前观察到的是一个真实的“优化器放大”过程，而不是一个持续处于小扰动 regime 的过程。

## 7. 第三层原因：response operator 的尺度不匹配

初始化时大致有：

$$
\lVert W\rVert\sim 10^{-2},
\qquad
\lVert F_a\rVert,\lVert U_r\rVert\sim 1
$$

一旦 Query branch 被 Adam 推离零，

$$
A_{ra}=W+F_a
$$

就不再是“小修正”，而是由 $F_a$ 主导的全新 operator。

随后 terminal 使用 same-column RBF routing：

$$
w_{rs}\propto
\exp\left(-\frac{\lVert z_{ra}-z_{sa}\rVert^2}{2h^2}\right).
$$

当 coordinate 从 `0.087` 变成 `15`、`77` 或 `396` 时，routing 权重会指数级塌缩，visible support 退化，MSE 随之爆炸。

因此“一个小项造成巨大影响”的完整链条是：

```text
小 q 初始化
  -> presence gate 保留非零梯度
  -> Adam 第一步按 lr 大幅移动 q
  -> residual identity 让 F_a 迅速变成 O(1)
  -> A=W+F_a 改变 response coordinates
  -> RBF routing 指数塌缩
  -> 拟合能力崩溃
```

## 8. 已完成的对照实验

### 8.1 旧 gate：看似成功的近零分支

旧实现下，`q_std=1e-8` 与 `q_std=0` 的 5-seed MSE 完全一致，约为：

```text
0.0195, 0.0203, 0.0356, 0.0541, 0.3864
```

这不是 $F/U$ 学习成功，而是它们被数值 gate 关闭后退化为 $A=W$。

### 8.2 修正 gate，开启 F/U

`q_std=3e-8, lambda_F=1, lambda_U=1`，5 seeds：

```text
MSE: 6118.6, 11500.0, 5867.8, 11006.4, 5781.5
F RMS: 3.01, 3.16, 3.34, 2.47, 4.10
U RMS: 3.02, 3.24, 2.88, 4.30, 2.71
```

### 8.3 只开启 Feature branch

`q_std=3e-8, lambda_F=1, lambda_U=0`，5 seeds：

```text
MSE: 160.9, 6218.1, 1811.4, 753.8, 7033.3
F RMS: 1.02, 3.28, 1.18, 0.97, 3.02
```

这说明 $U_r$ 会进一步放大问题，但 $F_a$ 单独已经不稳定。也就是说，问题不能归因于 Unit branch 独有的相互作用。

### 8.4 Warm-up

使用修正后的 gate，在前 240 steps 令 $\lambda_F$ 从 0 线性增加到 1，再保持 240 steps。

对 `q_std=3e-8, lambda_U=0`，3 seeds 得到：

```text
MSE: 4.3e-4, 0.053, 39.96
correlation: 1.0000, 1.0000, 0.9972
```

相比直接跳到 $\lambda_F=1$ 的 `160–7000` MSE，warm-up 显著改善了优化路径，但还不是完全 seed-stable 的解决方案。

## 9. 当前可以说什么，不能说什么

### 已确认

1. 旧 `presence_gate` 在小 float32 信号上会错误地把非零 Query 清零。
2. 修正 gate 后，$F_a$ 确实会被训练并迅速离开近零区域。
3. Adam 的第一步与初始化幅度弱相关，可以把极小 Query 推到约 `1e-3`。
4. 当前 OMAB final state 没有被 response readout 前的最终归一化保护。
5. $F_a$ 的直接 additive readout 会改变 coordinate 和 RBF routing，造成严重失稳。
6. warm-up 是有效的 continuation path，说明存在可以训练的参数路径。

### 尚未确认

1. $F_a$ 在理论上是否应该存在于 response law 中。
2. 直接加法本身是否错误，还是只缺少 scale-matched bank readout。
3. 如果使用显式 projection、固定小尺度或 centered bank，$F_a$ 是否可以稳定提升拟合。
4. warm-up 是否只是优化器技巧，还是反映了 response-law 的真实分阶段学习结构。

当前最稳妥的表述是：

> $F_a$ 不是已经被证明“不应该存在”；但未经尺度匹配、直接把最终 Feature carrier 当作 response operator 加到 $W$ 上，当前参数化是不稳定的。

## 10. 下一步最小验证

为了不引入过多复杂设计，优先做三个单轴实验：

1. **优化器对照**：保持修正 gate 和 response law 不变，只把 Adam 换成小步长 SGD，验证第一步跳变是否是主因。
2. **尺度对照**：保持 optimizer 不变，只给 $F_a$ 一个固定的 response-scale readout，验证是否能恢复稳定拟合。
3. **语义对照**：比较 $A=W$、$A=W+F_a$ 和一个经过 scale-matched 的 $A=W+\widetilde F_a$，不要同时改变 $U_r$、terminal 和数据协议。

在这些实验完成前，不应把“近零初始化成功”解释成从噪声中学出了 Feature structure，也不应仅凭当前失稳结果删除 $F_a$。

## 11. 实验记录与复现入口

DGX2 实验目录：

```text
/home/cms/tabu-v2-real-fit-20260905/
```

主要 artifact：

```text
real-fit-diabetes-q-init-window-20260905.json
real-fit-diabetes-q-init-fine-window-20260905.json
real-fit-diabetes-q-init-3e-8-5seed-20260905.json
real-fit-diabetes-gate-fix-q3e-8-20260905.json
real-fit-diabetes-lambdaF1-lambdaU0-20260905.json
real-fit-diabetes-warmup-f-only-20260905.json
```

这些 artifact 是本地诊断记录。它们支持本问题的复盘，不自动升级为正式 capability 或 publication evidence。

## 12. 关联设计文件

当前 canonical structural design：

```text
/Users/cms/.openclaw/workspace/projects/causal-superintelligence/TabU/tabu-genesis-seed/TABU_STRUCTURAL_DESIGN.tex
```

其中 response law 明确写作：

$$
A_{ra}=W+\lambda_F(F_a+\lambda_U U_r).
$$

本记录不是对 canonical design 的替换，而是记录该公式在真实实现、float32 gate、Adam 更新和 terminal routing 联合作用下产生的开放问题。

