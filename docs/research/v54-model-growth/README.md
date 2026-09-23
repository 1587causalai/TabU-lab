# TabU Token dynamic 复制加深与数学分析

2026-09-22。主问题是：只把训练好的 Token dynamic 复制多份并沿深度串联，保留其他组件，能否获得较好的拟合初始化与后续训练效果？这里按当前代码将 Token dynamic 对应到处理完整 carrier 表的 backbone。

本文是基于当前 V5.4 公式和代码的推导与小型 CPU 前向核验，不改变当前训练、设计默认或实验预算。已证明的构造、数值检查与未验证的训练收益分开记录。

**当前判断：复制已训练的 Token dynamic 在结构上可行，但首次真实四倍复制的固定 Query 起点明显退化；训练能否恢复仍待曲线验证。** 下文的幂等反例限定数学保证，不能单独替代真实训练；第4节等距扩宽是另外一个数学补充，不是此次四倍扩深。

新增训练诊断：四倍模型到 2026-09-23 03:02 UTC 的 3,306 次实际更新仍全部有限，但裁剪前梯度最高 `1.44e12`，loss 最大 `1.23e5`，出现明显数值失稳。它使用与另外两路 old120 v2 相同的优化后冻结训练源码，CUDA 的非零目标 readout 已实际启用；[逐项代码和早期窗口对比](../../../experiments/local/v54-nano-dynamics4x-old120-jarvis-20260922/code-comparison-20260923.md)列出可核对证据。当前没有后续固定 Query，不能用波动的单步 loss 宣称已恢复或已经失败。W&B 的一度 `Failed` 是[独立监控 SSH 超时](../../../experiments/local/v54-nano-dynamics4x-old120-jarvis-20260922/failure-diagnosis-20260923.md)，不是训练进程的终态。

2026-09-23 的[独立 Jarvis 实验](../../../experiments/local/v54-nano-dynamics4x-old120-jarvis-20260922/README.md)将 dustinstudio Nano 已落盘的 2 层 backbone 按 `[0,1,0,1,0,1,0,1]` 复制为四份独立参数，保持 width128／4 heads／Unit0，重置 AdamW、RNG 和训练游标。同一 Jarvis CUDA FP64、同一 120 表固定 Query bank 的起点对比：[原 Nano](../../../experiments/local/v54-nano-dynamics4x-old120-jarvis-20260922/parent-fixed-cuda.json)数值 R² 中位数 0.9786、nominal 准确率 0.8978、ordinal 准确率 0.5625；[四倍模型](../../../experiments/local/v54-nano-dynamics4x-old120-jarvis-20260922/grown-fixed-cuda.diagnostic-14d.json)分别为 -0.0837、0.5700、0.4203。数值 57/57 表均变差，nominal 42/43 变差、1 张持平，ordinal 19/20 变差。该对比采用仅时间上限不同的诊断 manifest；正式任务延长了时间护栏，重新冻结身份并启动。本输出的[正式 step0 固定 Query](../../../experiments/local/v54-nano-dynamics4x-old120-jarvis-20260922/status-20260923T022705Z.json)已完整读回，逐表指标与诊断四倍模型一致。训练于 UTC 2026-09-23 02:16 启动，不能把起点下降写成训练后结论。

## 当前研究假设：复制动态计算，复用编码和读出

设 D 为已经训练好的整段 Token dynamic。构造 k 个独立参数副本，各自初始化为同一组已训练权重：

\[
H_0=E(X),\qquad H_j=D_{\theta_j}(H_{j-1};\mathsf M),
\quad\theta_j\leftarrow\theta_{\rm trained},\quad
\widehat y_k=R_\phi(H_k;X_{\mathcal V}).
\]

编码器 E 和后续 Unit/LL 组件 R 保留其原有结构与初始化，不复制多份；只有 backbone 的动态计算加深。读出规则保持不变，但 shared-LL 的斜率本来就会在每次 forward 中根据当前表示与真实可见支持重新求解。

该假设有三项实现依据：

1. 每段动态计算接收和返回同形状的 carrier 表，Cell/Unit/Feature 地址与语义保持，已训练模块可以直接接到上一段后面。
2. 所有段复用同一可见 source mask，Query 不转为证据，Null 规则保持，无需改变编码或 episode 定义。
3. LL 根据新的 Cell/Unit 表示重新建立权重和拟合斜率；与固定 learned head 相比，新表示无需继续匹配旧的输出头系数。这给维持或恢复拟合提供了有利条件。

需要实验回答的主要问题是：复制块接收到更深表示后，是否仍保有有效信息与合适的 Unit 几何；LL 重新拟合不能保证修复任意表示退化。上面的真实四倍起点对比已发现显著退化；下一步看继续训练是否恢复、何时恢复以及实际成本，再决定是否推广多份复制。此次预算来自用户明确授权的每表 8,192 次正常更新，不能据代数推导追加新实验。

## 1. “复制”需要明确操作对象

设编码器为 E，carrier backbone 为 F，包含 Unit 支路及 shared-LL 的读出为 R，原预测为

\[
M(X)=R(F(E(X))).
\]

下文比较均固定同一 episode 的数据、可见支持答案、schema、masks、codebook 与声明的超参数。F、R 隐含读取这些固定侧信息；复合式只是简写，shared-LL 的实际入参并非只有 carrier。

| 操作 | 数学含义 | 起点预测是否必然保持 |
| --- | --- | --- |
| 已训练块沿深度串联 | `R ∘ F^k ∘ E`，或重复 Unit 块 | 不保证；这是新的初始化策略 |
| 新增恒等残差块 | 新块的残差出口为零 | 可以，前提是边界/掩码/有限值条件保持；Small-H4 使用了这一路线 |
| 独立预测副本平均 | `sum(M_i(X))/k`，初始所有 `M_i=M` | 相同副本起点相同，但这是一般模型也能做的并联平均 |
| 本文等距扩宽 | `d→kd`，状态和参数按第4节变换 | 精确实数上保持；有小型 FP64 验证 |

复制的参数可以是独立副本，也可以始终共享。前者增加参数自由度；后者是重复使用同一算子。不能用重复引用同一模块误当成独立复制。完整 TabU 返回预测结构，不是同维 carrier；把预测重新当作可见证据输入还涉及新的反馈协议，不能由层复制授权推出。

## 2. 深度复制：零闭合不等于幂等

在相关状态域上，保持整个隐藏映射需要

\[
F^2=F.
\]

这是幂等性；由归纳可得 `F^k=F`。若只要求最终预测相同，条件可以更弱，为 `R ∘ F^k = R ∘ F`；因此单凭 carrier 改变，不能断言 decoder 一定改变。

对残差映射 `F(h)=h+g(h)`，

\[
F(F(h))=h+g(h)+g(h+g(h)),
\]

所以隐藏映射的复制保持条件是

\[
g(h+g(h))=0.
\]

第一次输出必须已进入固定点集合。当前训练目标没有施加这一条件。`F(0)=0` 只约束零点；置换等变、Query 不供源、Null 闭合可在串联时继续成立，但不推出这个幂等条件。

### 当前 OMAB 内就有反例

取单个非零 receiver/source `h`，令 `Q=K=0`、`V=O=I`，关闭 FFN，presence 读出为恒等。记

\[
p(h)=\frac{\|h\|^2}{\tau_{\rm pres}+\|h\|^2}.
\]

实际 [OMAB](../../../src/tabu_lab/models/restoration/backbone.py) 给出

\[
F(h)=\left(1+\frac{p(h)^2}{\eta+p(h)}\right)h.
\]

非零时系数严格大于 1，再执行会继续改变状态；与此同时 `F(0)=0` 完全成立。单线程 CPU 验证的 `||h||=1`、`tau_pres=eta=1` 情形，第一次和第二次输出不同，见回执。

### shared-LL 也不会自动抵消这种变化

构造一个合法 Unit OMAB：关闭 attention 输出，FF presence 投影为恒等、RMS gamma 为全一，设置 FFN

\[
W_1=\begin{bmatrix}I\\-I\end{bmatrix},\qquad
W_2=\begin{bmatrix}I&-I\end{bmatrix}.
\]

由于 `GELU(z)-GELU(-z)=z`，对相同范数的非零 Unit，该块为 `T(u)=c u`，其中

\[
c=1+\frac{\rho(u)}{\sqrt{\|u\|^2/d+\varepsilon}}>1.
\]

取两个支持 Unit 为 `-av,+av`，Query Unit 为 `+av`，`||v||=1`。所有 Cell 回归特征相同，shared slope 为零；支持响应分别为 0 和 1。固定带宽为 t，预测为

\[
\widehat y=\frac1{1+\exp(-2a^2/t^2)}.
\]

加入这个 Unit 块后，

\[
\widehat y'=\frac1{1+\exp(-2c^2a^2/t^2)}\ne\widehat y.
\]

两者都可写在同一合法数值答案仿射线上；答案投影不能消除此差异。该结论对任意 a>0 成立，将 a 替换为 c(a)a，再执行相同块仍严格改变预测，因此 `R ∘ T² ≠ R ∘ T`。当前数值验证直接调用 shared-LL，核对的是上述一次几何变化 `R(U)` 与 `R(T(U))`；再复制的结论来自同一公式，不把它描述为额外的数值测试。该反例证明没有普遍结构定理，不宣称实际训练检查点必然出现同等变化。

## 3. 原样复制仍值得作为增长候选

不保持函数不等于没有训练价值。已学残差块可以提供有用的新层初始化；用户当前允许小的数值/起点变化，实际应检查复制后的固定 Query 退化程度、有限值、吞吐和随后恢复/改善曲线。

如希望减小插入扰动，可显式考虑 `T_alpha(h)=h+alpha g(h)`。例如把一个残差步替成两个半步，若 g 在相关线段上 L-Lipschitz，则

\[
T_{1/2}(T_{1/2}(h))-T_1(h)
=\frac12\left[g(h+g(h)/2)-g(h)\right],
\]

\[
\|T_{1/2}^2(h)-T_1(h)\|\le\frac L4\|g(h)\|.
\]

它给出“残差小、变化慢时可以接近”的条件，不是一般严格等价。这里的缩放针对定义好的单个残差映射；不能把完整网络所有参数除以二，尤其不能任意缩放 Q/K、presence 或 ridge。

## 4. 一个可以成立的正面构造：等距复制宽度

令 k 为正整数，定义复制嵌入

\[
T=\frac1{\sqrt k}\begin{bmatrix}I_d\\\vdots\\I_d\end{bmatrix}
\in\mathbb R^{kd\times d},\qquad T^\top T=I_d.
\]

新 carrier `h'=Th` 是 k 份原 carrier 的拼接，每份缩放 `1/sqrt(k)`。它满足

\[
\|Th\|=\|h\|,\qquad
\langle Th,Tz\rangle=\langle h,z\rangle,\qquad
\|Tu-Tv\|=\|u-v\|.
\]

固定层数、表地址、masks、inducing 槽数 S、答案编码维度128及随机 codebook；将 carrier width、heads、FFN width 同时乘 k。每个 head 的维度 `d/H` 不变。记 `D_k(W)=I_k⊗W`，逐个 OMAB 变换如下：

| 参数 | 新参数 |
| --- | --- |
| 输入投影 | `T W_enc` |
| Cell/Unit/Feature query seeds、inducing slot seeds | 每个 seed 左乘 T |
| 两个 presence 投影 | `D_k(W_P)` |
| Q、K 投影 | `sqrt(k) D_k(W_Q)`、`sqrt(k) D_k(W_K)` |
| V、O 投影 | `D_k(W_V)`、`D_k(W_O)` |
| RMS gamma | 原向量重复 k 次 |
| RMS 稳定项 | `norm_eps/k`，仅此项；不改 codec epsilon |
| FFN 第一、第二线性层 | `D_k(W_1)`、`D_k(W_2)/sqrt(k)` |
| presence 阈值、参考质量 eta、匹配带宽、LL ridge | 保持不变 |

这里只列当前无偏置、共享 presence、无 dropout 的实现合同；不同归一化/偏置/激活配置须重新推导。

### Attention 与 FFN 为什么能保持

Presence 读出满足 `W_P' T=T W_P`，因此范数及 receiver/source presence 不变。每个复制 head 的 Q/K 与原 head 相同，V 为原值的 `1/sqrt(k)`；每头维度和源数量不变，内容分数、参考质量与注意力权重不变。经复制后的 O 投影，attention 更新恰为原更新左乘 T。

RMS 需单独处理。对不含 gamma 的 RMS 有

\[
\operatorname{RMS}_{kd,\varepsilon/k}(Th)
=\begin{bmatrix}\operatorname{RMS}_{d,\varepsilon}(h)\\\vdots\\
\operatorname{RMS}_{d,\varepsilon}(h)\end{bmatrix}.
\]

因为新坐标的均方为原值的 `1/k`。复制 gamma、FFN 第一层后，各副本的 GELU 输入相同；第二层的 `1/sqrt(k)` 将 FFN 输出变回 T 倍。因此在有限精确实数运算下

\[
\boxed{\operatorname{OMAB}'(TH)=T\operatorname{OMAB}(H).}
\]

T 逐个地址作用；Null 和源掩码不变。collect seeds 也按 T 变换，空证据 gate 保持，故该等式可通过 direct/inducing column、row、backbone 多层、Unit 多层归纳到最终 carriers 与 Units。

### shared-LL 读出为什么也保持

这里采用当前默认 `regression_width=None`，Cell 回归特征也是 carrier。Unit 距离保持，所有 Gaussian 权重保持。记原共享协方差与交叉矩为 Sigma、J，

\[
B=J^\top(\Sigma+\lambda I_d)^{-1}.
\]

等距复制得到

\[
\Sigma'=T\Sigma T^\top,\qquad J'=TJ.
\]

利用

\[
(\Sigma'+\lambda I_{kd})T=T(\Sigma+\lambda I_d),
\]

可得

\[
B'=B T^\top.
\]

局部答案均值不变，Cell 偏移 `delta'=T delta`，故

\[
\boxed{\widehat e'=\bar e+B'\delta'
=\bar e+B T^\top T\delta=\widehat e.}
\]

答案 codec 与 decoder 保持，因此最终原值预测保持。该证明包含 ridge，不需要假设支持数大于扩宽后的维度。浮点差异与离散近邻 tie 的数值敏感性仍需实测。

**不能省略这些变换，直接堆参数就声称保持。** 不缩放的 k 份拼接会使平方距离乘 k；presence、attention 温度、RMS 与 ridge 的有效尺度也可能改变。新模型是明确的扩宽变体，不会自动等于标准 Medium/Large preset。

## 5. 当前 V5.4 代码上的前向核验

[验证脚本](verify_replication.py) 与[含深度反例的完整回执](verification-with-depth-20260922.json)已经保存；回执记录脚本及相关源码 SHA。

- CPU FP64、Torch 2.11.0、单线程；没有 optimizer、训练步骤、远端操作或真实 checkpoint 转换。
- 6 行、numeric/nominal/ordinal 三列，含 Null 和全 Query 行。
- direct / inducing 两种路径，k=2/3，共四组；1 层 backbone、1 层 Unit，显式微型配置，不是标准 Nano 的性能测试。
- presence 权重加入非恒等扰动，RMS gamma 非恒定，避免只验证默认初始化的特殊情形。
- 最大 carrier/Unit 差约 `9.58e-16`，shared slope 差约 `3.95e-14`，编码差约 `4.10e-14`，数值 decoder 差约 `4.74e-14`；nominal/ordinal 解码一致。
- 合法 OMAB 串联反例与 Unit 几何改变后的最终 shared-LL 预测，也与解析公式吻合。

从仓库根在已有本地 CPU Torch 环境复核，输出必须是新文件：

```bash
/opt/miniconda3/bin/python docs/research/v54-model-growth/verify_replication.py \
  --output /tmp/tabu-growth-verification-new.json
```

这是代数诊断环境，不替换正式训练运行时。真实检查点和 Jarvis CUDA FP64 的四倍复制起点另见本页顶部实验链接；它们不改变本节代数核验的范围，也尚未证明继续训练的拟合收益。

## 6. 复制之后的训练仍有三个问题

1. **函数保持不等于优化器保持。** Widening 改变参数坐标、梯度尺度及 AdamW 状态；不能机械复制 moments 并声称轨迹等价。当前小型验证只有 forward。
2. **完全相同副本可能保持训练对称。** 在相同输入、确定性优化和对称状态下，重复分支可能长期做相同工作。独立参数并不自动带来分工。后续可研究受控微扰或非对称学习设置，并测起点变化及收益；这些尚未执行。
3. **宽度乘 k 不等于参数和成本只乘 k。** 当前 dense 方阵通常增至约 k² 参数，默认全宽 LL 求解系统由 d 变成 kd。验证中的 direct 微型模型由410,368参数增至1,607,168（k=2）或3,590,400（k=3）。扩宽不能直接称为省计算。

因此当前可以提出两条独立实验候选：已训练块直接复制加深；按本文映射保持起点的等距扩宽。两者目的和成本不同，不应在一个实验里同时改变却只归因给“复制”。

## 7. 哪些数学性质不能混用

- 答案空间投影 `P_a²=P_a` 约束的是 decoder 的投影，不是整个 backbone 或 shared-LL 的幂等性。
- shared-LL 的有效系数满足 `S 1=1`，保证仿射闭包；即使读取地址与支持地址相同、S 为方阵，也不推出 `S²=S`。一般矩形读出系数的平方没有定义。
- Query projectivity 固定证据与参数，讨论增加接收地址；不讨论额外执行已训练块。
- V5.4 的 source replication 固定 receiver 和投影、复制 source entries。固定 eta 时混合从 `A/(eta+Z)` 变为 `kA/(eta+kZ)`，本身通常也不精确相同；它和本文不改变源数量的特征扩宽不是同一个操作。

文稿依据：[V5.4 TeX](../../design/TabU_V5p4_Unified_Composition.tex)，有关答案投影、shared-LL、OMAB、zero closure、source replication 与 projectivity 的章节。实现依据：[OMAB/backbone](../../../src/tabu_lab/models/restoration/backbone.py)、[编码器](../../../src/tabu_lab/models/restoration_v53/encoding.py)、[Unit 距离](../../../src/tabu_lab/models/restoration/readout.py)、[shared-LL](../../../src/tabu_lab/models/restoration_v53/readout.py)。

## 8. 与已有模型增长工作的关系

不能宣称“普通模型一般不能复制增长，TabU 独有”。[Net2Net（ICLR 2016）](https://arxiv.org/abs/1511.05641)已经研究把已有网络转换为更宽或更深且保持函数的网络；[Stacking Your Transformers（NeurIPS 2024）](https://arxiv.org/abs/2405.15319)直接研究 Transformer 深度 stacking 的训练效果。[Staged Training（ICML 2022）](https://proceedings.mlr.press/v162/shen22f.html)进一步区分 loss 保持与训练动态保持。

本文的实际增量是把复制增长逐项落实到 **TabU 当前 presence、固定参考质量、RMS、inducing/Unit 支路、固定答案空间与 shared-LL ridge** 的完整合同，并给出代码上的有限前向核验。它不是新颖性声明，也不是复制增长已改善 TabU 拟合的实验结论。
