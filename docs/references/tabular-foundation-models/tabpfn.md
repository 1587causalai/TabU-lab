# TabPFN：prior-data fitted 的表格预测路线

> 研究卡状态：`第一轮 source read`
>
> 最后核验：2026-08-26

## Canonical sources

- [PriorLabs/TabPFN repository](https://github.com/PriorLabs/TabPFN)
- [Prior Labs documentation](https://priorlabs.ai/docs)
- [TabPFN original paper](https://arxiv.org/abs/2207.01814)
- [TabPFN local demo notebook](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)

## 第一轮理解

TabPFN 的核心参考价值不是“Transformer 可以做表格”，而是把模型训练成一个面向任务分布的 predictor：模型先在合成数据或 prior 上学习，再把一个具体任务的 support / train rows 作为上下文，用一次前向过程预测 unseen rows。当前官方仓库把这一类模型描述为在 synthetic datasets 上训练、对未见真实数据集进行预测，并以单次 forward pass 为主要使用方式。

这里的“prior”必须继续拆开研究：

- prior 生成了什么类型的数据生成过程；
- support rows 如何进入模型；
- feature schema、target 和 missingness 如何编码；
- 训练时的任务分布与评测时的真实表格是否匹配；
- 计算预算、上下文长度和数据集规模如何限制泛化。

## 需要从论文和代码确认

本卡目前只完成 repository / paper 入口级阅读，以下内容不能从 README 直接推断：

- 当前 TabPFN-3 与原始 TabPFN 论文的架构连续性；
- 训练 prior 的确切分布、采样策略和数据泄漏防护；
- 不同版本的上下文长度、样本规模、特征处理和校准行为；
- benchmark 中的 baseline、数据清洗、调参预算和统计稳定性。

## 对 TabU 的关系

### 可以借鉴

- 把“表格任务”明确写成任务分布，而不是只写一个静态数据集。
- 把 support / context 与 query 分开记录，保持训练和评测的 leakage boundary。
- 把 prior、数据生成机制和模型能力主张放在同一份 receipt 里。

### 不能直接搬用

- prior-data fitted 的成功不等于 Unit-as-Primitive 语义成立。
- 单次前向预测不等于 learned abduction，也不等于 causal identification。
- TabPFN 的任务 prior 不能直接替换 USL01 中的 Unit、机制和噪声定义。

### 需要本地实验回答

- TabU 的 Unit-conditioned context 是否比普通 support rows 携带更多可识别的语义？
- 同一 Unit 下改变 query/context 组织，响应是否保持预期的结构？
- split-before-compile、context/query leakage 和 synthetic prior 的边界如何在 TabU-lab receipt 中固定？

## 最小后续阅读

1. 读原始论文的方法与 prior 生成部分；
2. 对照当前仓库的 data / model / inference 入口；
3. 选一个小表格，固定 split、seed 和 compute，做与 TabU 语义无关的 baseline reproduction；
4. 只有完成上述步骤，才把具体架构事实写入 TabU 的设计文档。
