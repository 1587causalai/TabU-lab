# 表格 foundation model 参考线

这一层只讨论模型与任务问题，不讨论“如何运营一个开放研究仓库”。Marin 的方法卡在上一级；这里比较的是表格模型如何使用 prior、context、missingness、任务标签和跨任务泛化。

## 当前参考对象

| 工作 | 主要问题 | 我们要看什么 | 当前状态 |
|---|---|---|---|
| [TabPFN](tabpfn.md) | prior-data fitted / in-context prediction | 合成 prior、上下文形式、单次前向预测、任务边界 | 第一轮 source read |
| [LimiX](limix.md) | 统一结构化数据任务 | 任务统一、缺失值、样本与特征交互、训练数据与 claim | 第一轮 source read |

## 共同比较维度

读这些工作时固定回答以下问题：

1. **任务是什么**：classification、regression、imputation 或生成是否共用一个学习对象？
2. **上下文是什么**：support / train rows、feature schema、task description、prior 还是 Unit 语义？
3. **泛化到哪里**：新行、新列、新数据集、新任务，还是新的数据生成机制？
4. **缺失值是什么**：数值上的空、显式 mask、未知状态，还是生成机制的一部分？
5. **训练数据来自哪里**：真实数据、合成数据、混合数据，split 是否可能泄漏？
6. **证据如何获得**：固定 benchmark、多个 seed、compute budget、baseline 和是否有独立复核？

## 与 TabU 的边界

TabPFN 与 LimiX 可以帮助我们理解表格 foundation model 的能力空间，但它们不会自动回答 TabU 的 Unit 语义、因果生成假设或响应律问题。尤其要保持：

- `Unit` 不是普通 sample token 的同义词；
- 任务泛化不等于因果识别；
- 缺失值处理不等于 null semantics 已成立；
- benchmark 胜出不等于 TabU 的理论主张成立。
