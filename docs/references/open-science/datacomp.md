# DataComp：把数据设计变成固定协议下的实验

> 研究卡状态：`方向卡`
>
> 最后核验：2026-08-26

## Canonical sources

- [ML Foundations DataComp repository](https://github.com/mlfoundations/datacomp)
- [DataComp website](https://www.datacomp.ai/)
- [DataComp paper](https://arxiv.org/abs/2304.14108)

## 它是什么

DataComp 官方仓库将其定义为设计预训练数据集的 competition：参与者主要改变数据集设计，模型架构和超参数固定，随后在下游任务上评估。它提供数据池、filtering / BYOD 赛道以及不同计算规模，让数据选择本身成为可比较的变量。

## 对 TabU 的启示

DataComp 对 TabU 的价值不在于 CLIP 或图文数据本身，而在于一个实验设计原则：当我们想研究数据或 prior 的作用时，应尽量冻结模型、训练预算和评测协议，让变化集中在被研究变量上。

### 可以借鉴

- 把数据 recipe、过滤规则、版本和预算作为一等实验对象。
- 为 data / prior ablation 固定模型与评测协议。
- 明确 filtering、external data 和 compute scale 的边界。

### 不能直接搬用

- DataComp 的多模态数据池不是 TabU 的表格语料协议。
- 固定架构只适用于特定 ablation；它不能替代 TabU 的整体模型设计。
- leaderboard 结果仍需检查数据许可、泄漏、seed 和统计不确定性。

## TabU-lab 待回答

- 我们的 tabular pretraining corpus 是否可以定义一个可审计的 data recipe？
- 如何区分数据质量增益、模型结构增益和训练预算增益？
- split-before-compile 如何与数据过滤、normalizer、codebook 和 synthetic prior 组合，避免 context/query leakage？
