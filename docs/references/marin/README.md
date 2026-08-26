# Marin：开放研究过程的 source map

> 研究卡状态：`第一轮 source read`
>
> 最后核验：2026-08-26

Marin 是我们当前最重要的**研究过程参考**，不是 TabU 要照抄的语言模型架构。当前官方 README 将 Marin 描述为 foundation-model 的 research program、software platform 和 community，并把 open development 放在核心位置：过程、实验和决策随着研究发生而记录，失败实验也属于记录的一部分。

## Canonical sources

- [Marin canonical repository](https://github.com/marin-community/marin)
- [Marin project site](https://marin.community/)
- [Marin documentation](https://marin.readthedocs.io/en/latest/)
- [First experiment tutorial](https://github.com/marin-community/marin/blob/main/docs/tutorials/first-experiment.md)
- [Experiment reports index](https://github.com/marin-community/marin/tree/main/docs/reports)
- [Open development article](https://openathena.ai/blog/open-development-of-frontier-ai/)
- [原始 Percy / Marin 综合稿](../../percy-liang-marin-open-science.md)

## 要深入看的不是一句“开放”

### 1. 研究对象

Marin 的参考价值首先在于它把训练研究拆成可追踪的过程对象：数据处理、tokenization、pretraining、posttraining、evaluation，以及它们之间的依赖关系。官方 README 的示例把 experiment 表达为一组带依赖的 steps，并按 topological order 执行。

### 2. 证据链

我们要实际跟读一条链，而不是只读首页叙事：

```text
问题 / issue
    → 实验代码 / PR
    → 配置、数据和执行图
    → run / curve / WandB 或其他报告
    → retrospective / 失败与后续决策
```

这条链是否完整、每一环是否可复现，需要针对具体 experiment 检查。不能因为仓库公开，就自动认为每个历史结果都已经被我们复核。

### 3. 对 TabU-lab 的可迁移部分

| Marin 观察 | TabU-lab 里的最小借鉴 |
|---|---|
| 过程公开比最后一句结果更重要 | 每个正式实验留下 hypothesis、command、seed、配置和 receipt |
| 依赖关系是研究上下文的一部分 | 记录数据版本、模型版本、前置 run 与结果 lineage |
| 失败实验也是知识 | 失败不删除；报告 kill criterion 是否触发以及下一步改变什么 |
| 社区参与需要低门槛入口 | 先提供小规模、可复现、成本明确的 vertical slice |

### 4. 不可直接迁移的部分

- Marin 的 LLM、MoE、TPU 和大规模训练 recipe 不是 TabU 的架构证据。
- Marin 的公开模型 benchmark 不能替代 TabU 的表格任务协议。
- “开放”不能被写成“没有缺口”；数据许可、外部复用、硬件可得性和历史 receipt 仍需逐项审计。
- 社区激励或 Speedrun 机制属于组织设计，不是当前 TabU-lab 的默认工程范围。

## 本地研究动作

下一轮不要泛读整个 Marin。选择一个小 experiment，记录：

1. 它的假设在哪里声明；
2. 代码如何表达依赖和配置；
3. run 的输入、seed、硬件和输出在哪里；
4. 结果是否有独立报告或 receipt；
5. 失败或反例是否改变了后续计划。

这五项之后，才决定哪些字段进入 TabU-lab 的 experiment contract。

## 未决问题

- issue → PR → run → report 是否在一个具体 Marin experiment 上闭合？
- 哪些 provenance 字段是 Marin 的稳定约定，哪些只是某一条实验线的实现细节？
- 外部贡献者真正参与了哪些关键研究节点？需要数据而不是印象判断。
- TabU-lab 需要公开到什么粒度，才能在安全、成本和可复现之间取得平衡？
