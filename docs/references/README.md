# TabU-lab 参考工作目录

> 这是 TabU-lab 的研究资料目录，不是简单的 bibliography。每个参考工作都要回答：它到底在解决什么问题、机制是什么、证据在哪里、哪些部分可以借鉴、哪些部分不能直接搬到 TabU。

更新日期：2026-08-26

## 先读哪一层

| 层 | 参考工作 | 我们借鉴的对象 | 当前状态 |
|---|---|---|---|
| 研究方法 | [Marin](marin/README.md) | open development、实验过程、receipt、社区协作 | 深入卡已建立；原始综合稿仍保留为历史叙事 |
| 表格模型 | [TabPFN](tabular-foundation-models/tabpfn.md) | prior-data fitted / in-context tabular prediction 路线 | 第一版卡，论文级审计待做 |
| 表格模型 | [LimiX](tabular-foundation-models/limix.md) | 统一结构化数据任务、缺失值与任务泛化问题 | 第一版卡，论文级审计待做 |
| 评测方法 | [HELM](open-science/helm.md) | 可重复、透明、多维度的评测组织方式 | 方向卡，TabU 适配待定义 |
| 数据方法 | [DataComp](open-science/datacomp.md) | 固定模型协议下考察数据设计 | 方向卡；不是表格模型架构参考 |
| 开放产物 | [LLM360](open-science/llm360.md) | 完成时的代码、数据、权重和过程打包方式 | 方向卡；当前未做完整 source audit |

## 目录结构

```text
docs/references/
├── README.md                         # 总索引、阅读路线、证据边界
├── _template.md                      # 新参考工作的研究卡模板
├── marin/
│   └── README.md                     # 开放研究过程与 Marin source map
├── tabular-foundation-models/
│   ├── README.md                     # 表格模型对比维度
│   ├── tabpfn.md
│   └── limix.md
└── open-science/
    ├── README.md                     # 过程、评测、数据、产物四种开放性
    ├── helm.md
    ├── datacomp.md
    └── llm360.md
```

## 推荐阅读路线

1. 先读 [Marin source map](marin/README.md)，理解“开放实验室”究竟是流程、软件、证据还是社区。
2. 再读 [TabPFN](tabular-foundation-models/tabpfn.md) 和 [LimiX](tabular-foundation-models/limix.md)，把“表格 foundation model”拆成 prior、任务分布、表示、缺失值和评测协议几个问题。
3. 最后读 [HELM](open-science/helm.md)、[DataComp](open-science/datacomp.md) 和 [LLM360](open-science/llm360.md)，区分评测开放、数据开放、完成时产物开放与进行时过程开放。
4. 回到 `experiments/`，把参考工作转成 TabU-lab 自己的 hypothesis、kill criterion、command、seed 和 receipt；参考资料本身不能替代实验。

## 证据边界

### 四种内容要分开

| 标签 | 含义 | 可以支持什么 |
|---|---|---|
| `source claim` | 项目官网、仓库 README、论文或官方报告明确写出的内容 | “该项目声称 / 提供 / 采用” |
| `observed` | 我们实际打开源码、配置、issue、报告或运行得到的观察 | 可复述的局部事实，带日期与路径 |
| `interpretation` | 我们对机制或研究策略的解释 | 研究假设，不是项目官方结论 |
| `TabU proposal` | 我们准备借鉴的做法 | 只有在本项目 receipt 验证后才升级为结果 |

尤其要避免以下跳跃：

- Marin 的开放流程不等于 TabU 已经有可复现训练结果。
- TabPFN 或 LimiX 的 benchmark 表现不等于 TabU 的模型能力证据。
- HELM 的评测框架不等于 TabU 已经拥有完整评测协议。
- 一个公开 checkpoint、仓库或网页不等于训练过程、数据 provenance 和质量结论都已公开。

## Source precedence

对时效性内容，按以下顺序使用来源：

1. 当前 canonical repository、论文正文、官方报告和官方实验 receipt；
2. 项目官网和官方文档；
3. 官方 blog、issue、PR 和 release note；
4. 我们的复盘稿、Discord 讨论和历史快照。

每张研究卡都记录核验日期。旧内容可以保留作研究史，但不能覆盖当前 source。

## 与 TabU-lab 的关系

当前只形成三条明确的借鉴线：

- **Marin → 研究过程**：预注册、可追溯实验、失败也留档、公开 receipt。
- **TabPFN / LimiX → 模型问题空间**：表格任务的 prior、上下文、缺失值、统一任务和泛化边界。
- **HELM / DataComp / LLM360 → 证据与公开方式**：如何固定协议、组织评测、开放数据或发布完整产物。

这三条线仍然是不同层级，不能压成一个“我们参考了某某，所以 TabU 已经成立”的故事。

## 维护规则

- 新增参考工作时，先复制 [`_template.md`](_template.md)，不要直接把长篇观点堆进总 README。
- 一张卡只负责一个工作或一个紧密的工作族；跨项目比较放在上层目录。
- 新的性能数字必须带论文 / 官方报告 / receipt 链接、数据协议、seed 和核验日期。
- 如果只有项目 README 被读过，明确写 `第一轮 source read`，不要写成“已理解架构”。
- 参考工作进入 TabU-lab 之前，必须在 `experiments/` 或 `docs/reports/` 中找到对应的本地验证入口。
