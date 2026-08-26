# LimiX：统一结构化数据任务的参考线

> 研究卡状态：`第一轮 source read`
>
> 最后核验：2026-08-26

## Canonical sources

- [limix-ldm-ai/LimiX repository](https://github.com/limix-ldm-ai/LimiX)
- [LimiX project page](https://www.limix.ai/)
- [LimiX technical report](https://arxiv.org/abs/2509.03505)
- [LimiX-2M paper](https://arxiv.org/abs/2606.04485)
- [LimiX model organization](https://huggingface.co/stableai-org)

## 第一轮理解

LimiX 官方仓库把自己定位为 large structured-data foundation model，并把 classification、regression、missing-value imputation 和 tabular generation 等任务放进一个统一的结构化数据建模叙事中。README 进一步描述了 feature 与 target 的 embedding，以及沿 sample 和 feature 维度应用 attention 的架构方向。

这里要严格区分两件事：

1. 这些是项目 README 的 source claims；
2. 它们是否在论文、代码、权重和可重跑 benchmark 中得到充分支持，需要逐项审计。

## 需要从论文和代码确认

- “joint distribution modeling” 在实现中对应的具体随机对象是什么？
- missing value 是训练目标、输入 mask、联合分布采样，还是若干 inference mode 的组合？
- classification、regression、imputation 是否共享同一个 backbone、loss 和任务条件？
- 论文所称的 causal inference 是哪一种 estimand、数据假设和评测协议？
- 10 个结构化数据集的 split、预处理、超参预算、baseline 版本和 seed 是否可重建？
- LimiX-16M 与 LimiX-2M 的能力差异来自规模、retrieval、训练数据还是 inference protocol？

## 对 TabU 的关系

### 可以借鉴

- 把缺失值、预测目标和任务条件作为统一学习问题来观察。
- 研究 sample 维度和 feature 维度交互对表格表示的影响。
- 在比较模型时，单独记录“支持哪些任务”与“在什么数据协议下有效”。

### 不能直接搬用

- “统一任务”不是 Unit 语义，也不是 shared response law 的证明。
- 代码公开和 checkpoint 可下载不等于训练数据 provenance 已经完整。
- 项目 README 的 SOTA / general intelligence 叙事不能直接成为 TabU 的 capability claim。

### 需要本地实验回答

- Unit-conditioned response 是否可以与普通 task-conditioned tabular model 分离比较？
- 缺失值的显式 mask、未知状态和 Unit-level semantics 是否需要不同的 carrier？
- 在固定 compute、split 和 seed 后，统一任务头是否真的优于任务专用 baseline？

## 当前结论

LimiX 是很有价值的**问题空间参照**，尤其适合帮助我们拆分“统一结构化任务”与“Unit 语义”之间的差异；它目前还不是 TabU 架构或性能的证据。
