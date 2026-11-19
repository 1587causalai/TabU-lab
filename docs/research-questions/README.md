# Research Questions

TabU 研究共享模型能否从表格观测学到各个 Unit 的规律，以及这些表征能否帮助预测新观测和新表。这里从问题进入公开研究记录。每项都写明已知依据和仍需检验的部分。[项目简介](../../README.md) · [模型版本](../design/README.md) · [公开报告](../reports/README.md)

## 表征、重构与读出

- **把 Unit 作为基本对象，能带来什么可检验的预测？** 公开理论框架用与同一 Unit 关联的观测来表达个体规律。一个关键的开放问题是，这样组织观测何时改善解释或留出预测，什么反例会促使我们修正设计。[Unit as a Primitive 预印本](https://arxiv.org/abs/2608.25118v1)
- **怎样保证待预测值在模型形成预测前不可见？** V8.2 的可选监督路径将跨行供源限制在有标签的 Support 行，Query 行仍可在行内利用已知特征。对应实现检查覆盖该配置的供源和梯度边界。[V8.2 验证](../reviews/v82-integration-20261012/README.md)
- **更准确地重构可见值，会让模型更擅长预测新观测吗？** V9 的受控对照得到更低的重构误差，对同一生成规律的新观测预测变化很小。什么目标能让模型学到可迁移的规律，仍需从新观测和新表检验。[V9 重构对照](https://research.wehub.us/tabu-lab/experiments/v9-reconstruction/)
- **直接读出值编码，能改善解码后的预测吗？** V8.4.2 比较了直接预测固定值编码的方案。所测任务上的收益和退步并存，下一步仍需找出哪些表征误差会影响解码后的预测。[V8.4.2 读出比较](https://research.wehub.us/tabu-lab/experiments/v842-value-code/)
- **按 Support 方差平衡回归损失，能帮助一个共享模型学多张表吗？** V8.4.0.2 的回归表现有改善，方差重加权相对对应对照的额外收益仍缺少证据。[V8.4.0.2 损失对照](https://research.wehub.us/tabu-lab/experiments/v8402-loss-balance/)
- **Unit–Cell 表征能否把非线性关系变成可线性读出的信息？** V11 计划用当前表的观测拟合共享规律，再从相似 Unit 的残差估计个体修正。要检验的是解码后的留出预测，以及局部修正能否带来额外收益。[V11 研究路线](../../README.md#v11-research-roadmap)

## 训练、适应与迁移

- **一个模型怎样学习多种生成规律，同时保留已学能力？** V8.2 的学习阶梯显示模型在所测的合成规律上能继续学习，任务顺序和回放会改变已有能力的保留。跨更多任务和生成机制的稳定性仍待检验。[V8.2 学习阶梯](https://research.wehub.us/tabu-lab/experiments/v82-learning-ladder/)
- **集中给困难表加训，会改善同表的留出预测吗？** V8.2 的定向训练对照改善了训练行拟合，留出 Query 的表现没有随之改善。这个结果约束了该配方下继续追加曝光的判断。[V8.2 定向训练](https://research.wehub.us/tabu-lab/experiments/v82-targeted-training/)
- **预训练、当前表的上下文和表内拟合各贡献什么？** V10.1 的等任务量对照记录了从起点到后续检查点的变化，起点已能用 Support 标签拟合读出。要拆开三种贡献，仍需在相同评测条件下逐项对照。[V10.1 同任务量比较](https://research.wehub.us/tabu-lab/experiments/research-update-20261029/)
- **表格基础模型的预测在什么意义下可解释为贝叶斯推理？** TabPFN 的原论文提供了从合成先验和上下文学习预测的出发点。开放问题是，给定先验、模型可见信息和实际输出，什么观察能区分贝叶斯解释与其他预测机制。[TabPFN 原论文](https://arxiv.org/abs/2207.01848)
- **在合成世界学到的规律，哪些能迁移到新世界和真实表？** 同一 V10.1 比较分别报告了新合成世界和真实表的表现，数值回归是明确的差距。数据先验、任务覆盖和训练配方各自的作用仍待厘清。[V10.1 对照与局限](https://research.wehub.us/tabu-lab/experiments/research-update-20261029/)

## 上下文规模、效率与评测

- **给定一个模型，推理时该用多少 Support？** 固定 CellRidge-1024 权重的本地 Lite 对照中，将 Support 上限从 2,048 提到 8,192，整体分数提高，推理开销也增加。按目标或 Query 选行能否胜过等量随机样本或全量，仍是开放问题。训练样本选择需要单独的对照。[Support 上限对照](https://github.com/1587causalai/kumo-small-cellridge-1024/blob/main/support-8192/README.md)
- **怎样在百万行表上兼顾训练、推理和预测效果？** V10.1 的测速找到了中间状态和执行开销的影响，给出了两项匹配任务上的提速结果。能力保持和百万行级的准确率、内存与耗时仍待测量。[训练瓶颈回顾](https://research.wehub.us/tabu-lab/blog/speed-bottleneck-breakthrough-20261029/)
- **一个改造版应怎样证明自己比父模型更有价值？** TabArena 维护者关闭了首次 CellRidge-1024 提交，理由是已公开的本地 Lite 表现低于 Kumo Small。公开报告披露了检查点选择时接触过的划分。后续仍需要在匹配的预算与信息边界下开展独立比较。[本地 Lite 报告](https://github.com/1587causalai/kumo-small-cellridge-1024/blob/main/LITE.md) · [维护者说明](https://github.com/autogluon/tabarena/pull/735#issuecomment-7015532014)

## 编号诊断

- [001 · 近零 Query 初始化与 Feature 分支失稳](001-near-zero-feature-bank-fit-instability.md)记录了 TabU-v2 的 float32 presence gate 缺陷及修复。固定表诊断中，warm-up 改善了部分拟合表现。Feature 分支的稳定读出和一般作用仍待验证。

独立诊断沿用 `NNN-short-question-slug.md`。记录应注明模型、任务和证据范围，并写清已复核的现象、仍待检验的解释以及能改变判断的下一项观察。
