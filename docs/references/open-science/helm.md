# HELM：统一、透明、可复现的评测参考

> 研究卡状态：`方向卡`
>
> 最后核验：2026-08-26

## Canonical sources

- [Stanford CRFM HELM repository](https://github.com/stanford-crfm/helm)
- [HELM documentation](https://crfm-helm.readthedocs.io/en/latest/)
- [CRFM project page](https://crfm.stanford.edu/helm/)
- [HELM maintenance mode policy](https://crfm-helm.readthedocs.io/en/latest/maintenance_mode/)

## 它提供什么参考

官方 README 将 HELM 描述为用于 foundation models 的 open-source evaluation framework，重点是 holistic、reproducible 和 transparent evaluation。它把数据集 / benchmark、模型接口、多个指标、prompt-response inspection 和 leaderboard 组织进同一套框架。

截至本卡核验日，官方 README 已标注 HELM 于 2026-06-01 进入 maintenance mode。这说明它仍然是评测方法的高价值参考，但不能把当前维护状态误读成 TabU 可以直接复用一套持续开发中的实现。

## 对 TabU 的关系

### 可以借鉴

- 评测不只报告单一 accuracy，还要显式记录任务、效率、稳健性和适用边界。
- 模型接口、数据集、指标和结果要可机器读取，减少手工叙述漂移。
- 评测结果必须能回到固定协议和原始输出，而不是只有网页上的排名。

### 不能直接搬用

- HELM 面向 foundation models 的评测框架，不等于表格模型的 task protocol。
- “holistic” 不是无限增加指标；TabU 仍需要由研究问题决定最小充分评测。
- leaderboard 排名不替代 mechanism evidence 或 causal claim。

## 待定义的 TabU 版本

- 表格任务的最小 scenario 集合；
- classification / regression / missingness / Unit-conditioned query 的统一输入输出契约；
- accuracy 之外的 calibration、compute、data efficiency 和 shift 指标；
- 每个结果如何链接到 immutable run receipt。
