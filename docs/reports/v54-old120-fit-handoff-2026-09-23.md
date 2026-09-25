# V5.4 old120 拟合阶段读回与下一代设计输入

**后续状态（2026-09-25）：** 本文是 2026-09-23 阶段快照。三路 V5.4 old120 replay-v2 后来均已主动提前收束并封存；当前终态、checkpoint 与固定 Query 证据见[收口报告](v54-old120-fit-closure-2026-09-25.md)。

2026-09-23 09:54–09:57（Asia/Shanghai）只读快照。三路 replay-v2 当时均在训练，**没有终态**；数字只表示训练行固定 Query 拟合，不涉及 reserved / final_test 或未见表泛化。完整的 120 表逐表曲线、来源身份、曝光和参考线保存在 [dgx2 Small 状态](../../experiments/local/v54-small-replay-v2-old120-20260922/status-20260923T015407Z.json)、[dustinstudio Nano 状态](../../experiments/local/v54-nano-replay-v2-old120-20260922/status-20260923T015405Z.json)、[gongqian-mini Small-H4 状态](../../experiments/local/v54-small-h4-replay-v2-old120-20260922/status-20260923T015406Z.json)。本报告是有时间戳的阶段证据，不取代后续终态。

上述三份完整状态 JSON 与检查点保留在原工作区，不随 Git 分发；Git 中的本报告和冻结索引只保存摘要、路径及 SHA，不能替代原始逐表回执或权重。

| 任务 | 实际 / 正常更新 | 最近完整固定 Query | 57 张数值表 R² 中位数 | 43 张 nominal 准确率中位数 | 20 张 ordinal 准确率中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dgx2 标准 Small，3 层／8 heads／Unit3 | 363,975 / 300,579 | 353,280 | 0.799 | 0.686 | 0.460 |
| dustinstudio Nano，2 层／4 heads／Unit0 | 695,692 / 566,280 | 691,200 | 0.989 | 0.931 | 0.555 |
| gongqian-mini Small-H4 变体，3 层／4 heads／Unit3 | 239,864 / 182,024 | 230,400 | 0.983 | 0.966 | 0.600 |

上述为各自**当前**固定评估的逐表中位数，模型之间不构成公平排名：架构、父权重、训练历史、当前 normal 曝光和 CUDA FP64／MPS FP32 都不同。每表 `best` 只在自己的 v2 输出内取最佳；它与当前值不可混用。三路已读回真实命令、身份匹配、连续有限更新、不可变检查点及无终态。旧均衡／v1 输出保留历史原意，不能用来代替此处 v2 曲线。

相对每张表的**可见支持常数参考**：三路数值表均为 57/57 严格高于支持均值 R²；nominal 分别有 30/43、41/43、41/43 严格高于支持多数类准确率，另各有 1 张持平；ordinal 为 11/20、13/20、15/20 严格高于参考。这个参考与训练行固定 Query 同属拟合诊断，不是 MLP／XGBoost 的配对对照。组中位数不能掩盖尾部：Nano 的 `scm_mixed_v1_045` nominal 为 0.047，支持多数类为 0.122；`discoscm_087` ordinal 为 0.282，对应参考 0.354。需继续逐表看 current、best、曝光和参考，不能宣布 120 表全部拟合。

## 当前加训策略暴露的资源分配偏斜

当前每轮先均衡做 120 个 normal 更新，随后按这 120 个**原始训练 loss** 跨表排序，top2 做三轮、top6 做两轮、top24 做一轮，总计 42 extra。实现与计数已验收，但从本快照的逐表 `extra_new` 汇总，三路新增 extra **全部给了数值目标表**：

| 任务 | 新增 extra | 得到 extra 的数值表 | 得到 extra 的 nominal / ordinal 表 |
| --- | ---: | ---: | ---: |
| dgx2 Small | 62,286 | 52 / 57 | 0 / 63 |
| dustinstudio Nano | 127,282 | 50 / 57 | 0 / 63 |
| gongqian-mini Small-H4 | 55,440 | 50 / 57 | 0 / 63 |

这个事实说明当前 raw loss 排名让不同目标类型的加训机会严重失衡；数值误差和离散损失的量纲、分布不能直接解释为同一种“还需要学多少”。一些低于各自多数类参考的离散表也没有获得 extra。均衡的 120 次正常覆盖仍然存在，不能说离散表完全没训练；也不能凭这份观察断言某个新排名法一定更好，或将现有拟合改善归因于 extra。

下一代方案应把**按类型分层配额**、**相对各表支持参考或历史改善幅度归一的难度排名**与当前 raw loss 排名作为竞争选项。比较时固定同一数据、Query bank、初始化和总 actual 预算，同时列出 normal 与 extra 曝光，重点看 nominal／ordinal 尾部及数值表是否退化。这个选择尚未写成 V5.4 默认或下一代数学定理。

## 保存与继续使用的边界

三份可恢复的模型／AdamW／RNG 检查点已按原内容地址另存于 [冻结回执](../../experiments/local/v54-old120-freeze-20260923/freeze.json) 所列路径；SHA、训练身份和 checkpoint update 均完成载入核对。它们分别位于实际更新 360,960、691,200、238,080，早于本快照实时更新几千步；不能将快照的当前值硬贴到这些权重上。各自冻结 source、data、manifest 和真实运行目录仍在对应 panel；任何新代初始化必须选定具体 checkpoint 并重新做同一后端的起点固定 Query。

Nano→Small-H4 的[权重扩深](../../experiments/local/v54-small-warm-old120-20260922/README.md)采用新增残差出口零初始化来保留起点；[Token dynamic 复制研究](../research/v54-model-growth/README.md)讨论将训练好的两层动态计算重复串联，它允许起点变化。这是两条不同增长候选。jarvis 的四份复制实验在本报告三路快照时仍处于准备阶段，随后于 UTC 02:16 独立启动；UTC 02:30 已有 actual 717 / normal 549 / extra 168。其正式 step0 120 表固定 Query 与同后端诊断一致，复制后起点明显退化，训练恢复仍未知。正式初始化 checkpoint 已另存，独立 [W&B run](https://wandb.ai/zj3712/restoration/runs/v54-nano-dynamics4x-old120-jarvis-20260922)的 step0 与前三个正常循环经 API 读回、与本地 outbox 对账；详见[独立 panel](../../experiments/local/v54-nano-dynamics4x-old120-jarvis-20260922/README.md)。

监控沿用[训练手册](../guides/v54-training-and-monitoring.md)：单步原始 loss 本地保留；完整 120 张表 normal loss 的 mean／median／P05／P95／P99 单独画图；extra 单列；W&B 收到点与本地 outbox 对账。用户的 [restoration5.4 工作区](https://wandb.ai/zj3712/restoration?nw=brg2pegawui)精确显示原三路，此设置不随新实验自动改变。
