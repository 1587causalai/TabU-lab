# old120 replay-v2 阶段检查点冻结

2026-09-23 为下一代设计和复核，逐台选取一个**已有不可变 generation**，从训练主机只读复制到本目录 `checkpoints/<host>/<sha256>.pt`。三份文件均重新计算 SHA256，并用冻结 runner 的 checkpoint loader 验证身份、模型配置、源码/数据摘要、checkpoint 更新、模型/AdamW/RNG 载入和有限状态。精确路径、远端来源、时间戳、实际步与对应固定 Query 位置见 [freeze.json](freeze.json)。

这些权重比同一时间戳 status 中的实时 update 稍早；阶段报告只能使用对应[逐表状态](../../../docs/reports/v54-old120-fit-handoff-2026-09-23.md)的 fixed 时点。三个训练仍按原配方继续，复制本身没有停训、接续或增加曝光。以后从其中一个权重启动新架构时，要写明 weights-only 或 optimizer 迁移，不能把父更新计入新训练预算。
