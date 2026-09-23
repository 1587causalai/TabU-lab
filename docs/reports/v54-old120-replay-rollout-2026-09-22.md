# Old120：三个现有任务接入高损失表加训

> 历史 v1 阶段记录。用户随后调整为 P99 × 3、P95 × 2、P80 × 1 叠加，共 42 次加训；最新策略与切换入口见 [v2 说明](../../experiments/local/v54-old120-replay-v2-20260922/README.md)。下文的 30 次加训和预算只描述当时的 v1 阶段。

2026-09-22，用户确认在保留 120 表均衡覆盖后给高损失表加训，并将 gongqian-mini 已采用的策略扩展至 dgx2 与 dustinstudio。这里记录策略切换和预算；实时训练与逐表拟合结果以各 panel 的带时间戳读回为准。

每轮先执行 120 次正常更新，每张表一次。依据这 120 个原始 loss 固定排名，最高 6 张表各额外更新一次，再给最高 24 张表各额外更新一次；前 6 张表因此共额外更新两次。每轮共 150 次实际更新，相对于正常更新新增 25% 计算量。并列按表 ID 排序，保证配额；加训 loss 不参与本轮排名。

## 已固定的接续预算

| 主机 / 模型 | 切换正常步 S | 每表已有正常更新 | 正常累计上限 N | 切换后正常更新 | 新增加训 | 累计实际步上限 |
|---|---:|---:|---:|---:|---:|---:|
| dgx2 / 标准 Small，3 层、8 heads、Unit3 | 118,080 | 984 | 983,040 | 864,960 | 216,240 | 1,199,280 |
| dustinstudio / Nano，2 层、4 heads、Unit0 | 194,040 | 1,617 | 1,105,920 | 911,880 | 227,970 | 1,333,890 |
| gongqian-mini / Small-H4 变体，3 层、4 heads、Unit3 | 13,920 | 116 | 983,040 | 969,120 | 242,280 | 1,225,320 |

加训预算为 `(N-S)/120 × 30`，累计实际步上限为 `N + 加训预算`。原正常预算保持不变，仅对切换后的完整轮加训，不追补旧轮。Nano 的正常累计上限包含最初每表 1,024 次和随后授权的每表新增 8,192 次；joint5 的五张旧表父历史另记。Small-H4 的 Nano 初始化历史也另记，不能算作该模型此次新增更新。

dgx2 原任务在 118,030 步安全保存，以原源码继续 50 步到完整轮边界；dustinstudio 在 194,029 步保存，以原源码继续 11 步到边界。gongqian-mini 此前在 13,814 步保存，以原源码继续 106 步到边界。所有接续保留模型、AdamW、RNG、正常游标及曝光；迁移明确标记 `strategy_change_preserve_optimizer_rng`。旧输出与补齐边界的输出仅保留为证据，不恢复。

三个任务共用冻结调度源码 `eae13514fda04c067cb7f111a56b76e66c0a3efa1dc110cbb2dc4ecf44664306`；模型架构不变。继续使用 `train-20260920`：dgx2 CUDA/FP64，两台 Mac MPS/FP32。组合编码、`supervised_row` Query-only、固定训练行评估均保持。

## 当前入口与监控含义

| 主机 | 事实目录 | W&B 当前阶段 |
|---|---|---|
| dgx2 | [Small replay](../../experiments/local/v54-small-replay-old120-20260922/README.md) | [Small replay](https://wandb.ai/zj3712/restoration/runs/v54-small-replay-old120-dgx2-20260922) |
| dustinstudio | [Nano replay](../../experiments/local/v54-nano-replay-old120-20260922/README.md) | [Nano replay](https://wandb.ai/zj3712/restoration/runs/v54-nano-replay-old120-dustin-20260922) |
| gongqian-mini | [Small-H4 replay](../../experiments/local/v54-small-h4-replay-old120-20260922/README.md) | [Small-H4 replay](https://wandb.ai/zj3712/restoration/runs/v54-small-h4-replay-old120-gq-20260922) |

每步原始 loss 完整保留本地，W&B `train/loss` 按 100 个实际步抽样；每个加训 loss 单独进入 `replay/loss`。`train_cycle` 的均值、中位数、P05、P95 只计算正常的 120 个 loss，横轴为此阶段新增正常步，另附累计正常与实际步数。加训不会混入正常轮统计。

每 15,360 个实际步继续做固定 Query 评估，分别记录每表正常、加训及实际曝光。当前值与最佳值仅来自当前输出已有的完整固定评估；尚无新评估时明确留空，不把父任务最佳值带入。表均值/多数类参考保留，不能用整体平均替代逐表拟合判断。

本次验证目标是接续状态正确、120+6+24 调度实际执行、日志增长且数值有限，以及 W&B 云端收到实际历史。策略切换成功不表示拟合已改善。三个任务架构、初始化、历史、设备精度不同，不能视为公平的尺寸或策略对照，也未进入泛化评估。

2026-09-22 12:09（Asia/Shanghai）独立读回：dgx2 实际 118,967 步，其中正常 118,800、加训 167；dustinstudio 实际 197,255 步，其中正常 196,625、加训 630。两项均正在运行，身份、冻结源码、数据身份匹配，已有日志数值有限；当前新输出尚无完整固定 Query 评估。两项的初始状态和首轮 120+6+24 均已验证。dgx2 另经独立代理核对迁移、预算和执行回执。

只读监控已修正 CUDA 容器启动器与实际 Python 进程重复计数的问题，保留原始进程证据，同时分别记录 launcher 和 trainer；6 项相关检查通过。该修改不涉及远端训练代码或训练进程。

W&B 真实云端历史也已验收：Small 固定快照的 6 个正常轮、172 个加训点及 8 个原始采样点全部匹配；Nano 的 10 个正常轮、300 个加训点及 16 个原始采样点全部匹配。均值与分位数使用独立事件；身份、heads、device/dtype、起始步与策略元数据相符。两项云端读回均无缺失或数值差异，证据见各 panel 的 `wandb-cloud-readback.json` 和 `wandb-implementation-readback.json`。Small 快照截在第 6 轮加训尚未结束时，172 个点是该快照已产生的加训点数。

既有 `tabu-nano` 自动回查已改为这三个新目录，每两小时只读监测；不自动启动、重启、追加预算。三个当前任务均终结后汇总并暂停回查。
