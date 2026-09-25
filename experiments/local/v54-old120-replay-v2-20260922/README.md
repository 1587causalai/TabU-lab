# Old120 高损失表加训 v2

**当前状态（2026-09-25）：三路 V5.4 old120 replay-v2 均已主动提前停止并封存，不再有此阶段在训任务。** dgx2 标准 Small 与 gongqian-mini Small-H4 由用户亲自停止；dustinstudio Nano 此前已封存。完整终态、固定训练行 Query 拟合与后续复用边界见 [收口报告](../../../docs/reports/v54-old120-fit-closure-2026-09-25.md)。下文是当时的策略、切换和运行证据，不表示当前仍在训练。

2026-09-22 用户调整：保留正常 120 表均衡覆盖；P99 档额外 3 次、P95 档额外 2 次、P80 档额外 1 次，三个档位叠加。最高损失的小组每轮额外 6 次。

120 表的配额使用比例向上取整：P99 对应 top2，P95 对应 top6，P80 对应 top24。用该轮 120 个正常原始 loss 降序、表 ID 升序固定排名，然后执行三个完整 top2 pass、两个完整 top6 pass、一个 top24 pass。共额外 42 次，正常与加训合计 162 次。第一组 2 表各额外 6 次，随后 4 表各额外 3 次，再后 18 表各额外 1 次，其余 96 表仅保留正常更新。

策略名 `normal120_p99x3_p95x2_p80x1_v2`。旧 v1 配方与既有日志保持原定义。新配方只适用于切换后完整轮，不追补过去训练。

## 接续约束

三台当前 v1 任务在新源码和工具准备完毕后安全保存。若停止点处于未结束的正常轮或加训队列，以原冻结源码在新的 bridge 输出补完当前完整轮；不回滚，不跳过已选加训队列。补齐最多 149 个实际步，属于原 v1 配方已有预算。

设完整边界时累计正常更新为 $S$、累计旧加训为 $E$，则已有实际更新为 $U=S+E$。保持原正常累计上限 $N$，新增 v2 加训预算为 $(N-S)/120\times42$；新的实际累计上限为 $N+E+(N-S)/120\times42$。旧加训计数和逐表曝光保留，不能从零抹去或二次加到预算里。

`migrate_replay_v2.py` 只接受已安全停止、完整正常加训轮边界的 v1 父 checkpoint，保留模型、AdamW、RNG、正常游标、历史 extra、曝光与祖先链。改变仅为明确的策略、预算和相关冻结源码；迁移不是不改变配方的严格续跑。

正常 episode namespace 保持；v2 加训使用独立 `/loss_replay_v2` namespace，表内 extra episode index 继续累计。新策略记录 `extra_top2`、`extra_top6`、`extra_top24`。训练正常预算不变：dgx2 标准 Small 为 983040；dustinstudio Nano 为 1105920；gongqian-mini Small-H4 变体为 983040。

新阶段目录分别为 `v54-small-replay-v2-old120-20260922`、`v54-nano-replay-v2-old120-20260922`、`v54-small-h4-replay-v2-old120-20260922`。各目录的决策、迁移、启动和首轮回执才是完成切换的证据；本说明本身不表示训练已启动。

## 监控

原始每步 loss 留存。`train_cycle` 均值、中位数、P05、P95 继续只统计正常的 120 个 loss，新策略增加 P99 分位数供查看。加训 loss、各档实际次数及每表额外曝光单独记录，历史继承量与本阶段新增量分开。

继续固定训练行 Query-only 拟合；组合编码、模型结构、学习率、环境和数据不变。新阶段的固定 Query 当前与最佳值仅来自新输出的完整评估，不继承父最佳。未使用 reserved/final_test，未进入泛化。

## 已完成切换的实际边界

| 主机 / 模型 | 原正常步 S | 历史加训 E | 已有实际步 U | 剩余正常 | 本阶段新加训 | 实际累计上限 |
|---|---:|---:|---:|---:|---:|---:|
| dgx2 / 标准 Small | 122,520 | 1,110 | 123,630 | 860,520 | 301,182 | 1,285,332 |
| dustinstudio / Nano | 202,560 | 2,130 | 204,690 | 903,360 | 316,176 | 1,424,226 |
| gongqian-mini / Small-H4 变体 | 23,520 | 2,400 | 25,920 | 959,520 | 335,832 | 1,321,272 |

三台均已完成普通 CUDA/MPS 预检、保留状态迁移与唯一启动。实际初始 checkpoint 的模型/AdamW/RNG 与历史逐表加训计数均已读回验证。首个真实 162 步的正常覆盖、六个 pass、继承 episode index 与 6/3/1 次叠加均通过，日志数值有限。209 项 curriculum 回归、15 项迁移检查、37 项监控检查通过；详见 `validation.json`。三台训练均使用冻结源码 `0dfd1f3a6e350d5a5434f128ec25ce4277699f1a240f9891444b31374534c421`。

2026-09-22 UTC 04:44–04:47 独立只读验收确认三项均 running，source/data/identity 匹配、实际更新增长。此时新输出均尚无完整固定 Query 结果，以上仅证明切换、执行与监控成立，不表示拟合已改善。

各机的启动、迁移、首轮和实时证据：

- [dgx2 标准 Small](../v54-small-replay-v2-old120-20260922/README.md)
- [dustinstudio Nano](../v54-nano-replay-v2-old120-20260922/README.md)
- [gongqian-mini Small-H4](../v54-small-h4-replay-v2-old120-20260922/README.md)

H4 的新 panel 最初误复制了只识别 v1 的旧监控入口，产生 `extra_top2`/历史 extra 计数误报；现已替换为共享 v2 监控薄入口并重新读回 running。训练日志与训练进程未受影响，早期误报快照保留作诊断。

既有 `tabu-nano` 每两小时回查已指向三个 v2 目录，仅读取，不自动启动或追加预算；旧 v1 与 bridge 不再恢复。

三个 W&B 新阶段均已完成真实云端读回，正常轮均值/中位数/P05/P95/P99、历史 extra 与新增 extra、各档加训记录和身份元数据全部匹配固定快照，零缺失、零差异：

| 当前 W&B 阶段 | 已验证正常轮 | 已验证加训点 | 原始 loss 采样点 |
|---|---:|---:|---:|
| [dgx2 Small v2](https://wandb.ai/zj3712/restoration/runs/v54-small-replay-v2-old120-dgx2-20260922) | 6 | 252 | 10 |
| [dustinstudio Nano v2](https://wandb.ai/zj3712/restoration/runs/v54-nano-replay-v2-old120-dustin-20260922) | 8 | 336 | 13 |
| [gongqian-mini Small-H4 v2](https://wandb.ai/zj3712/restoration/runs/v54-small-h4-replay-v2-old120-gq-20260922) | 4 | 168 | 7 |

以上为验收所用固定快照，不是训练当前总进度。云端回执位于各 panel 的 `wandb-cloud-readback.json`，总验收见本目录 `validation.json`。
