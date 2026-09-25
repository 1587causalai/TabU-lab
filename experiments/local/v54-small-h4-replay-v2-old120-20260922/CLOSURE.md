# gongqian-mini Small-H4 replay-v2：用户主动提前收束

2026-09-25 用户确认已亲自停止本任务，认为此阶段拟合训练已足够。runner 写出 `interrupted` / `local_unissued`、`error=null`；这是有意停止的技术终态，不是失败，也不是完成全部预算。无活跃 trainer，禁止自动恢复、重启或追加剩余预算。

- 终态：710,158 actual / 1,321,272 上限；530,392 normal / 983,040 上限；179,766 累计 extra，其中父 v1 继承 2,400。本 v2 接续新增 684,238 actual。最后完整固定训练行 Query bank 在 actual 706,560，距停止点 3,598 actual；**没有**把该 bank 的指标冒充停止点精确值。
- 模型与身份：**Small-H4 变体，不是标准 Small**；backbone3 / 4 heads / Unit3，由 Nano 权重扩深初始化，再经后续训练。MPS FP32、`train-20260920`。冻结 source SHA-256 `0dfd1f3a6e350d5a5434f128ec25ce4277699f1a240f9891444b31374534c421`，data SHA-256 `f14ca644085f62c06a2c172f20773d6f4fe8dbdd6b1481478ff3f62d2e81c8dd`。组合编码、`supervised_row` 25%、Query-only，仅训练行。
- [本地 checkpoint](frozen-owner-stop-20260925/fae80fcfcb7f571df7b92a2675edfa5b1a85c82efa539fbd97d9453b076dff41.pt) SHA-256 `fae80fcfcb7f571df7b92a2675edfa5b1a85c82efa539fbd97d9453b076dff41`（26,505,507 bytes）与[远端 terminal 副本](frozen-owner-stop-20260925/terminal.json)一致；[archive 回执](frozen-owner-stop-20260925/archive.json)记录复制及哈希。原远端输出 `/Users/gongqian/tabu-v54-small-h4-replay-v2-old120-20260922-0dfd1f3/runs/old120.small-h4.replay-v2` 保留未覆盖。
- [终态只读检查](status-20260925T010041Z.json)核对 source/data/identity、有限日志、checkpoint 和无活跃训练进程；[W&B 云端读回](frozen-owner-stop-20260925/wandb-cloud-closure.json)为 `finished` / `closed_after_stop`，并明确 `training_budget_completed=false`。[run](https://wandb.ai/zj3712/restoration/runs/v54-small-h4-replay-v2-old120-gq-20260922)。
- 最后完整 bank 的 57 张 numeric R² 当前中位 0.99324，43 张 nominal accuracy 当前中位 0.98670，20 张 ordinal accuracy 当前中位 0.70112。逐表当前、v2 输出内最佳、可见支持参考和曝光见[收口指标](closure-metrics.json)，完整原始状态保留在终态只读检查。`scm_mixed_v1_045` nominal 0.0507 仍低于支持多数类 0.1215；`discoscm_087` ordinal 0.3012 低于 0.3536。不能宣称全部表拟合或未见表泛化。

跨三路 V5.4 的解释与后续复用边界见[收口报告](../../../docs/reports/v54-old120-fit-closure-2026-09-25.md)。这个 checkpoint 若作为新代权重起点，必须明确是 weights-only 还是严格续跑，并保留其父模型、优化器和曝光谱系。
