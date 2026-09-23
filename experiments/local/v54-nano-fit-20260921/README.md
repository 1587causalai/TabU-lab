# V5.4 Nano 单表拟合主线

**20:04 主线更新：gongqian-mini 的 r0 已完成，r1 已于 20:03 启动，learning rate=1e-4，MPS/FP32；20:04 已读回 246 步。其 r0 结果和检查点保留。**

每表执行 **r0/r1/r2 三次独立重复，每次 16,384 步，总预算 49,152 步**。统一 AdamW learning rate **1e-4**，每次从头初始化；预算定义不等于同一模型连续训练 49,152 步。当前只运行主线任务。

当前结果：[CURRENT.md](CURRENT.md)。主机调度：[ALLOCATION.md](ALLOCATION.md)。状态：[followup-decisions.json](followup-decisions.json)。每 30 分钟的线程回查保持启用，automation id 为 `tabu-nano`。

## 配置

Nano：backbone layers=2、width=128、heads=4、FF width=256、slots=256、Unit layers=0、subtokens=1。组合编码 `constant_weight_composition_v1`，zscore，shared_ll。Episode 为 supervised_row、Query fraction=0.25，state_weights=[0,1,0,0]。

每个 run 保留 204 行 train、52 行 reserved，当前只使用 train_fit，固定 8-mask 每 1024 步评估、每 512 步保存 checkpoint。reserved 不进入当前训练或评分。CUDA 使用 FP64，MPS 使用 FP32 与 device-local readout；重用 train-20260920 环境。拟合效果与速度优先，数值更新微差只用于诊断。

| 主机 | 表 | 目标 | device/dtype | chunk |
| --- | --- | --- | --- | ---: |
| dgx2 | discoscm_095 | 12 类分类 | CUDA/FP64 | 128 |
| deepthought | scm_mixed_v1_001 | 数值 | CUDA/FP64 | 32 |
| dustinstudio | discoscm_076 | 数值 | MPS/FP32 | 128 |
| gongqian-mini | scm_mixed_v1_017 | 二分类 | MPS/FP32 | 128 |
| zichao-mini | scm_mixed_v1_040 | 分类 | MPS/FP32 | 128 |

## 身份与回执

Source SHA-256：`8db026b30a6a2b071f0c0bfc8745ed686dcf4f0f68db886d61f69393d106ba87`。冻结根：各主机 `~/tabu-v54-nano-fit-20260921-8db026b`。r0 manifest 在 `manifests/`，r1/r2 在 `replicates/`。原始 manifest、checkpoint、数据及主线历史读数保留。

- [首轮准备回执](preparation-receipt.json)：启动前的历史快照。
- [r1/r2 准备回执](replicates/preparation.json)：10 份独立重复配置。
- [CUDA/dustinstudio 首轮启动](launch-dgx2-dustinstudio.json)。
- [两台 mini 首轮启动](launch-minis.json)。
- [19:37 主线曲线与首轮终态](progress-20260921-1937.json)。
- [19:47 主线状态核验](status-20260921-1947.json)。
- [r0 固定 Query 常数参考](naive-query-reference.json)：只用可见支持预测均值或多数类。

这些结果只建立单表训练行上的遮挡拟合证据，不构成多表或泛化结论。

- [gongqian-mini 主线 r1 启动回执](launch-gongqian-mini-r1.json)。
