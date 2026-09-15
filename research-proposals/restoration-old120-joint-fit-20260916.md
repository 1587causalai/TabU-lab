# old120 Small-128 joint-fit 首段记录（2026-09-16）

## 状态与范围

首段已在 DGX2 `spark-b5b3` 使用提交 `def4954` 的源码完成：同一个
FP64 Small-128 恢复模型遍历 old120 全部 120 表，每表每轮一次更新，共
2 轮、240 次更新。每表固定 8 个评估 masks；每列 Query 68 个 cell；每个
离散列保留每个训练集类别的一个可见样本。每表的 204 个训练行参与目标，
52 个 reserved 行没有进入模型或评分。

远端结果目录：
`/home/cms/tabu-restoration/runs/restoration-old120-small128-def4954/`

关键输出为 `terminal.json`、`initial-metrics.json`、
`metrics-round-0001.json`、`metrics-round-0002.json`、`updates.jsonl`、
`checkpoint-round-0001.pt`、`checkpoint-round-0002.pt` 和
`checkpoint-progress.pt`。terminal receipt 的 outcome 为 `completed`，
update 为 240，elapsed 为 1927.14 s（约 32.12 min），peak allocated 为
11,631,290,880 bytes。`updates.jsonl` 每轮 120 行，单步耗时均值 0.783 s，
最大 2.081 s。

## 固定-mask 汇总

| point | retained encoding MSE | Query encoding MSE | Query numeric MSE (original units) | Query discrete accuracy |
|---|---:|---:|---:|---:|
| initial | 0.0146223 | 29.0779 | 226.984 | 0.258783 |
| round 1 | 0.00456072 | 11.5671 | 111.254 | 0.278651 |
| round 2 | 0.00111226 | 9.66216 | 104.498 | 0.296756 |

相对 initial，round 2 的 retained encoding MSE 下降 92.39%，Query encoding
MSE 下降 66.77%，Query numeric MSE 下降 53.96%；Query 离散准确率提高
3.80 个百分点。这个变化只描述冻结 old120 训练行上的拟合诊断，不代表
reserved-row 或 unseen-table 泛化。

每个固定评估点的覆盖计数一致：Query `1,072,768`、总目标
`3,218,304`、Query fraction `1/3`；保护的离散支持 cell `87,168`，其中
singleton 类别 `23,816`。汇总同时写入每表、四个 source family
（`discoscm`、`scm_mixed_v1`、`scm_numeric_v0`、`sklearn_synthetic`）和
列类型分组。

## 启动前验收

- 当前源码的 CUDA FP64 CPU↔CUDA forward/backward 对照通过。
- 两步 optimizer continuation（model、optimizer、CPU/CUDA RNG）通过，最大
  表 `204×32` 的 forward/backward/update 有限性通过；该有限性检查峰值显存
  为 `11,454,275,072` bytes。
- 所有 808 个仓库测试通过，9 个因本地缺少 `sklearn`/`xgboost` 跳过；新
  joint-fit 专项测试为 4 passed。

CUDA 对照 receipt 位于 `/home/cms/tabu-restoration/cuda-check-def4954-r2.json`，
最大表 receipt 位于 `/home/cms/tabu-restoration/max-scale-def4954.json`。

这些检查和本次首段结果仍是本地未发行的实现／拟合证据；没有据此声明
benchmark readiness 或公开能力。
