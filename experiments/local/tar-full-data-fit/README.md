> 后续验证优先使用 [Small 验证入口](../tar-small-validation/README.md)。本目录保留其原始 Standard 或尺寸对照协议，不作为默认后验验证入口。

# TAR 全量数据监督拟合

使用默认 54,071,520 参数模型，从随机初始化开始。固定外层 80/20 划分，全部原始行恰好属于 train 或 test；不再抽取小子集。

| 数据 | 总行数 | Train | Test | 每次训练可见标签 | 每次训练 mask 标签 |
|---|---:|---:|---:|---:|---:|
| Iris | 150 | 120 | 30 | 80 | 40 |
| Diabetes | 442 | 353 | 89 | 235 | 118 |

Iris 按类别分层，每类 test 10 行；Diabetes 随机划分。固定 split seed 20260906，所有初始化 seeds 共用同一外层划分。原始 values 不变；数据文件记录来源与哈希，preregistration 绑定完整文件 SHA-256。`tar_data.full_train_test_split` 可复算划分。

每次更新包含整个 train，按地址随机选择 ceil(train_rows/3) 个 target labels 遮住并计算 loss；重新生成 episode nominal codebook，同列同类别在该 episode 内共用向量。保留模型和优化器状态。训练 episode 不含 test 行。

测试使用整个 train 的全部标签作 context，整个 test 的标签都隐藏，在一个 episode 内一次 forward 输出全部预测。所有已知 covariates（含 test covariates）共同参与列计算，这是明确的 joint-test / transductive 输入配置。Test labels 只用于输出之后评分，不用于 checkpoint 选择；独立逐行预测 API 仍作为另一输入模式保留。

训练拟合评估使用预固定的 8 个训练 mask episodes；初始/最终评估重放同样的 mask/codebook，和随机训练流分开。测试初始/最终也重放相同的单个联合 episode。记录 loss、分类 accuracy/NLL 或回归 MSE/RMSE、context 常数基线。不可把训练随机 loss 的首末差值当成固定任务改善。

```bash
uv run --no-sync tabu-lab tar fit \
  --preregistration experiments/local/tar-full-data-fit/preregistration.yaml \
  --dataset iris --seed 1729 --device cuda:0 --output-root /tmp/tar-full-data-iris-new
```

CPU 管线检查使用 `--smoke --device cpu`：仅缩小模型与更新次数，保持全量数据、外层划分和整批 test。输出目录必须全新。数据覆盖检查在模型分配前执行，拒绝漏行、重复、重叠及截断数据；`data-coverage.json` 和每步行 ID 提供可核验证据。

默认单初始化 seed：1729；每个数据集最多 300 次更新或 600 秒训练，恒定学习率 1e-4。整个批次（含启动、评估、保存）上限 1800 秒，每个数据集墙钟上限 780 秒，最后 90 秒预留评估与保存。多 seed 复验需另开有明确总预算的实验。时间预算耗尽且未达拟合阈值记录 budget_limited，不等于收敛失败。仅作为六阶段中的真实数据基本预测诊断，不能证明预训练、frozen ICL 或微调提升。

W&B 需显式启用 observer，并配置在线认证；当前授权 DGX2 启动器使用 `tabu-lab` 项目、`tar-full-data-80-20` 组。仅同步标量，URL 单独记录为 observation，本地结果/checkpoint/哈希为证据。检查云端实际 history 后才报告监控在线。

历史 `full-context-fit` 使用小子集；其源码、配置和产物保留，不与本次全量数据指标混用。此次只改训练数据协议、检查与记录，模型数学及 batched backbone 未改。

## 整批 30 分钟入口

```bash
uv run --no-sync python experiments/local/tar-full-data-fit/run_batch.py \
  --output-root /tmp/tar-two-datasets-new --command tabu-lab
```

`--command` 后可传入已配置的 GPU/W&B 启动器命令前缀，例如 DGX2 源码快照内的 `tabu-lab`。脚本按顺序运行 Iris、Diabetes，各一个 seed。墙钟预算覆盖命令启动、训练、评估、保存和退出；每条独立记录结果。到最终截止时间仍未退出则中断该次运行并保留日志，不把中断视为完整拟合评估。旧六次运行配置及结果已留在归档快照。

本仓库保留协议与输入作为可复现配置。原实验的结果与 checkpoint 继续绑定原始归档；当前集成的来源字段改变 checkpoint source identity，新的运行会产生新身份。
