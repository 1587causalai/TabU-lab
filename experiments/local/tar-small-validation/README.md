# 当前 TAR 验证入口：Small first

后续实现改动、数值修复、组件组合与基本拟合的后验验证，先使用 **Small**：3 dynamics blocks、token dim 96、FFN dim 192、721,464 参数。模型构造默认仍是 Standard（54,071,520 参数），数学设计和 ModelSpec 不改。

## 组件与 checkpoint 验证

```bash
uv run --no-sync tabu-lab tar verify                 # 真实 Small：前向/反向/更新/Null/恢复
uv run --no-sync tabu-lab tar verify --size medium   # 显式容量复核
uv run --no-sync tabu-lab tar verify --full          # 显式 Standard，等同 --size standard
uv run --no-sync tabu-lab tar verify --smoke         # 历史微型 CPU 管线，不是 Small
```

`tar inspect`、`TARConfig()`、`build_model("tabu.tar")` 的默认仍是 Standard；“验证默认 Small”和“标准模型默认配置”各有清晰入口。验证返回实际尺寸、完整 config、参数量和 config/source 哈希。

## 全量数据基本拟合

```bash
uv run --no-sync tabu-lab tar fit \
  --preregistration experiments/local/tar-small-validation/preregistration.yaml \
  --dataset iris --seed 1729 --device cuda:0 --output-root /tmp/tar-small-iris-new
```

使用完整 Iris 120/30、Diabetes 353/89；训练每次包含全部 train，重新采样 label mask 和 nominal codebook；test 使用全部 train 的可见标签支持整批 test。每 32 步重放固定的 8 个 train-mask episodes，test 只初始/最终评估，不按 test 选择 checkpoint。保留梯度预警与 W&B 标量监控。

配置明确绑定 `model_size: small` 和 `expected_parameter_count: 721464`。新验证实验复制本入口再写明本次唯一变化，不能因省略尺寸而意外启动 Standard。历史缺省尺寸配置继续按 Standard 解释，历史结果及 checkpoint 不改写。

## 整批时间预算

```bash
uv run --no-sync python experiments/local/tar-small-validation/run_batch.py \
  --output-root /tmp/tar-small-batch-new --command tabu-lab
```

DGX2 源码快照内，用已授权在线 W&B 的启动器前缀：`--command tabu-lab`。Iris、Diabetes 各 seed 1729；每条最多 128 更新或 240 秒训练循环、300 秒墙钟；两条共享 1800 秒总上限（含启动、评估、保存与退出）。W&B 组为 tar-small-validation，运行名称含 small；观察链接独立于模型证据。

本目录交付可执行协议；历史运行记录保留在原实验归档，不随本次源码接入迁移。性能与拟合结论需要核对对应 source/config/receipt，不能从本目录配置推断。

## 复核边界

六步评估仍然分别进行：组件正确性 → 可解耦/扩展/生长 → 合成拟合 → 真实预测 → 合成预训练 frozen ICL → 真实任务微调提升。Small 是后续验证的优先计算尺度，不代表这六步已经完成。关键结论在 Medium/Standard 上另作显式复核；不同尺寸、协议、stage 的结果分别记录。历史 Standard 学习率对照及尺寸对照保留原始身份。

本仓库保留协议与输入作为可复现配置。原实验的结果与 checkpoint 继续绑定原始归档；当前集成的来源字段改变 checkpoint source identity，新的运行会产生新身份。
