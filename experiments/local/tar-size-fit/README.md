> 后续验证优先使用 [Small 验证入口](../tar-small-validation/README.md)。本目录保留其原始 Standard 或尺寸对照协议，不作为默认后验验证入口。

# TabU-TAR 尺寸系列快速拟合

同一 TAR 架构的三个命名尺寸由 `tabu_lab.tar_sizes.config_for_size` 唯一解析：

| 尺寸 | Dynamics blocks | Token dim | FFN dim | 参数量 | 用途 |
|---|---:|---:|---:|---:|---|
| Small | 3 | 96 | 192 | 721,464 | 快速组件、数值与基本拟合诊断 |
| Medium | 6 | 192 | 384 | 5,521,008 | 容量扩大后的行为与拟合复核 |
| Standard | 15 | 384 | 768 | 54,071,520 | 现有标准设计，保持不变 |

三个尺寸保留 8 heads、32 semantic slots、128 fixed inducing slots、同一公式和两个 gates、typed terminal、mask/codebook 协议。只改深度、token dim 及 FFN dim（保持 2 倍 token dim）。它们是同一架构的尺寸预设，不是新增模型代际、版本号或 registry 身份；模型仍为 `tabu.tar`，尺寸及完整配置/哈希在运行记录与 checkpoint 中显式绑定。默认 TARConfig、模型正文、ModelSpec、预训练指针均不修改。

```bash
uv run --no-sync tabu-lab tar sizes
uv run --no-sync tabu-lab tar inspect --size small
uv run --no-sync tabu-lab tar inspect --size medium
uv run --no-sync tabu-lab tar inspect  # 原来的 Standard 默认入口
```

```python
from tabu_lab.models.tar import TabUTARModel
from tabu_lab.tar_sizes import config_for_size
from tabu_lab.models.tar.checkpoint import load_checkpoint
cfg = config_for_size("small", initialization_seed=1729)
model = TabUTARModel(cfg)
# 加载已有 checkpoint 时明确要求尺寸：
# model = load_checkpoint(path, expected_config=cfg)
```

训练配置用 `model_size: small` 或 `medium`，并绑定 `expected_parameter_count`；不设置 model_size 时仍选择 standard。尺寸预设变更若导致参数量不匹配，旧实验配置会拒绝运行。`--smoke` 仍是独立两步 CPU 管线检查，不是 Small，也不产生 Small 的拟合证据。模型实现文件和 source_digest 不因增加外部尺寸选择器变化；已有标准 checkpoint 可按原配置验证。

本目录先跑 Small/Medium × Iris/Diabetes 四个短实验，单 seed 1729、恒定学习率 1e-4、无 warmup、batch=1、每条最多 128 步。Iris 120/30、Diabetes 353/89 的全量划分不变，每步重新 mask/codebook。每 32 步重放同一组 8 个拟合 episodes；test 只初始/最终评估。保留梯度异常预警、固定评估、W&B 在线记录。不同 token dim 的实际 codebook 向量维数不同，不能称为数值上完全相同的 codebook；地址 mask 与 seed 序列相同。

```bash
uv run --no-sync tabu-lab tar fit --preregistration experiments/local/tar-size-fit/small.yaml \
  --dataset iris --seed 1729 --device cuda:0 --output-root /tmp/tar-small-iris-new
```

整批上限 30 分钟：在 GPU 源码快照中运行 `python3 experiments/local/tar-size-fit/run_batch.py --output-root ../batch --command tabu-lab`。每条最多 240 秒训练循环、300 秒墙钟，四条同一个总截止时间。W&B 使用独立组 tar-size-family-quick-fit，运行名称显式包含尺寸；结果不能与 Standard 混报。

这一系列用于快速反馈与容量递进：先在 Small 排查协议/数值问题，再在 Medium 复核，最后回到 Standard 验证。Small/Medium 成功不自动证明 Standard 成功；小尺寸失败也不直接否定整个架构。三者当前都只是实验配置。

本仓库保留协议与输入作为可复现配置。原实验的结果与 checkpoint 继续绑定原始归档；当前集成的来源字段改变 checkpoint source identity，新的运行会产生新身份。
