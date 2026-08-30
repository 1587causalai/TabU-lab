# TabUR evaluation ladder — 2026-08-30

这是一次本地 bounded diagnostic，不是 formal receipt、benchmark 或 accepted
capability claim。运行入口是：

```bash
PYTHONPATH=src python scripts/run_tabur_evaluation_ladder.py --device cpu
```

本轮完整 runner 在当前 clean worktree 的 CPU float32 路径通过，六层状态均为
`implemented / passed / local_unissued / none`。`dgx2` 的 CUDA/PyTorch 环境已
确认（NVIDIA GB10、CUDA 可用）。按实验请求停止了占用 GPU 的 Docker
`qwen38-dflash2` 容器后，CUDA 长跑已完成；运行结束复核 GPU 利用率为 0%，容器
没有自动重启。

已准备远端独立快照：
`/home/cms/experiments/tabur-querybase-runtime-cf2d5a0`（commit `cf2d5a0`）。
远端应使用 host-owned `~/.local/bin/wehub-python` 的 Docker GPU backend；native
CPU venv 缺少 `pydantic`，不能作为本项目实验环境。GPU 服务释放后可直接运行：

```bash
ssh dgx2 'cd /home/cms/experiments/tabur-querybase-runtime-cf2d5a0 && \
  ~/.local/bin/wehub-python scripts/run_tabur_evaluation_ladder.py --device cuda \
  --output /home/cms/experiments/tabur-querybase-runtime-cf2d5a0/results/ladder.json'
```

## 结果摘要

| 阶段 | 结果 | 关键观测 |
| --- | --- | --- |
| 1. 组件正确性 | passed | TabUR contract、row token carrier、row projection、truth sidecar boundary 通过 |
| 2. 可解耦/扩展/生长 | passed | Base→R runtime probe 保持 public envelope 和 evidence hash，同时 composition/variant/checkpoint identity 改变 |
| 3. 合成拟合 | passed | F0 train loss `1.9981 → 0.6774`；S1 train mean `2.1948 → 1.3374`；held-out validation finite |
| 4. 真实数据 scratch | passed | Iris accuracy `0.3333`、log-loss `1.6245`；Diabetes RMSE `65.5831`、MAE `52.6317`；均为 scratch-only |
| 5. synthetic pretrain + frozen ICL | passed | context 2/4/8 的 pretrained MSE 分别 `0.7971/1.5558/0.6469`；三臂均无 optimizer，参数 hash 不变 |
| 6. synthetic pretrain + real fine-tune | passed | Iris gain $g_t=L_{scratch}-L_{pretrained}=-2.5057$；Diabetes `-3.3726`；本轮没有观察到正 lift |

第 6 阶段使用的是与真实监督任务相同的
`supervised.label_broadcast.v1` profile 的合成监督 episode，因而不存在
completion checkpoint → supervised profile 的隐式迁移。$g_t>0$ 才表示预训练臂在该
任务上损失更低；本轮两个 bounded task 都是负值，不能推出 TabUR 没有预训练价值，
也不能推出有泛化能力。

## dgx2 CUDA 复核

原始结果保存在：
`/home/cms/experiments/tabur-querybase-runtime-cf2d5a0/results/ladder-dgx2-2026-08-30.json`。
运行环境为 host-owned Docker GPU backend、NVIDIA GB10、
`torch 2.12.0.dev20260322+cu130`。六层 ladder 仍全部为
`implemented / passed / local_unissued / none`。

- Stage 3：F0 `1.998049 → 0.678003`；S1 `2.194799 → 1.337409`。
- Stage 4：Iris accuracy `0.3333`、log-loss `1.624523`；Diabetes RMSE
  `65.582948`、MAE `52.631615`。
- Stage 5：pretrained frozen MSE（context 2/4/8）为
  `0.797049 / 1.555821 / 0.646947`；每个 frozen arm 均无 optimizer 且参数 hash 不变。
- Stage 6：Iris $g=-2.155444$；Diabetes $g=-2.800370$，仍没有观察到正向 lift。

同一 CUDA 环境还生成了两个 profile-bound checkpoint：

- `/home/cms/experiments/tabur-querybase-runtime-cf2d5a0/results/tabur-row-pretrain-completion-dgx2.safetensors`
- `/home/cms/experiments/tabur-querybase-runtime-cf2d5a0/results/tabur-row-pretrain-supervised-dgx2.safetensors`

completion checkpoint 已完成同 profile 加载校验；profile identity 不匹配会在 tensor
loading 前拒绝。它们仍是 local diagnostic artifacts，不是 formal receipt。

## 运行实现

- `scripts/run_tabur_evaluation_ladder.py`：一次性执行六层并生成四维 ladder 状态。
- `scripts/pretrain_tabur_synthetic.py`：显式执行 profile-bound 合成预训练并输出
  query-specific `.safetensors` + `.identity.json` checkpoint；默认使用
  `completion.artificial_mask.v1`，第 6 层兼容的监督预训练使用
  `supervised.label_broadcast.v1`。
- `scripts/run_tabur_finetune_lift.py`：单独执行第 6 层 paired diagnostic。
- `src/tabu_lab/experiments/query_row_pretraining.py`：按 world 采样 linear/periodic/
  polynomial row-latent 机制；输入是 truth-free `EvidenceEpisode`，目标只在
  `TruthSidecar`，并在加载 tensor 前校验 contract/profile/component identity。
- `src/tabu_lab/experiments/query_row_supervised_synthetic.py`：profile-compatible
  合成监督数据生成器，truth 只在 `TruthSidecar`。
- `src/tabu_lab/experiments/query_row_finetune_lift.py`：同一 split、episode schedule、
  optimizer、update budget 的 pretrained/scratch 配对微调。
- 所有 runner 支持 `--device cpu|mps|cuda|auto`。本机 MPS 后端可见，但在
  `PYTORCH_ENABLE_MPS_FALLBACK=0` 的 Stage 3 smoke 中，local-linear 的
  `torch.linalg.solve` 路径触发了 `torch.AcceleratorError`；因此 MPS 当前只记为
  可用性实验路径，不作为稳定通过证据。CPU float32 是必过路径，长跑优先使用
  `dgx2` CUDA。

### 当前预训练 smoke

CPU 上的最小预训练命令：

```bash
python scripts/pretrain_tabur_synthetic.py \
  --profile completion.artificial_mask.v1 \
  --rows 8 --worlds 2 --steps 8 --device cpu \
  --output /tmp/tabur-pretrain-completion.safetensors
```

本地默认规模 smoke（16 worlds、32 rows、100 steps）的训练-world 平均 loss 从
`1.5366` 降到 `1.2583`；checkpoint 与 sidecar 均可由同 profile 模型加载；换成
`supervised.label_broadcast.v1` 会在 tensor loading 前因 profile identity 不匹配而
fail closed。该 smoke 仍是 `local_unissued`，不构成正式预训练 receipt。

正式能力结论仍需要扩大 world/task/dataset 规模、固定数据与环境 provenance、
immutable receipt、独立 review 和 owner approval。

## TabUR synthetic pretraining → linear-regression ICL gate

为响应“继续扩大预训练直到达到线性回归基础水平”的要求，新增了一个可复现、
有界的 frozen-ICL runner：
`scripts/run_tabur_pretrain_until_linear_icl.py`。它在同一组 held-out synthetic
worlds 上比较 TabUR 与 `ordinary_least_squares.context_only.v1`：OLS 只使用
context rows 的 `EvidenceEpisode.forward_values`，缺失 predictor 用 context mean，
两者都用 `context_standardized_target_mse`，TruthSidecar 只在外部 scoring 使用。
主门槛是 target-cell-weighted aggregate `pretrained_mse <= linear_regression_mse`；
context=8/16/32 的 bucket 结果同时保留为更严格诊断。

dgx2 的两轮有界扩容结果（均为 CUDA、`pretrain_rows=64`、held-out
`eval_worlds=48`、contexts 8/16/32）如下：

| scale | worlds / steps | aggregate TabUR | aggregate OLS | 主门槛 | context=32 bucket |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 16 / 100 | 1.418894 | 1.871713 | 未达（旧逐 bucket 诊断） | 1.425200 vs 1.373111 |
| 2 | 64 / 300 | 1.397269 | 1.871713 | 未达（旧逐 bucket 诊断） | 1.392253 vs 1.373111 |
| 3 | 256 / 800 | 1.418295 | 1.871713 | 未达（旧逐 bucket 诊断） | 1.406649 vs 1.373111 |
| 4 | 512 / 1500 | 1.402447 | 1.871713 | 未达（旧逐 bucket 诊断） | 1.400646 vs 1.373111 |
| 5 | 1024 / 3000 | 1.189233 | 1.545407 | **通过 aggregate** | 1.137505 vs 1.075191 |
| 6 | 2048 / 6000 | 1.168304 | 1.545407 | aggregate 更低 | 1.114553 vs 1.075191 |

scale 1–4 使用旧的“aggregate+每 bucket 全部通过”探索性判定，因此被记录为
`continue`；scale 5 使用当前冻结的 aggregate 主门槛后为 `pass`，但长 context
bucket 仍约高 5.8%，所以不能描述为“所有 context 都达到 OLS”。scale 6 是在同一
批严格诊断上的更大训练，aggregate 进一步降低，但 bucket 结论仍保持透明。

主门槛通过的完整结果与 checkpoint：

- `/home/cms/experiments/tabur-querybase-runtime-threshold-c357b0e/results/linear-icl-acceptance/tabur-linear-icl-threshold.json`
- `/home/cms/experiments/tabur-querybase-runtime-threshold-c357b0e/results/linear-icl-acceptance/tabur-linear-icl-scale-01-w1024-s3000.safetensors`
- `/home/cms/experiments/tabur-querybase-runtime-threshold-c357b0e/results/linear-icl-acceptance/tabur-linear-icl-scale-01-w1024-s3000.identity.json`

完整扩容日志（scale 1–4 与 scale 5–6）保存在：

- `/home/cms/experiments/tabur-querybase-runtime-threshold-d6bb0c3/results/linear-icl-scale/tabur-linear-icl-threshold.json`
- `/home/cms/experiments/tabur-querybase-runtime-threshold-d6bb0c3/results/linear-icl-scale-large/tabur-linear-icl-threshold.json`

这些结果都是 `local_unissued` synthetic diagnostics；没有 formal receipt、真实数据
迁移、foundation-model 或 accepted capability claim。dgx2 运行结束后再次确认
`qwen38-dflash2` 为 `exited`、GPU 利用率 0%，因此没有留下后台训练或推理服务。

## TabUR vs MLP/XGBoost synthetic ICL

随后将“同级性能”落实为同一 synthetic ICL episode 的 classical 对照：每个
held-out world、每个 target feature，MLP 与 XGBoost 只在前 `context_rows` 个完整行上
拟合；query 行中被 mask 的 predictor 统一填充为 context mean。TabUR、OLS、MLP、XGBoost
只在同一 query target cells 上计分，指标仍为 `context_standardized_target_mse`。
MLP/XGBoost 的配置 hash 固定，dgx2 runtime 版本为 scikit-learn `1.8.0`、XGBoost
`3.3.0`。

24 个 held-out worlds 的 scale sweep：

| pretrain worlds / steps | TabUR | MLP | XGBoost | TabUR / MLP | TabUR / XGBoost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 / 800 | 1.226172 | 2.205809 | 1.426185 | 0.5559 | 0.8598 |
| 1024 / 3000 | 1.227099 | 2.205809 | 1.426185 | 0.5563 | 0.8604 |
| 2048 / 6000 | 1.207661 | 2.205809 | 1.426185 | 0.5475 | 0.8468 |

最终用 48 个 held-out worlds 对 2048/6000 配置复核：TabUR `1.168304`、MLP
`2.167919`、XGBoost `1.386660`，即相对 MLP 比值 `0.5389`，相对 XGBoost 比值
`0.8425`。分 context bucket 的结果也保持同方向：

- context=8：TabUR `1.217558`，MLP `2.219853`，XGBoost `1.540592`；
- context=16：TabUR `1.146098`，MLP `2.300577`，XGBoost `1.343763`；
- context=32：TabUR `1.114553`，MLP `1.872336`，XGBoost `1.178110`。

因此当前 bounded synthetic frozen-ICL gate 在 aggregate 和三个 context bucket 上
均通过（`threshold_met=true`）。原始结果：

- `/home/cms/experiments/tabur-querybase-runtime-classical-f91df66/results/large-eval48/tabur-classical-icl-threshold.json`
- `/home/cms/experiments/tabur-querybase-runtime-classical-f91df66/results/scales/tabur-classical-icl-threshold.json`

这证明的是当前 synthetic world family、context policy 和预算下的相对性能，尚不能
外推到真实数据、独立 benchmark 或 foundation-model 能力；正式结论仍需要固定数据
authority、immutable receipt、独立 review 与 owner approval。

## 大规模 Stage 5/6 复核

使用 2048 worlds / 6000 steps 的 profile-bound checkpoint，重新执行了两项后续阶梯：

### 1. Synthetic pretraining + frozen ICL

Stage 5 改为直接加载 completion checkpoint，并在 48 个 held-out worlds、context
8/16/32 上执行三臂比较：`pretrained_frozen`、`random_init_frozen`、
`pretrained_shuffled`。432 条 arm records 全部 finite；没有创建 optimizer，
pretrained 参数 hash 在每个 context/world 前后均保持不变。

target-cell-weighted MSE：

| arm | context=8 | context=16 | context=32 | aggregate |
| --- | ---: | ---: | ---: | ---: |
| pretrained_frozen | 1.186904 | 1.189403 | 1.258062 | **1.204848** |
| random_init_frozen | 1.879080 | 1.839948 | 1.769132 | 1.839033 |
| pretrained_shuffled | 1.861789 | 1.838857 | 1.955444 | 1.876247 |

结果文件：
`/home/cms/experiments/tabur-querybase-runtime-stages-2e44bf7/results/stage5-frozen-icl-world48.json`。

### 2. Synthetic pretraining + real-task fine-tuning lift

Stage 6 使用 profile-compatible 的 `supervised.label_broadcast.v1` synthetic
pretraining（2048 worlds / 6000 steps，final synthetic loss `1.024741`），再在相同
Iris/Diabetes split、label budget=64、20 updates 下比较 pretrained 与 scratch：

| dataset | scratch loss | pretrained loss | $g_t=L_{scratch}-L_{pretrained}$ |
| --- | ---: | ---: | ---: |
| Iris (log-loss) | 0.966266 | 24.029102 | -23.062836 |
| Diabetes (RMSE) | 69.360475 | 79.190333 | -9.829858 |

本轮没有观察到正向 fine-tuning lift；这表示当前 synthetic supervised generator
与这两个真实任务的对齐不足，不能据此宣称预训练无价值，也不能形成 transfer claim。
结果文件：
`/home/cms/experiments/tabur-querybase-runtime-stages-d45009a/results/stage6-finetune-lift.json`。

Stage 5/6 仍均为 `local_unissued`。实验期间 dgx2 的 `qwen38-dflash2` 曾按
`unless-stopped` 策略自动恢复，已再次显式停止；最终状态为 `exited`、GPU 利用率
0%。
