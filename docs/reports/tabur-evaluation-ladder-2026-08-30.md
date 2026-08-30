# TabUR evaluation ladder — 2026-08-30

这是一次本地 bounded diagnostic，不是 formal receipt、benchmark 或 accepted
capability claim。运行入口是：

```bash
PYTHONPATH=src python scripts/run_tabur_evaluation_ladder.py --device cpu
```

本轮完整 runner 在当前 clean worktree 的 CPU float32 路径通过，六层状态均为
`implemented / passed / local_unissued / none`。`dgx2` 的 CUDA/PyTorch 环境已
确认（NVIDIA GB10、CUDA 可用），但检查时该主机的 SGLang `qwen38-27b` 服务正在
占用 GPU，因此没有抢占式长跑。

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
- 所有 runner 支持 `--device cpu|mps|cuda|auto`；本机 MPS Stage 3 smoke 通过。

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
