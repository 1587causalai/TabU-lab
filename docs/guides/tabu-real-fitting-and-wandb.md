# TabUBase / TabUR 真实数据拟合与 W&B 实验教程

本文保留旧 TabUBase / TabUR 实验的协议与经验。当前 TAR 的入口见 [TAR 实现指南](../tutorials/tabu-tar.md)。下述历史脚本需与原始源码归档配套使用，不能作为 TAR checkpoint 的加载或训练入口。它的目标不是给出一个固定的“最终训练脚本”，而是保证我们在改变数学设计、数据生成器、组件或训练配方之后，仍然能够回答：

1. 模型本身有没有拟合能力？
2. 预训练是否提供了真实的数据迁移收益？
3. frozen ICL 失败，是模型能力不足，还是适配协议不足？
4. 训练 loss 下降以后，是否真的能泛化到 held-out 数据？
5. 一个新实验的结果是否可以复现、比较和追溯？

当前所有真实数据实验都属于 `local_unissued` 诊断结果，不能直接升级为正式 capability claim 或 publication evidence。

## 1. 先理解实验问题

### 1.1 两个模型

当前比较的两个独立模型是：

- `tabu.query.base`：TabUBase query-family sibling。
- `tabu.query.row`：TabUR，带 row-token readout 的 sibling。

两者共享 Episode/Prediction ABI、数据协议和评估协议，但没有 checkpoint 硬依赖。TabUBase 的 checkpoint 不能被默认当成 TabUR 的前置阶段；如果要 warm start，必须显式记录映射和验证结果。

当前容量配置为：

```text
d_model=64
n_heads=4
d_ff=256
n_blocks=4
inducing_slots=2
matched_slots=4
max_features=1024
dropout=0
```

TabUR 额外使用：

```text
row_token_count=4
row_readout_mode=anchored
anchored_gamma_initial=0.01
```

### 1.2 三种能力必须分开测量

不要把下面三件事混成一个指标：

| 实验 | 是否更新参数 | 测量什么 |
| --- | --- | --- |
| Frozen full-context ICL | 否 | 预训练模型能否直接迁移到真实任务 |
| Supervised fine-tuning | 是 | 模型能否利用真实标签适配任务 |
| Scratch training | 是 | 架构和优化器的纯拟合容量 |

最有信息量的对照是：

```text
pretrained frozen
pretrained fine-tuned
scratch frozen
scratch fine-tuned
Linear baseline
```

一个模型 frozen ICL 很差、fine-tuning 很好，说明它不一定“不会学习”；更可能是预训练目标与真实任务之间存在 transfer gap。只有 pretrained fine-tuned 相对 scratch fine-tuned 有稳定优势，才能说明预训练本身带来了收益。

## 2. 数学和数据边界

### 2.1 Truth 不能进入 forward

每个 episode 分成两个对象：

- `EvidenceEpisode`：模型能看到的 context、query features、mask、support 和 roles。
- `TruthSidecar`：只在 loss 边界提供目标值。

query response 的 truth 必须被置空：

```text
context response: visible support
query response: hidden receiver/target
query features: visible
```

如果把 query response truth 放回 evidence，训练 loss 会虚假变好，整个实验失去意义。

### 2.2 数值响应的坐标必须一致

当前模型内部的 numeric readout 通常在 `context_standardized` 坐标中工作；真实数据指标通常在 raw response 坐标中报告。

绝不能直接比较：

```text
standardized prediction - raw truth
```

合法做法只有两种：

1. 将 raw truth 投影到 context-standardized 坐标后计算 loss；
2. 将 standardized prediction 逆投影成 raw prediction 后，与 raw truth 比较。

当前 query-model 真实训练诊断使用第二种：

```text
predicted_raw = predicted_standardized * context_scale + context_mean
loss = MSE(predicted_raw, raw_truth)
```

报告 regression 时同时保留 raw `rmse`、`mae`、`r2`，以及 train-scale normalized `scaled_rmse`、`scaled_mae`。

### 2.3 categorical response 必须有 categorical NLL

分类 response 需要：

- `FeatureKind.CATEGORICAL`；
- 明确的 class domain；
- context label 的 one-hot support；
- query prediction 的 probability；
- categorical NLL / normalized NLL。

只报告 accuracy 不够。至少要同时看：

```text
accuracy
balanced_accuracy
macro_f1
log_loss
normalized_nll
roc_auc_ovr_macro
```

其中 `normalized_nll = log_loss / log(number_of_classes)`，便于跨分类数据集比较。

## 3. 已验证的运行环境

### 3.1 实验主机与路径模板

以下命令使用部署时自行配置的 `EXPERIMENT_HOST`（SSH 目标）和
`EXPERIMENT_ROOT`（主机上的实验根目录）。在执行端和远程 shell 中配置这些变量；
文档不保存具体主机身份或个人目录。路径模板不是已发布的 checkpoint 下载地址。

进入选定 GPU 主机后，检查 GPU、已有 workload 和可用磁盘；旧日志不能证明当前资源空闲。

```bash
ssh "$EXPERIMENT_HOST"
nvidia-smi
df -h "$EXPERIMENT_ROOT"
```

### 3.2 Docker image

当前已验证可用的 image：

```text
wehub/ml-gpu:20260901-wandb
```

它包含 CUDA PyTorch 和 sklearn，但有两个容易踩到的环境问题：

1. 镜像默认 entrypoint 是 `python3`，因此需要显式使用 `--entrypoint python3`；
2. 镜像的 sklearn 位于 `/opt/wehub-packages`。如果设置 `PYTHONPATH`，必须保留这个路径。

标准路径：

```bash
--env PYTHONPATH=/workspace/src:/opt/wehub-packages:/opt/wehub-python
```

当前容器环境还会继承一个不可用的本地 proxy。远程 GPU 训练必须清空：

```bash
--env HTTP_PROXY= \
--env HTTPS_PROXY= \
--env ALL_PROXY= \
--env http_proxy= \
--env https_proxy= \
--env all_proxy=
```

### 3.3 OpenML 数据

当前固定数据面板是 OpenML `new6`：

```text
banknote_authentication
segment
spambase
airfoil_self_noise
concrete_compressive_strength
qsar_fish_toxicity
```

数据 manifest：

```text
${EXPERIMENT_ROOT}/20260901-capacity64-phase2-scale-v1/source/experiments/transfer-base-v2/real-full-context-frozen-icl-openml-new6.yaml
```

缓存目录：

```text
${EXPERIMENT_ROOT}/scikit_learn_data
```

OpenML direct API 有时会被 `api.openml.org` 路由阻断。当前 runner 会将其改写为 `www.openml.org`，并添加普通 User-Agent。不要绕过 pinned manifest，也不要用未记录的自动下载数据替换它。

## 4. 当前 checkpoint 与 source identity

当前两个 best checkpoint 位于：

```text
${EXPERIMENT_ROOT}/20260901-capacity64-phase2-scale-v1/checkpoints/tabu-query-base-best.pt
${EXPERIMENT_ROOT}/20260901-capacity64-phase2-scale-v1/checkpoints/tabu-query-row-best.pt
```

它们是旧式 weights-only PyTorch checkpoint，加载方式为：

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
```

它们不是当前 Evolution Kernel 的 identity-bound safetensors，因此：

- 可以用于当前的 local diagnostic；
- 不能当作 formal exact-resume checkpoint；
- 不能在没有显式 `StateProjection` 的情况下宣称 warm start 兼容。

本轮实验使用的 source commit：

```text
7a3cec41f18920fa7bfc994b600ac4b0cbadc72e
```

source archive：

```text
${EXPERIMENT_ROOT}/20260901-capacity64-fit-v1/source-7a3cec4.tar
```

source archive SHA-256：

```text
18f7f17eaaf14010d4235548641a4bf0b949c9b21fb674a4daff8ec252f06e6b
```

开始分析前，先记录：

```bash
sha256sum ${EXPERIMENT_ROOT}/20260901-capacity64-phase2-scale-v1/checkpoints/*.pt
git -C /path/to/source rev-parse HEAD
```

## 5. Frozen full-context ICL 基线

### 5.1 协议

对每个数据集和每个 split seed：

1. classification 使用 stratified 70/30 split；regression 使用固定随机 70/30 split；
2. 70% 的 train rows 全部作为 context；
3. 30% 的 held-out rows 全部作为 query；
4. query response truth 不进入 evidence；
5. 不创建 optimizer；
6. evaluation 前后参数 hash 必须完全相同；
7. Linear baseline 使用同一组 context 和 query。

固定 split seeds：

```text
1729, 2718, 31415
```

### 5.2 之前的 frozen 结果

较低更好。`new6` 的 frozen ICL 曾得到以下代表性结果：

| Dataset | Linear | TabUBase | TabUR |
| --- | ---: | ---: | ---: |
| banknote_authentication | 0.030 | 0.992 | 0.988 |
| segment | 0.080 | 0.997 | 0.986 |
| spambase | 0.329 | 0.961 | 0.960 |
| airfoil_self_noise | 0.702 | 1.080 | 1.099 |
| concrete_compressive_strength | 0.628 | 0.992 | 1.034 |
| qsar_fish_toxicity | 0.680 | 0.994 | 0.990 |

这只能说明 frozen transfer 很弱，不能说明模型没有拟合能力。

原始 receipt：

```text
${EXPERIMENT_ROOT}/20260901-capacity64-phase2-real-icl-v1/real-icl-eval.json
```

## 6. 监督拟合诊断协议

### 6.1 外层数据划分

外层 split 与 frozen ICL 相同。外层 held-out query rows 从训练开始到结束都不能被 optimizer 使用。

### 6.2 内层动态 episode

每一个 optimizer step 动态重新采样一个 episode：

```text
training context rows: up to 256
masked training query rows: up to 64
outer held-out rows: never used for optimization
```

训练 query 从 outer train partition 中采样，context 从剩余 train rows 中采样。分类任务的 context 必须覆盖所有 class；否则 categorical routing 没有合法 support。

动态 episode 的目的，是避免模型反复记忆固定的几张表。每次生成都要产生新的 episode identity，并将采样 seed 记录到 receipt。

### 6.3 当前推荐训练配方

第一轮诊断使用：

```text
optimizer: AdamW
learning_rate: 3e-4
weight_decay: 1e-4
gradient_clip_norm: 1.0
updates: 500
context_rows: 256
query_rows: 64
query_readout_chunk_rows: 64
```

每个数据集/seed 至少跑四个 arm：

```text
pretrained_base
pretrained_row
scratch_base
scratch_row
```

其中 scratch arm 使用同样的 architecture、optimizer、updates、episode sampler 和 split seed，只是不加载 pretrained checkpoint。

### 6.4 必须记录的结果

每个 arm 记录：

```text
initial_parameter_sha256
final_parameter_sha256
checkpoint_sha256
loss_first
loss_last
loss_min
fit_masked_query_loss
frozen_heldout metrics
finetuned_heldout metrics
```

解释方式：

- `fit_masked_query_loss` 高：模型可能没有基本拟合能力，或训练接口/坐标有 bug；
- `fit_masked_query_loss` 低、held-out 高：模型会记忆/过拟合，但规律泛化不足；
- pretrained 与 scratch 都低且相近：架构容量够，但当前预训练没有明显帮助；
- pretrained 明显优于 scratch：预训练 prior 对真实任务有迁移收益；
- frozen 很差、fine-tuned 很好：主要是 adaptation gap，不是纯容量失败。

## 7. 已验证的一次 smoke 结果

`banknote_authentication / seed=1729 / 100 updates` 的诊断结果：

| Arm | Frozen normalized NLL | Fine-tuned normalized NLL | Masked fit loss |
| --- | ---: | ---: | ---: |
| pretrained Base | 0.992 | 0.468 | 0.214 |
| pretrained TabUR | 0.989 | 0.048 | 0.002 |
| scratch Base | 0.991 | 0.015 | 0.0002 |
| scratch TabUR | 0.965 | 0.074 | 0.037 |
| Linear | — | 0.027 | — |

之后的 500-update `banknote / seed=1729` 结果中，pretrained Base 的 held-out normalized NLL 达到约 `0.0037`，pretrained TabUR 约 `0.0327`。

这组结果证明：

1. 当前模型确实能在真实任务上拟合；
2. frozen ICL 失败不能直接归因于容量不足；
3. Base 与 TabUR 的优化行为不同；
4. 必须保留 scratch control，才能判断 pretrained 是否真的贡献了收益。

## 8. 选定 GPU 主机 上运行独立实验

### 8.1 单个诊断 runner

当前 one-off runner：

```text
${EXPERIMENT_ROOT}/20260901-capacity64-real-finetune-panel-v2/real-finetune-diag.py
```

它的参数顺序是：

```text
output_path
base_checkpoint
row_checkpoint
panel_manifest
dataset_id
split_seed
updates
```

示例：

```bash
ssh "$EXPERIMENT_HOST"
docker run --rm --gpus all --ipc=host --network host \
  --entrypoint python3 \
  --env PYTHONPATH=/workspace/src:/opt/wehub-packages:/opt/wehub-python \
  --env HTTP_PROXY= --env HTTPS_PROXY= --env ALL_PROXY= \
  --env http_proxy= --env https_proxy= --env all_proxy= \
  -v ${EXPERIMENT_ROOT}/20260901-capacity64-phase2-scale-v1/source:/workspace:ro \
  -v ${EXPERIMENT_ROOT}/20260901-capacity64-phase2-scale-v1:/checkpoints:ro \
  -v ${EXPERIMENT_ROOT}/20260901-capacity64-real-finetune-panel-v2:/run \
  -v ${EXPERIMENT_ROOT}/scikit_learn_data:${EXPERIMENT_ROOT}/scikit_learn_data \
  -w /workspace \
  wehub/ml-gpu:20260901-wandb \
  /run/real-finetune-diag.py \
  /run/results/banknote_authentication-s1729.json \
  /checkpoints/checkpoints/tabu-query-base-best.pt \
  /checkpoints/checkpoints/tabu-query-row-best.pt \
  /workspace/experiments/transfer-base-v2/real-full-context-frozen-icl-openml-new6.yaml \
  banknote_authentication 1729 500
```

### 8.2 Panel launcher

当前 panel launcher：

```text
${EXPERIMENT_ROOT}/20260901-capacity64-real-finetune-panel-v2/launch-panel.sh
```

启动 500-update panel：

```bash
ssh "$EXPERIMENT_HOST" \
  'nohup bash ${EXPERIMENT_ROOT}/20260901-capacity64-real-finetune-panel-v2/launch-panel.sh \
    500 \
    ${EXPERIMENT_ROOT}/20260901-capacity64-real-finetune-panel-v2 \
    </dev/null \
    > ${EXPERIMENT_ROOT}/20260901-capacity64-real-finetune-panel-v2/panel.log 2>&1 &'
```

启动后立刻检查：

```bash
ssh "$EXPERIMENT_HOST" \
  'ps -eo pid,etime,stat,cmd | grep -E "launch-panel|real-finetune-diag|docker run" | grep -v grep'

ssh "$EXPERIMENT_HOST" \
  'find ${EXPERIMENT_ROOT}/20260901-capacity64-real-finetune-panel-v2/results \
   -type f -size +0c -printf "%f\\n" | sort'
```

查看当前任务：

```bash
ssh "$EXPERIMENT_HOST" \
  'latest=$(ls -t ${EXPERIMENT_ROOT}/20260901-capacity64-real-finetune-panel-v2/logs | head -1); \
   tail -30 ${EXPERIMENT_ROOT}/20260901-capacity64-real-finetune-panel-v2/logs/$latest'
```

不要把新的短实验写到已有的正式目录；每个新预算、source commit、数据协议或 sampler 都应创建新的 run root。

## 9. W&B 使用规范

### 9.1 何时使用 online

需要边训练边观察时使用：

```bash
export WANDB_MODE=online
export WANDB_PROJECT=tabu-lab
export WANDB_RUN_GROUP=real-finetune-new6
```

每个 arm 使用独立 run name，例如：

```text
real-ft-banknote_authentication-s1729-pretrained-base-v1
real-ft-segment-s2718-pretrained-row-v1
```

当前 `panel-v2` 的 one-off 诊断 runner 主要写本地 JSON receipt 和 step log；本节的 W&B namespace 是后续 canonical runner 的接入规范。不能因为 image 名称包含 `wandb`，就把没有实际上传的本地结果称为 W&B evidence。

不要把 API key 写进 repo、manifest、日志或 receipt。机器如果已经登录，直接继承登录状态；否则在主机上按既有凭证流程登录。

### 9.2 推荐的 W&B metric namespace

使用稳定的分层命名，避免不同实验把同名 loss 混在一起：

```text
train/loss
train/loss_mse
train/loss_categorical_nll
train/masked_query_loss
train/context_rows
train/query_rows
train/episode_index

eval/heldout_rmse
eval/heldout_scaled_rmse
eval/heldout_mae
eval/heldout_r2
eval/heldout_log_loss
eval/heldout_normalized_nll
eval/heldout_accuracy
eval/heldout_macro_f1

diagnostic/support_coverage
diagnostic/parameter_changed
diagnostic/pretrained_vs_scratch_delta
```

### 9.3 训练曲线应该看什么

不要只看一条总 loss。最小观察面是：

1. `train/loss` 是否下降；
2. `train/masked_query_loss` 是否在动态 episode 上持续下降；
3. 分类任务的 NLL 是否下降，而不仅是 accuracy 上升；
4. regression 的 raw 和 scaled 指标是否一致改善；
5. train loss 与 held-out loss 的 gap 是否不断扩大；
6. pretrained 和 scratch 是否在相同 step 上分离；
7. support coverage 是否始终合法；
8. W&B step 是否与 optimizer step 一一对应。

推荐的曲线布局：

```text
Panel A: train loss / masked query loss
Panel B: held-out primary metric
Panel C: pretrained vs scratch
Panel D: support coverage and episode size
Panel E: learning-rate / gradient norm / parameter delta
```

W&B 曲线是观察投影，不是真相源。最终结果仍以本地 JSON receipt、checkpoint hash、source identity 和 panel manifest 为准。

## 10. Receipt 结构

每个独立实验至少包含：

```text
run-root/
  real-finetune-diag.py
  launch-panel.sh
  logs/
    <dataset>-s<seed>.log
  results/
    <dataset>-s<seed>.json
```

JSON receipt 必须包含：

```json
{
  "schema_version": "...",
  "status": "local_unissued",
  "claim_boundary": "...",
  "host": "...",
  "device": "cuda",
  "source_commit": "...",
  "source_archive_sha256": "...",
  "panel_manifest_sha256": "...",
  "dataset_source_manifest_sha256": "...",
  "checkpoint_sha256": "...",
  "split_seeds": [1729],
  "optimizer": {},
  "updates": 500,
  "dynamic_episode_generation": true,
  "results": {}
}
```

Receipt 不应原位修改。改变模型、数据、objective、optimizer 或 evaluation protocol 时，创建新的 schema/version/run root。

## 11. 结果汇总方法

### 11.1 先看单数据集，再看 macro

不要先只看总平均。顺序应为：

1. 每个 dataset 的三个 seed；
2. 每个 arm 的 mean 和 seed dispersion；
3. classification macro；
4. regression macro；
5. pretrained-vs-scratch paired delta；
6. 与 Linear 的 dataset win count。

classification 的 primary metric 使用 `normalized_nll`；regression 使用 `scaled_rmse`。两者都是 lower-is-better。

### 11.2 推荐的结论表

最终汇总至少包含：

| Dataset | Task | Linear | Pretrained Base | Pretrained TabUR | Scratch Base | Scratch TabUR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ... | classification/regression | ... | ... | ... | ... | ... |

另加两列：

```text
pretrained gain over scratch
fine-tuned gain over frozen
```

### 11.3 不能做的推断

以下推断都不成立：

- 单个 dataset 超过 Linear，就说模型已经普遍超过 Linear；
- train loss 下降，就说模型学到了可泛化规律；
- smoke 成功，就说正式 pretraining 成功；
- W&B 页面有曲线，就说 receipt 可复现；
- fine-tuning 结果好，就说 frozen ICL 好；
- scratch 结果好，就说 pretrained 有用。

## 12. 常见故障排查

### `ModuleNotFoundError: tabu_lab`

检查：

```text
PYTHONPATH=/workspace/src:/opt/wehub-packages:/opt/wehub-python
```

### `ModuleNotFoundError: sklearn`

通常是只设置了 `PYTHONPATH=/workspace/src`，覆盖了 `/opt/wehub-packages`。将三个路径一起保留。

### `python3: can't open file '/workspace/python'`

镜像 entrypoint 是 `python3`。加入：

```text
--entrypoint python3
```

### checkpoint 找不到

确认 checkpoint 所在目录已单独挂载：

```text
-v ${EXPERIMENT_ROOT}/20260901-capacity64-phase2-scale-v1:/checkpoints:ro
```

并且容器内使用 `/checkpoints/...`，不能继续使用主机路径。

### `query_response_objective_loss` 报 cell readout 错误

这是模型 family 接口不匹配：该 helper 面向 cell readout，不适用于当前 `tabu.query.base/row`。query models 必须走 query-specific differentiable response loss，不能仅把 cell helper 强行复用。

### loss 不降或出现异常好结果

依次检查：

1. query truth 是否进入 evidence；
2. numeric prediction 和 truth 是否在同一坐标；
3. categorical response 是否声明 domain；
4. context 是否覆盖全部 class；
5. train query 是否与 context 重叠；
6. held-out rows 是否误用于 optimizer；
7. checkpoint 是否真的加载成功；
8. 每一步是否动态生成 episode；
9. parameter hash 是否符合 frozen/fine-tune 预期。

### W&B 没有曲线

检查：

```bash
echo "$WANDB_MODE"
env | grep '^WANDB_'
```

并确认训练进程实际继承了这些环境变量。W&B online 失败时，不能丢弃本地 JSON receipt；本地 receipt 仍然必须写完整。

## 13. 后续实验顺序

### 阶段 A：完成当前拟合能力 panel

完成：

```text
6 datasets × 3 split seeds × pretrained/scratch × Base/TabUR
```

先回答模型是否能拟合、是否能超过 Linear、预训练是否有收益。

### 阶段 B：建立真正的 W&B training curve

将当前 one-off runner 的日志进一步规范化为 W&B online run，同时保持 JSON receipt。至少做：

```text
updates = 100, 500, 2000
```

观察 fit loss、held-out loss 和 pretrained-scratch gap 的曲线，而不是只比较最后一个点。

### 阶段 C：训练预算与上下文敏感性

固定 dataset/seed，改变：

```text
training context rows = 64, 128, 256, 512
training query rows = 32, 64, 128
```

该实验用于区分：模型容量不足、support 不足、还是 episode 规模不匹配。

### 阶段 D：预训练迁移分析

固定真实 fine-tuning 配方，对比：

```text
current synthetic v3 checkpoint
older synthetic checkpoint
random initialization
```

比较的不只是最终分数，还包括：

- 达到 Linear 所需的 optimizer steps；
- 最低 held-out loss；
- seed variance；
- 不同任务类型的迁移一致性。

### 阶段 E：最后才改变 architecture

只有在确认：

1. loss coordinate 正确；
2. categorical NLL 正确；
3. dynamic episode 正确；
4. pretrained/scratch 对照完成；
5. W&B 曲线和 receipt 完整；

之后，才适合比较 `d_model`、`n_blocks`、row-token 数量或新的 readout。否则很难判断收益来自模型变大，还是来自修复了训练协议。

## 14. 最小执行清单

每次新实验开始前：

```text
[ ] 确认 source commit 和 source archive hash
[ ] 确认 checkpoint hash
[ ] 确认数据 panel manifest hash
[ ] 确认实验主机 GPU 和已有 workload
[ ] 创建新的 run root
[ ] 写清楚 outer split、inner episode 和 held-out 边界
[ ] 明确 numeric loss coordinate
[ ] 明确 categorical domain 和 NLL
[ ] 选择 pretrained/scratch 对照
[ ] 设置 WANDB_MODE=online（若需要在线观察）
[ ] 记录 optimizer、updates、context/query rows 和 seed
```

每次实验结束后：

```text
[ ] 检查结果 JSON 是否完整
[ ] 检查 parameter hash
[ ] 检查 train/held-out 指标
[ ] 检查是否有数据泄漏
[ ] 计算与 Linear 的 paired comparison
[ ] 保持 local_unissued，直到正式 review
[ ] 不覆盖旧 run、旧 checkpoint 或旧 receipt
```

## 15. 当前现场状态

截至 2026-09-01，正式 500-update panel 的运行目录是：

```text
${EXPERIMENT_ROOT}/20260901-capacity64-real-finetune-panel-v2/
```

该 panel 的描述属于 2026-09-01 历史快照；本页不声称任务当前仍在运行。后续判断必须以原始 receipt 和新的主机检查为准。

本教程本身是实验操作文档，不是对任何模型能力的证明。所有能力结论都必须回到对应 receipt、source identity、checkpoint identity 和 evaluation protocol。
