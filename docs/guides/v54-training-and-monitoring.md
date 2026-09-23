# TabU V5.4 训练与监控经验手册

整理日期：2026-09-22，2026-09-23 补充阶段拟合审计。适用范围：V5.4 Nano 单表、五表联合、old120、old120 replay-v2 三路及独立的 Token dynamic 扩深准备。

本文把已经用过、遇到过问题并核验过的做法整理为后续操作依据。**训练目标、数据权限和预算以用户最新指令及对应实验 manifest / decisions 为准；运行状态以带时间戳的真实读回为准。** 本文不把历史快照当实时状态，不替代设计文档或启动授权。

- 数学设计与实现合同：[V5.4 设计入口](../design/restoration-v54.md)。
- 当前加训合同：[old120 replay-v2](../../experiments/local/v54-old120-replay-v2-20260922/README.md)。
- 日常查看：[restoration5.4 专用工作区](https://wandb.ai/zj3712/restoration?nw=brg2pegawui)。
- 最新有界结论：[old120 逐表拟合与下一代设计输入](../reports/v54-old120-fit-handoff-2026-09-23.md)。
- 本文审阅所用最近一次训练快照：[2026-09-22 15:32–15:35 回查](../../experiments/local/v54-old120-wandb-20260922/heartbeat-report-20260922T0731Z.md)。当时三路仍在训练，不能据此宣称完成。

## 1. 保留下来的基本判断

1. **先验证可用、速度和数据拟合能力。** 浮点梯度或单步参数更新的小差异作为诊断，不再作为探索阶段优化的唯一否决条件；非有限值、数据泄漏、错接检查点和预算混淆仍必须定位。
2. **训练行拟合按阶段推进。** 单表 → 多表联合 → 更多表规模化 → 持续学习与抗遗忘 → 真实数据预训练后微调 → 未见表泛化。某一阶段通过预检、训练 loss 下降或平均指标变好，不自动完成下一阶段。
3. **默认、备选与经验结论分开。** 当前是组合编码 `constant_weight_composition_v1`、`supervised_row`、Query-only；`random_cell` 是训练稳定后重点验证的第一备选。Nano 是重要竞争选项。高损失表加训是当前三路采用的实验策略，尚未证明普遍优于均衡训练。
4. **先接续已有成果，再决定新动作。** 先读实验目录、决策、冻结身份、启动和终态回执。新增输出、新 run ID 必须讲清与父任务的关系；旧任务不能因为监控缺点而被误恢复。
5. **监控从属于训练。** 原始日志和检查点是事实源，W&B 是可恢复的展示与分析层。接入监控、添加统计和调整布局通常不需要动训练。

当前适配器每表使用一个 `target_column`：选中训练行的目标标签隐藏，其余可见特征作为证据；Query truth 只在 scorer/loss 边界使用。Query-only 指损失权重，不意味着删除可见支持或把完整表的 forward 改成只输入 Query。当前 old120 不使用 reserved / final_test，也不报告泛化。

## 2. 先分清实验谱系和预算

| 阶段 | 模型与预算口径 | 后续复用时要保留的区别 |
| --- | --- | --- |
| Nano 单表 | 每表 r0/r1/r2 三次独立训练，每次 16,384 步 | `16,384 × 3` 是重复预算，不是同一模型连续训练 49,152 步；各机完成性读终态 |
| Nano 五表联合 | 一个共享模型；每重复 81,920 总步，即每表 16,384 次 | r0/r1 完成；r2 后来因转入 old120 保存停止，不把原三重复计划当三次完成 |
| dgx2 old120 | 标准 Small 从头；120 × 8,192 = 983,040 正常步 | 没有 Nano 权重初始化；后续 replay 延续其状态 |
| dustinstudio old120 | Nano 从 joint5.r1 权重初始化，初始重置优化器；先每表 1,024 次，完成后接续新增 8,192 次 | old120 正常累计上限 1,105,920；更早五表历史单列 |
| gongqian-mini old120 | Nano 权重扩深为 Small-H4；新任务每表 8,192 次正常更新 | 新任务从 0 计数，父系曝光另列；不是标准 Small，也不是严格续跑 |
| 三路 replay-v2 | 保留各自正常预算，在剩余完整轮增加有界加训 | 记录 normal、extra、actual 三套计数和各自继承量 |

谱系证据：[单表](../../experiments/local/v54-nano-fit-20260921/README.md)、[五表](../../experiments/local/v54-nano-joint5-20260921/README.md)、[Small old120](../../experiments/local/v54-small-old120-20260922/README.md)、[Nano old120](../../experiments/local/v54-nano-old120-20260922/README.md)、[Nano 延长](../../experiments/local/v54-nano-old120-extend8192-20260922/README.md)、[Small-H4 初始化](../../experiments/local/v54-small-warm-old120-20260922/README.md)。这些目录的早期调度文字保留其历史含义，不能覆盖后来的迁移决定。

### Nano 扩深：可复用的是转换方法，不是名称推断

| 配置 | Backbone 层数 | Heads | Unit 层数 |
| --- | ---: | ---: | ---: |
| Nano | 2 | 4 | 0 |
| 标准 Small | 3 | 8 | 3 |
| Small-H4 变体 | 3 | 4 | 3 |
| Nano-Dynamics4x 变体 | 8 | 4 | 0 |

这些配置 width=128、FFN=256、slots=256。标准 Small 同时改变 heads，不能因张量形状可复制就假定计算函数不变。用户选择的 Small-H4 保留 4 heads，新增一层 backbone 和三层 Unit；复制 Nano 的 61 个 state entries，新增 55 个，将新增模块的 12 个 out/FF 残差出口张量清零，其余新增权重按 seed 初始化。Nano-Dynamics4x 则将原两层动态计算按 `[0,1,0,1,0,1,0,1]` 复制为四份独立可训练权重，编码与读出仍各一份，不对新增残差清零或缩放；它不承诺函数保持，效果须读独立实验回执。

父权重使用 dustinstudio 当时已落盘的 **old120 138,240 步**检查点，冻结文件与 SHA；不能在复制过程中反复读取仍在变化的 `latest` 指针。转换后使用 weights-only 初始化，**AdamW、RNG、训练游标重新开始**。初始三类 episode 输出差为 0；完整 120 表初始固定 Query 比较中，63 张离散目标表 accuracy 一致，数值最大 R² 差约 `9.06e-7`、NMSE 差约 `8.22e-7`。这支持该检查点的起点行为得到保留，不证明以后训练轨迹相同、每步更快或最终更优。

可复核：[转换回执](../../experiments/local/v54-small-warm-old120-20260922/conversion-small-h4.json)、[初始状态](../../experiments/local/v54-small-warm-old120-20260922/initial-checkpoint-readback.json)、[完整初始 Query 比较](../../experiments/local/v54-small-warm-old120-20260922/initial-fixed-query-comparison.json)。起点已有能力不得记作新任务的训练收益。

### 从一次 Small-H4 转换，发展为可逐步加深的模型

Small-H4 展示了一条值得持续发展的路线：**先把较小模型训练好，再在保持起点函数的条件下增加深度，继续利用已积累的权重。** 它不必止于 Nano → Small-H4；固定兼容的宽度、heads、编码和 readout 合同后，backbone 深度可进一步从 3 增至 6、12 等，Unit 深度也可独立增加。这里的层数序列是后续方向，不是已运行实验或已获预算。

依据是当前 OMAB 的两次残差相加。简写为 `h' = h + A(h)`、`h'' = h' + F(h')`，将新增模块的 attention 输出矩阵和 FFN 最后输出矩阵置零，有限值条件下新增模块起点为恒等映射；串接多个这样的模块仍保持起点函数。归一化在 FFN 分支内部，不会单独改写主残差。已有层和 heads 保留，新增分支的内部参数正常随机初始化，**不是把新增层所有参数全部清零**。出口可先获得学习信号，随后内部参数逐步参与学习；起点不改变输出不等于新增层被永久冻结。

必须区分三个层次：

- **结构已支持自定义深度**：[V54Config](../../src/tabu_lab/models/restoration_v54/config.py) 可覆盖 backbone 层数和 `unit_layers`；[backbone](../../src/tabu_lab/models/restoration/backbone.py) 没有仅限 2/3 层的结构约束。
- **已验证的是一次具体转换**：[grow_from_nano.py](../../experiments/local/v54-small-warm-old120-20260922/grow_from_nano.py) 目前严格检查标准 Nano / Small-H4，并写定新增的 `backbone.layers.2`。它尚不接受任意父深度和目标深度，不能拿现有命令直接宣称支持 3→6→12 层。
- **后续可推广通用转换器**：保留所有旧 backbone/Unit 层，只初始化新增层，逐张量验证映射，检查有限前向及固定 Query 起点，并显式选择优化器/RNG是重置还是迁移。当前只验证过扩深时重置优化器；保留旧层 AdamW moments 的通用增长尚待实现验证。

结构上可选择任意有限深度，工程上受内存、每步耗时和优化稳定性限制。起点保留不保证继续训练时不会遗忘，也不保证越深越好；增加宽度、修改已有 heads 或编码还需要另外的转换设计。后续应按需要分阶段扩深，同时检查新增层是否学到有效修正、逐表 fixed 是否改善以及收益是否抵得上计算成本。深度变体记录实际 backbone/Unit 层数，不能借用标准 Medium/Large 名称掩盖宽度和 heads 的差异。

进一步研究见[Token dynamic 复制加深与数学分析](../research/v54-model-growth/README.md)。重点候选是只复制训练好的 backbone，保留编码和读出，以已有动态计算初始化新增深度；LL 用新表示和可见支持重新拟合，为保持或恢复拟合提供条件，但尚未验证该方案的训练收益。原权重串联不要求严格幂等，不能用不等价直接否定其价值。文中的等距扩宽是另一个独立数学补充。

## 3. 主机和运行时：记住入口，每次核对实际能力

以下为这批实验使用的配置，不是机器当前空闲的证明：

| SSH 入口 | 既有训练环境 | 这批 TabU 的精度 |
| --- | --- | --- |
| `dgx2` | 用户 cms；launcher 绑定的 GPU 容器 | CUDA / FP64 |
| `deepthought` | 用户 windb；`/home/windb/tabu-cuda/venv` | CUDA / FP64 |
| `dustinstudio` | 该用户下 `~/.wehub/envs/train-torch213-py312-mps-20260920` | MPS / FP32 |
| `gongqian-mini` | 同名固定环境，相对该机用户 home | MPS / FP32 |
| `zichao-mini` | 同名固定环境，相对该机用户 home | MPS / FP32 |
| `jarvis` | `/home/dustin/tabu-cuda/venv-jarvis`；launcher 尚未绑定，使用已记录的主机本地 Python | CUDA / FP64 |

主机事实源为工作区 `team/machines.md`；运行时事实源为 `team/runtime-environments.md`，位于 `/Users/cms/.openclaw/workspace/`。正式复现沿用对应回执的固定环境；前五台既有任务使用固定 profile，jarvis 尚无 launcher 绑定，须显式使用 `/home/dustin/tabu-cuda/venv-jarvis/bin/python`：

```bash
# 在目标主机查看本批固定环境；不是启动训练。
~/.local/bin/wehub-python --profile train-20260920 --runtime-info
```

启动前依次核对 SSH 用户、实际代码根、source/data/manifest SHA、运行时、device/dtype、剩余磁盘、在跑的真实命令和普通 preflight。CUDA 的 docker launcher 与容器内 Python 是同一任务的两个层次，不能重复计为两项训练。每台最多一项当前授权训练；仅有 `nvidia-smi` 或进程存活不足以证明训练在增长。

MPS 已有端到端 FP32 和 device-local readout 路径。当前冻结配方在导入 Python 前设置 `PYTORCH_ENABLE_MPS_FALLBACK=0`、`PYTORCH_MPS_FAST_MATH=0`；新 runner 未接好时应查适配差异，不能据此静默退回 CPU 或重装环境。跨 CUDA FP64 / MPS FP32 不是逐位等价接续。

授权实验若被矿工占用，可按既有授权核对路径、UID、自动拉起入口后让位并读回资源；不自动恢复矿工，不触碰无关服务，不把主机清空扩大为额外实验授权。dgx1 不自动并入训练池。

## 4. 速度优化：先减无用计算，再用实测判断

本轮已落地的关键优化：完整 episode 仍先验证、编码，完整可见支持和所有 shared-LL fit centers 保留；**只减少损失不需要的 readout**。CUDA/CPU 训练只求非零损失目标，MPS 跳过完全不参与损失的列，保留活跃列原请求行。固定评估仍使用完整 readout。CPU 采样时保存 mask audit，再搬到设备，减少立即回读造成的同步；没有跨训练步复用随机 episode。

2026-09-21 的同模型、同表、同种子短步比较如下。每次同步测量完整 `train_step`，包含 episode 构造、forward/backward、裁剪、优化器更新及有限值检查；排除定期固定评估、checkpoint 与报告 I/O。数值是三轮均值的中位数。

| 主机 / 表 | 完整 readout 秒/步 | 优化后秒/步 | 测得比值 |
| --- | ---: | ---: | ---: |
| dgx2 / discoscm_076 | 1.3807 | 0.2968 | 4.65× |
| dgx2 / discoscm_095 | 1.3820 | 0.2971 | 4.65× |
| dustinstudio / discoscm_076 | 0.9378 | 0.2659 | 3.53× |
| dustinstudio / discoscm_095 | 0.9231 | 0.3773 | 2.45× |
| gongqian-mini / scm_mixed_v1_017 | 0.4484 | 0.1572 | 2.85× |
| zichao-mini / scm_mixed_v1_040 | 1.8014 | 0.5575 | 3.23× |

[测速方法与六份原始回执](../../experiments/local/v54-nano-speed-20260921/README.md)。这是 Nano 内部实现比较，不能写成 Nano 相对 Small 的速度优势，也不能外推为全部 old120 或整段训练同等加速；deepthought 没有这组速度比值。

一次早期 MPS 行裁剪尝试超过当时的严格更新容差，旧脚本因而没有测速。这没有证明拟合失败，也没有证明现用列裁剪是最快方案。后来按用户要求，把细粒度数值比较降为诊断：后续候选以运行正确、有限值、速度和固定 Query 曲线判断。历史测速使用当时脚本，其结果不能改称后来宽松默认下的新实验；可选 full-versus-full 控制入口存在，但未运行。

后续优化可复用的顺序是：

1. 固定模型、数据、mask/code seeds、优化器和设备，写明改变的计算范围。
2. 在同一后端交替测原版/新版，做 warmup 和设备同步；分别报告数值诊断与计时是否完成。
3. 单步收益成立后，再测包含评估、保存、数据与同步开销的实际训练吞吐，用其估算剩余时间。
4. 用固定 Query 检查拟合是否可用；启用此前被省略的损失状态时重新准备完整 readout。
5. 改正式任务实现必须冻结新源码并记录迁移，不修改运行中的源码目录。每步 `readout_scope` / `readout_targets` 优先于笼统的 benchmark 顶层标签。

## 5. 均衡覆盖后加训：最新精确定义

当前策略名：`normal120_p99x3_p95x2_p80x1_v2`。代码只有在 manifest 显式声明 `loss_replay` 时启用；不能把当前选择写成所有 V5.4 的无条件默认。

每轮先做 120 个正常更新，各表一次。用这 120 个 **normal 原始训练 loss** 按降序排名；相同 loss 按表 ID 升序。固定本轮排名后，执行三个完整 top2 pass、两个完整 top6 pass、一个完整 top24 pass。

| 档位 | 120 表的选择数 | 完整 pass 数 | 贡献 extra 步数 |
| --- | ---: | ---: | ---: |
| P99 档 | `ceil(120 × 1%) = 2` | 3 | 6 |
| P95 档 | `ceil(120 × 5%) = 6` | 2 | 12 |
| P80 档 | `ceil(120 × 20%) = 24` | 1 | 24 |
| 合计 | 档位重叠 | — | **42** |

因此前 2 表各加训 **6 次**，随后 4 表各加训 3 次，再后 18 表各加训 1 次，其余 96 表只做正常一次。每轮 **120 normal + 42 extra = 162 actual**；extra 较正常预算增加 **35%**，占实际步数约 25.93%。旧 v1 的 120+30 是历史配方，不能继续用它算当前预算。

调度的 P99/P95/P80 是固定排名配额，**不是拿监控的插值分位数作数值阈值筛表**；后者在相同 loss 时可能产生不同数量。Extra loss 不重排当前轮，也不参与正常轮统计。正常 episode stream 保持可寻址；extra 使用独立 `/loss_replay_v2` namespace，表内 extra index 连续累计，避免反复训练完全同一个额外 episode。

该做法保证所有表持续被覆盖，同时把额外计算给高 loss 表。但不同类型、尺度、噪声与偶然困难 episode 都会影响 raw loss。加训可能追逐尺度或不可约误差；是否值得，仍须比较逐表固定 Query、尾部改善与计算成本，不能仅因 median 更平滑就判定有效。

2026-09-23 的[逐表阶段审计](../reports/v54-old120-fit-handoff-2026-09-23.md)确认了具体偏斜：三路 v2 的新增 extra 分别为 62,286／127,282／55,440，**全部落在数值目标表**，43 张 nominal 与 20 张 ordinal 表均没有 extra；其中仍有一些离散表低于各自支持多数类参考。这不改变现行任务的已冻结策略，却要求下一代将目标类型分层配额、表内相对基线难度等方案列为对照，并匹配实际计算预算。不能把跨类型原始 loss 排名直接等同于跨表学习价值。

## 6. 预算和曝光必须能对账

统一使用三个量：`normal` 是均衡主循环更新，`extra` 是额外更新，`actual = normal + extra` 是真实优化器更新。表曝光还需分正常/额外、累计/本阶段新增；父模型在别的阶段的历史单独保留。

设 v2 切换边界：

- `S`：已有累计 normal；`E`：已有累计 extra；`U = S + E`：已有累计 actual。
- `N`：原合同正常累计上限；切换要求处于完整正常＋加训轮边界。

```text
remaining_normal    = N - S
new_extra_budget    = (N - S) / 120 * 42
new_actual_budget   = (N - S) + new_extra_budget
cumulative_actual_cap = N + E + new_extra_budget

运行中：remaining_actual = cumulative_actual_cap - current_actual
         new_normal = current_normal - S
         new_extra  = current_extra - E
         new_actual = current_actual - U = new_normal + new_extra
```

三路已核验的迁移边界如下；这是冻结预算，不是当前进度：

| 任务 | S | E | U | N | 本阶段 extra 预算 | actual 累计上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dgx2 标准 Small | 122,520 | 1,110 | 123,630 | 983,040 | 301,182 | 1,285,332 |
| dustinstudio Nano | 202,560 | 2,130 | 204,690 | 1,105,920 | 316,176 | 1,424,226 |
| gongqian-mini Small-H4 | 23,520 | 2,400 | 25,920 | 983,040 | 335,832 | 1,321,272 |

例如 Small 的 v2 新 actual 预算为 `860,520 + 301,182 = 1,161,702`；加上已做的 `123,630`，得到累计上限 `1,285,332`。不能把 `N` 误当尚未训练的步数，也不能把历史 `E` 清零或重复相加。

完整 normal 轮结束时各表正常次数相同，轮中可相差一次；extra 曝光本来就不均衡。报告“平均每表”时必须说明是 normal 还是 actual，不能用 `actual / 120` 代替每张表的实际曝光。ETA 使用本阶段近期含评估/保存开销的吞吐，并说明其不确定性；不拿短步 benchmark 直接乘全部剩余步数。

## 7. 检查点接续与策略迁移

| 操作 | 模型 | AdamW / RNG / 游标 | 应如何命名 |
| --- | --- | --- | --- |
| 同配方、同后端断点续跑 | 保留 | 完整保留，包括部分轮状态 | resume；具体一致性由回执限定 |
| 完成后增加预算 | 保留 | 按接续合同保留，旧输出冻结 | 新不可变 continuation，明确新增与累计预算 |
| Nano → Small-H4 | 显式转换权重 | 新优化器、新 RNG、新任务游标 | 架构转换后 weights-only 初始化 |
| 均衡 / v1 → 新加训策略 | 保留 | 保留，另记录新策略和冻结代码 | strategy migration，不能称配方完全不变 |

策略迁移的已验证流程：先备齐新源码、manifest、工具并核对 runtime；父训练安全停止落盘；若处于未完成 v1 轮，用**原冻结代码和唯一新 bridge 输出**补完该轮 normal 与既定 extra；待主机空闲，完成新版本普通 preflight，再从完整边界迁移并唯一启动。不能在同机父 trainer 仍运行时并行做加速器预检。v1 一轮 150 actual，补齐最多 149 步，属于旧配方预算。不能回滚、跳过已排队 extra 或追补所有历史轮。

保存的不只是模型：还包括 AdamW、RNG、normal cursor、部分轮 loss、冻结 extra 队列及位置、逐表 extra episode index、normal/extra 曝光、父 SHA 和祖先链。`base_counts` 含继承的总曝光；`base_extra_counts` 单独记录继承 extra。新任务启动后读回**实际初始 checkpoint**，不能只检查内存转换成功或 launcher 回执。

部分轮 loss 和待执行队列是停止 checkpoint / 同策略 resume 必须保留的状态。切换策略时先由 bridge 消费完旧队列，新策略在空队列边界开始；不是直接把旧待执行队列交给新策略。

本次三路 v2 共同冻结 41 文件，源码 SHA-256 为 `0dfd1f3a6e350d5a5434f128ec25ce4277699f1a240f9891444b31374534c421`；数据 SHA-256 为 `f14ca644085f62c06a2c172f20773d6f4fe8dbdd6b1481478ff3f62d2e81c8dd`。三路首个真实 162 步均核对过覆盖、六个 pass、继承 extra index 和 6/3/1 配额；这是执行正确证据，不是拟合改善证据。见 [冻结身份](../../experiments/local/v54-old120-replay-v2-20260922/source-freeze.json)与[迁移、首轮及验证总回执](../../experiments/local/v54-old120-replay-v2-20260922/validation.json)。

## 8. Loss 与固定评估：每张图回答不同问题

| 数据 / 图 | 计算内容 | 横轴与用途 |
| --- | --- | --- |
| 远端 `updates.jsonl` | 每个 actual 更新的原始 loss、表、类型与计数 | 全量事实源；正常和 extra 都保留 |
| `train/loss` | 每 100 个本阶段 actual 步采样一次的原始单步值 | `train/update`＝本阶段新增 actual；看突变与瞬时难度，不当完整曲线 |
| `replay/loss` | 每次 extra 的原始 loss | `replay/update`＝本阶段新增 actual；同时保留 group / table |
| `train_cycle/*_loss` | 完整 120 个 normal loss 的分布 | `train_cycle/update`＝本阶段新增 normal；看均衡覆盖后的趋势 |
| `fixed/<table>/r2`、`nmse`、`accuracy` | 同一检查点、固定训练行 Query bank 的逐表指标 | `fixed/update`＝本阶段新增 actual；用于判断拟合 |

W&B `_step` 在这套镜像中是 outbox 投递序号，不是训练步数；绘图必须使用对应的显式横轴。晚到的 fixed 事件按自己的评估步数追加，不能因单步训练点已经向前推进而丢弃。

### 完整循环统计的约定

从全量 journal 聚合，每 120 个 normal 必须恰好覆盖 120 个表；未完整的轮跨 poll / observer 重启保存，不提前出点，不从 `train/loss` 的 100 步抽样再算均值，也不做混入 extra 的滑动窗口。

- Mean：120 个表 loss 的等权平均，容易被少数极端值拉高。
- Median：典型表的本轮 loss；不能反映尾部是否拟合。
- P95 / P99：高损失尾部；需要结合对应表及 fixed 曲线定位。
- P05：低损失一侧的补充诊断，仍保留在历史中。

分位数对升序 loss 在位置 `(120 - 1) × q` 做线性插值。它们是**120 个表级单步 loss 的分布**，不是 Query 单元格误差分位数、均值置信区间或同一 checkpoint 的评估。一轮跨越参数更新，换了采样方式或loss定义后也不能不加说明地拼接。

### 固定 Query 的最小报告单位是每张表

每次至少保留：表 ID、target column/type、固定 bank/Query 地址身份、评估 actual/normal、当前值、本输出 best 及其步数、可见支持均值/多数类参考、Query 数与逐表 normal/extra/total 曝光。R² / accuracy 越大越好，NMSE 越小越好；不同聚合定义的 R² 和 NMSE 不应假定可互相推算。

当前 old120 每 `15,360 actual` 做固定评估，checkpoint 间隔 `7,680 actual`；停止和最终落盘另看终态合同。新的 v2 输出尚无完整 fixed 时留空，不能借父任务 best 填数。本输出 best 只来自本输出完整评估；若显式做了起点评估，要注明它是继承能力。

**采样时刻也要区分。** 当前 journal 曝光可能晚于最近 fixed；云端 fixed 事件绑定的是该次评估时的精确曝光。不能把“现在已经训了多少次”贴到较早的评估值上。先比较相同 target/Query bank/曝光，再谈模型差异。

2026-09-22 回查提供了一个具体例子：Nano 的 `discoscm_090` 在 actual 261,120 的当前 R² 为 0.326604，本输出最佳 0.381102 出现在 245,760。只展示 best 会掩盖回落；只展示总体 median 又会掩盖弱表。完整逐表证据见本文开头的时间戳报告。

## 9. 监控与训练解耦，历史才能安全补录

```text
远端冻结 trainer
    └─ 原始 journal / evaluation / checkpoint / terminal
          └─ SSH 只读 observer（本机，每 panel 一个）
                └─ SQLite outbox + 原始/聚合游标 + 部分轮状态
                      └─ W&B history / summary
                            └─ 具名 workspace
```

这次给已有训练接入 W&B、增加 mean / median / P05 / P95 / P99 都沿用这条路径。训练不需要为了展示功能而停止或重跑；observer 失败也不启动或重启远端训练。复用 [共享监控入口](../../experiments/local/v54-old120-wandb-20260922/README.md)、[cycle_loss.py](../../experiments/local/v54-old120-wandb-20260922/cycle_loss.py) 和 [wandb_mirror.py](../../experiments/local/v54-old120-wandb-20260922/wandb_mirror.py)，不要每个 panel 复制一套易漂移的解析器。

关键状态包括唯一 `event_id`、本地事件序号、raw/cycle 独立游标、未满轮 loss、加训 phase/pass、run ID/source/output/起点/表集合身份。聚合状态与产生事件在同一事务持久化；文件锁保证单 observer。历史补录使用独立指标轴与事件身份，保留旧原始点，不通过清空 outbox 达到“重来一次”。

升级 observer 时，先核验准确的本机 PID、命令和 panel 根；用 SQLite backup 保存一致快照与配置/游标。此次 pinned SDK 0.23.0 的刷新验证使用一次 SIGINT，使旧 observer / wandb-core 有机会清理，再等退出和文件锁释放，用原 argv、SQLite 和 run ID 恢复。SIGTERM 不能假定等价于这种 flush 路径。**这是特定版本的已验证办法，不是向任意 Python 或远程 trainer 发信号的通用命令。** 退出或锁释放不符合预期时先定位，不强杀重试。

本机监控环境是 `/Users/cms/.wehub/envs/tabu-wandb-monitor-20260922`，与训练 runtime 分离。认证复用已有登录；不把密钥、表格内容、权重、原始异常敏感文本或代码上传到监控。新增图表不要求在训练环境安装 W&B。

## 10. 云端“运行中”不等于收到数据

验收必须分别回答四个问题：远端 trainer 是否增长；本地 observer 是否消费 journal；outbox 是否持久化；云端 history 是否收到对应 payload。SDK enqueue、observer PID 存活或页面 Running 都只证明其中一层。

可复用的核验步骤：

1. 固定一个有限 outbox 快照或序号窗口，记录边界、事件数量与身份；不要不断追逐训练新增的最新尾部。
2. 对该窗口读真实 API history，逐事件比较字段和值，包含各表 fixed 当前/best/reference、评估时曝光与计数语义。
3. 分页同边界可能返回重复行；按 [cloud_history.scan_unique](../../experiments/local/v54-old120-wandb-20260922/cloud_history.py) 去重。同一 `_step` 内容冲突必须定位，不能随意保留一个。
4. 恢复前后各取有限区间，检查缺失、重复、差异和单调序号；不要只信 `last_enqueued`。本地去重并不单独证明云端 exactly-once。
5. 分别保存 SDK 状态、云端读回和远端训练增长证据；轻微上传滞后先等下一次观测，不机械重启。

Mean 与分位数可以是不同的稀疏 history 事件，读取时按事件类别分别取字段；一次要求所有类别的键可能只得到空交集。耗时也要对齐边界：接续的新增 wall time 减去父 terminal 的累计时间（包含末次评估），不能改减父 journal 尾行时间。

本轮 observer 升级后的验证使用每路恢复边界前 24、后 48 个事件，72 个 payload 全字段匹配，并确认旧 outbox 前缀保留；最近回查另核对完整 fixed 事件与 120 表集合。零差异仅覆盖冻结快照，不应宣传为未来永久无丢点。证据见 [lifecycle-resolution.json](../../experiments/local/v54-old120-wandb-20260922/lifecycle-resolution.json) 与[最近云端回查](../../experiments/local/v54-old120-wandb-20260922/heartbeat-cloud-20260922T073444Z.json)。

## 11. 主动停止、失败和预算完成分开记录

旧镜像曾把所有非 `completed` 终态都以非零 exit 结束，导致用户授权的迁移停止显示 Failed。修复的关键是拆开三个事实：**训练为什么结束、预算是否用完、监控会话是否正常关闭**。

| 情形 | 必要证据 | 监控处理与结论 |
| --- | --- | --- |
| 正常完成 | 无 error/error_type 的 completed，update 合法且无明确持久化/身份矛盾；预算另核对 actual 与 normal 上限 | 可正常关闭；有结束状态仍不能直接宣称拟合达标 |
| 用户授权的保存停止 | 无 error/error_type；update=durable_update；终态身份一致；checkpoint 实际字节 SHA 验证 | `finish(exit_code=0)`；`closed_after_stop`，`training_intentionally_stopped=true`，`training_budget_completed=false` |
| 真实异常或证据不全 | 错误、身份矛盾、未知终态，或停止终态缺所需 durable/hash 证据 | 保留失败/待诊断，不把标签美化为成功 |

检查点指针和文件名匹配不够，要验证实际字节；字节完整性又不等于 Torch 加载或张量兼容性。训练最终完成还需退出原因、最终 checkpoint、最终固定结果和计数对齐。**W&B Finished 只表示会话关闭，不表示训练预算完成。**

对已有六个误标旧 run，本次先审计父 checkpoint 与 bridge，再用原 ID `resume="must"`、不调用 `log`、正常 finish，恢复原 summary 并添加停止原因和后继链接；核对原 storage ID/config/name/history 末步及原 summary。SDK 会更新会话元数据，不能说云端毫无变化。这是针对已确认主动停止的修复，不可批量应用于所有 Failed，也不要为改标签恢复旧 trainer 或旧 observer。

复用的是 [terminal_lifecycle.py](../../experiments/local/v54-old120-wandb-20260922/terminal_lifecycle.py) 的证据判断；一次性修复脚本不是无条件运维入口。证据：[终态验证](../../experiments/local/v54-old120-wandb-20260922/terminal-lifecycle-fix-readback.json)、[旧 run 云端最终读回](../../experiments/local/v54-old120-wandb-20260922/lifecycle-cloud-final.json)。消费者先排空 raw/cycle outbox，SDK finish 返回后才持久化本地已关闭状态，避免过早“完成”。

## 12. 给用户看的工作区应少而明确

用户最关心的四图依次是 **P99、P95、median、mean loss**，放在首屏 2×2，横轴统一 `train_cycle/update`。横轴 `train_cycle/update` 的数值除以 120 是本阶段新增的正常覆盖轮数，不是总 actual，也不包含父训练历史。P05、逐表 fixed、replay 和原始单步数据仍保留，不需要全挤在首屏。

这次只改个人默认视图的过滤后，用户仍从通用项目入口看到三十多个实验。由此保留的经验是：**服务端过滤配置正确，不代表用户打开的就是该视图。** 后来创建具名 [restoration5.4](https://wandb.ai/zj3712/restoration?nw=brg2pegawui)，显式过滤以下三个 ID，最多三条 run 曲线，关闭自动生成 panels，直接给带 `nw` 的链接：

```text
v54-small-replay-v2-old120-dgx2-20260922
v54-nano-replay-v2-old120-dustin-20260922
v54-small-h4-replay-v2-old120-gq-20260922
```

新 workspace 复用实时 run，不复制、移动、删除或重记实验。视图修改前保存原 spec，修改后通过准确 URL 重载检查 ID 过滤、panel 顺序、轴和自动生成设置。这里已完成配置/API 读回；没有把未登录浏览器的结果称为用户登录视角的视觉验收。[工作区回执](../../experiments/local/v54-old120-wandb-20260922/restoration54-workspace-receipt.json)记录了验证范围。

## 13. 日常回查与常见故障

最小回查顺序：

1. 读最新用户决定、panel README/decisions、launch/migration、冻结身份；`preparing` / `launching` 仅读取，不与主线程争抢启动。
2. 核对真实命令和 trainer，比较至少两个时间点的 update；读取 loss/gradient 有限性、checkpoint 与 terminal，区分活着、增长、完成。
3. 对账 normal/extra/actual、每表曝光、剩余预算；核对正在执行的 replay 版本，不能让旧解析器定义新日志。
4. 读最近完整 fixed：逐表当前、本输出最佳、参考、评估时曝光；没有就明确写没有。
5. 核对 observer/outbox/cloud，按需要抽有限窗口读回；保存时间戳 JSON/Markdown 和当前入口。
6. 只读回查不自动启动、重启或追加预算。终态后按当前授权收口，全部当前任务终结再总结并暂停回查。

当前四个 panel 的 `monitor.py` 可从仓库根调用；它们通过 SSH 只读训练，**会写本地时间戳快照及 CURRENT.md**，不会启动或改变训练。jarvis 的新阶段曝光从零开始，不能继承 Nano 父任务的统计：

```bash
/Users/cms/.wehub/envs/tabu-wandb-monitor-20260922/bin/python \
  experiments/local/v54-small-replay-v2-old120-20260922/monitor.py

/Users/cms/.wehub/envs/tabu-wandb-monitor-20260922/bin/python \
  experiments/local/v54-nano-replay-v2-old120-20260922/monitor.py

/Users/cms/.wehub/envs/tabu-wandb-monitor-20260922/bin/python \
  experiments/local/v54-small-h4-replay-v2-old120-20260922/monitor.py

python3 experiments/local/v54-nano-dynamics4x-old120-jarvis-20260922/monitor.py
```

| 现象 | 先查什么 | 合理处理 |
| --- | --- | --- |
| 单步 loss 大幅跳动 | 当前表、目标类型、raw loss 尺度与采样 episode | 同时看完整 normal 分布及逐表 fixed，不删原始值、不只加平滑 |
| mean 变好但仍有弱表 | P95/P99、逐表 R²/NMSE/accuracy、参考与当前/best 差 | 明确未通过的表；不宣称整体达标 |
| 新阶段 fixed 图为空 | 新输出是否已完成首次固定评估 | 明确等待；不继承父 best 充当新结果 |
| `extra_top2` 或 inherited extra 被报错 | parser 是否仍只识别 v1 | 升级共享只读监控；本次 H4 就是解析器误报，训练无需重启 |
| 云端曲线停了但 journal 增长 | observer、锁、outbox 游标、SDK/网络 | 修监控层，保留训练；有限快照验证实收 |
| “一台两项训练” | docker launcher 与实际 Python 的父子关系 | 核对命令与输出，避免误停同一任务 |
| 旧 run 显示 Failed | terminal、durable update、checkpoint bytes、用户停止意图 | 符合证据才正常关闭；真实错误保留失败 |
| 仍看到很多旧曲线 | 用户链接是否带正确 workspace ID | 给具名视图链接并重载验证，不删历史 run |
| 某机慢或不增长 | live runtime、资源、I/O、预检、日志及阻塞位置 | 先诊断；旧主机能力表不是实时健康证明 |

复用报告的最小模板：

```text
时间（UTC及本地）／panel／run ID／source与manifest身份
主机／模型变体／初始化父SHA／runtime／device-dtype
状态：真实trainer命令、增长区间、terminal与checkpoint证据
预算：actual/上限，normal/上限，extra继承/新增，剩余actual
正常轮：本阶段轮数，P99/P95/median/mean（仅完整normal）
fixed：评估actual/normal，表/target/bank，当前/本输出best/最佳步/参考
曝光：该次fixed时各表normal/extra/total；当前journal曝光另列
监控：observer状态、固定outbox窗口、云端缺失/冲突/重复数
结论：本次证明了什么、未证明什么、下一项已授权动作或继续观察
```

## 14. 哪些经验成立，哪些问题仍要实验

**已有执行证据**：Nano 在所测 CUDA/MPS 配置的 readout 优化；Small-H4 的明确权重转换与初始行为检查；保留 optimizer/RNG/曝光的策略迁移；完整 120 normal 的统计；独立 observer 补录和升级；有限窗口云端实收；主动停止生命周期处理；只展示三路的具名工作区配置。2026-09-23 的[阶段报告](../reports/v54-old120-fit-handoff-2026-09-23.md)和[三份冻结检查点](../../experiments/local/v54-old120-freeze-20260923/README.md)提供继续研究的具体起点。

**还不能写成结论**：加训普遍优于均衡；高 raw loss 就代表最值得学；warm start 一定更快收敛；Nano/Small/H4 的公平排名；120 表全部拟合成功；对 MLP/XGBoost 的公平优胜；reserved 行或未见表泛化。

后续有价值的比较是在相同数据/target/Query bank、种子计划、初始化、预算口径与后端条件下，只改变要研究的因素。策略比较同时报告 matched actual 计算预算与 normal 覆盖，不能让额外计算隐藏在“每表相同步数”中。本次规范化 manifest 核对表明三路的表、数据 SHA、target、probe 设置及 mask/code/window/evaluation seeds 一致；差异在架构、初始化、父历史、部分 model/order seeds、预算、策略切入点与后端精度。它们可用于发现问题，不能承担单因素因果结论。

MLP/XGBoost 的历史报告也明确存在训练标签量、掩码与聚合差异，见[基线边界复核](../reports/v54-nano-fit-baselines-monitoring-2026-09-21.md)。进入 held-out 阶段应先冻结方案、checkpoint 选择规则和信息权限；不要把最终测试集变成日常调参面板。

## 15. 后续维护入口

| 要查的问题 | 首选事实源 |
| --- | --- |
| 模型、编码、Episode 与实现范围 | [V5.4 设计入口](../design/restoration-v54.md)；数学改动回 owner TeX |
| 历史速度证据 | [Nano speed](../../experiments/local/v54-nano-speed-20260921/README.md) 的 results；不要覆盖旧回执 |
| 当前 replay 规则、预算与迁移 | [replay-v2 根](../../experiments/local/v54-old120-replay-v2-20260922/README.md)及三个 panel |
| dgx2 标准 Small | [Small v2 panel](../../experiments/local/v54-small-replay-v2-old120-20260922/README.md) |
| dustinstudio Nano | [Nano v2 panel](../../experiments/local/v54-nano-replay-v2-old120-20260922/README.md) |
| gongqian-mini Small-H4 | [H4 v2 panel](../../experiments/local/v54-small-h4-replay-v2-old120-20260922/README.md) |
| jarvis Nano-Dynamics4x 独立任务 | [复制扩深 panel](../../experiments/local/v54-nano-dynamics4x-old120-jarvis-20260922/README.md)；只认该目录真实 launch / terminal 回执 |
| 统计、outbox、云端实收、生命周期和工作区 | [共享监控目录](../../experiments/local/v54-old120-wandb-20260922/README.md)；旧章节按所述阶段阅读 |
| 旧 v1 的 120+30 历史 | [v1 rollout 报告](../reports/v54-old120-replay-rollout-2026-09-22.md)，不能代替 v2 合同 |

后续稳定经验补入本文；新实验进度写各 panel 时间戳回执，避免把手册变成滚动日志。协议变化先改对应设计/manifest/实现并冻结身份，再更新本指南。保留唯一清楚的当前入口，同时保留历史配方的原意。
