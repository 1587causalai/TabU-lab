# TabU V5.4 Nano：单表拟合基线、训练监控与 reserved 评估草案

记录日期：2026-09-21。状态：独立研究记录，供用户审阅后决定如何接入训练系统。

本文保存本次侧边讨论中的结果快照与复核结论，并提出后续评估口径。本文没有启动、停止或延长实验，没有修改训练代码、活动 manifest 或调度。文中的历史 checkpoint 不代表阅读时的最新运行状态。

## 1. 用户目标与当前结论

当前优先验证 Nano TabU 的单表数据拟合能力，以 MLP、XGBoost 和常数预测作为参照；以后再加入 reserved 指标。监督学习的 `supervised_row` 是本轮已用设置，`random_cell` 是需要单独实验的另一种任务，不能直接混入同一排名。

本次已经观察到 Nano 在若干表上持续改善，尤其 `scm_mixed_v1_040` 在 6,144–8,192 updates 间有明显跃升。但先前对话中的“公平比较、Nano 在五表中赢四表”必须撤回：临时基线与 Nano 的训练信息、固定 mask 和汇总方式没有完全对齐。这些数值有诊断价值，尚不是正式的架构胜负结论。

后续方案应同时回答两个问题：模型最终能把已训练表拟合到什么程度；为达到相同拟合质量，需要多少时间和资源。不能以当前几台主机上的不同进度替代收敛后的比较。

## 2. 本轮实验身份

本地 repo：`/Users/cms/.openclaw/workspace/projects/causal-superintelligence/TabU/tabu-lab`。

以下链接均相对此文：

- [r0 manifests](../../experiments/local/v54-nano-fit-20260921/manifests/)
- [冻结数据](../../experiments/local/v54-nano-fit-20260921/data/)
- [常数参考及逐表指标](../../experiments/local/v54-nano-fit-20260921/naive-query-reference.json)
- [当前主线预算与调度来源](../../experiments/local/v54-nano-fit-20260921/ALLOCATION.md)
- [主线状态记录](../../experiments/local/v54-nano-fit-20260921/followup-decisions.json)

| 项目 | 本轮 r0 设置 |
| --- | --- |
| 数据 | 每表 256 行；train 204 行、test 52 行，无独立 validation split |
| 任务 | `supervised_row`，target column 为 Query，fraction 0.25 |
| 模型 | V5.4 Nano；width 128、2 backbone layers、4 heads、256 slots、FF width 256、unit layers 0、subtokens 1 |
| 编码 | `constant_weight_composition_v1` 组合编码；numeric z-score |
| 优化 | AdamW，学习率 1e-4、weight decay 0.01、gradient clip 1.0 |
| 监督 | Query-only loss，state weights `[0,1,0,0]` |
| 固定评估 | `train_fit`、train partition、8 masks；每 mask 51 个 Query，共 408 次 Query 测量 |
| 单次预算 | r0 每表 16,384 updates，manifest 时间上限 172,800 秒 |
| 周期 | checkpoint 每 512 updates；固定评估每 1,024 updates，另有 initial/final |

写本文时，本地 ALLOCATION 已记录每表 r0/r1/r2 三次独立重复，每次 16,384 updates，共 49,152 updates/表；五表共 245,760 updates。它们不是单个模型连续训练 49,152 步。本文下表仅保存先前读到的 r0 快照，不表示三次重复已完成，也不由本文触发后续运行。

| 表 | target index（从 0 起） | 主机 | 已记录设备 / 精度 |
| --- | ---: | --- | --- |
| discoscm_095 | 31 | dgx2 | CUDA / FP64 |
| scm_mixed_v1_001 | 11 | deepthought | CUDA / FP64 |
| discoscm_076 | 31 | dustinstudio | MPS / FP32 |
| scm_mixed_v1_017 | 7 | gongqian-mini | MPS / FP32 |
| scm_mixed_v1_040 | 11 | zichao-mini | MPS / FP32 |

远端冻结根为各主机 `~/tabu-v54-nano-fit-20260921-8db026b/`，结果位于 `runs/v54-nano-fit-20260921/<table>.r0/`。ALLOCATION 登记的 source SHA 为 `8db026b30a6a2b071f0c0bfc8745ed686dcf4f0f68db886d61f69393d106ba87`；正式复现还需核对远端实际 source/runtime 与 manifest 身份，不能以此目录名代替验证。

## 3. 已观察结果：保留原值，禁止据此正式排名

下表为先前对话实际计算或读回的结果；Nano 来自固定评估 JSON，MLP/XGBoost 来自临时本地 CPU 计算。两类结果目前不是完全配对比较，不标赢家。

| 表 | 指标（越高越好） | Nano r0 快照 | Nano update | 临时 MLP | 临时 XGBoost |
| --- | --- | ---: | ---: | ---: | ---: |
| discoscm_095 | accuracy | 22.74% | 13,312 | 21.81% | 22.30% |
| scm_mixed_v1_001 | R² | 0.9754 | 5,120 | 0.9347 | 0.9512 |
| discoscm_076 | R² | 0.9709 | 11,264 | 0.7981 | 0.8471 |
| scm_mixed_v1_017 | accuracy | 83.77% | 16,384，final 文件 | 76.96% | 77.94% |
| scm_mixed_v1_040 | accuracy | 88.11% | 7,168 | 97.30% | 99.26% |

随后读到 `scm_mixed_v1_040` 的 8,192 update 结果为 **95.93%**，因此 88.11% 是历史进度而非最终水平。存在 final 评估文件不单独证明进程正常完成；终态仍需 terminal receipt。

Nano r0 的常数参考记录：`discoscm_095` 多数类 12.33%，`scm_mixed_v1_017` 多数类 56.22%，`scm_mixed_v1_040` 多数类 52.43%；`scm_mixed_v1_001` 可见均值 R² -0.01132，`discoscm_076` 可见均值 R² -0.00913。它们属于各自 r0 mask bank，不能直接用于 r1/r2。

### 3.1 临时基线怎样计算

对每个 mask，重新建立一个模型：只用 target 可见的约 153 行拟合，预测该 mask 的 51 个 Query；predictors 为其他所有列。这是 context-only 基线。虽然这些 Query 属于原始 train split，它们的标签并没有用于该临时模型的拟合。

- MLP：对话执行记录为 sklearn `MLPRegressor/MLPClassifier`，hidden `(64,64)`、ReLU、Adam、学习率 1e-3、alpha 1e-4、max_iter 500、tol 1e-4，StandardScaler，seed `1729 + mask_index`。完整临时执行脚本与逐地址预测尚未形成持久回执，因此口径仍待重跑核验。
- XGBoost：可复核的临时脚本 `/tmp/tabu_compare_xgb.py`；300 trees、depth 6、学习率 0.05、subsample/colsample 0.8、hist、单 CPU thread、seed `1729 + mask_index`；回归 `reg:squarederror`，分类标签先映射到连续编号。
- 临时 predictors 使用原始数值/类别编号，附加有限值指示。当前 supervised-row 的非 target 列均可见，所以这些指示基本为常数；这套临时处理不能直接当成 random-cell 的通用缺失值实现。
- 本地曾遇到同进程加载 Torch 后 XGBoost 拟合崩溃；随后用不导入 Torch 的独立子进程运行 XGBoost 得到数值。这里只记录环境现象，尚未确定根因。
- 没有进行公平的超参数搜索、计算预算匹配或多随机种子置信区间计算。临时文件可能被清理，正式基线不能依赖 `/tmp`。

### 3.2 本次复核发现的关键差异

1. **训练信息不同。** Nano 在 train 204 行上跨 episode 接受 Query 监督，同一行标签可以在其他 episode 中参与训练。临时 MLP/XGBoost 每个 mask 从头只拟合可见行，未获得同样的历史监督。故不能把 Nano 的训练行拟合与 context-only 基线直接解释成相同学习条件下的胜负。
2. **XGBoost mask seed 不同。** 正式 evaluator 先用 `SHA256(str(evaluation_seed) + '/train_fit')` 的前 8 bytes、小端整数派生 seed，再调用 `build_episode`。临时 XGBoost 脚本直接传原始 evaluation seed。两者即使都用 8 个 masks，Query 地址也不保证一致。MLP 的 mask 实现也需重新核验，不能沿用“已经一致”的说法。
3. **指标权重不同。** Nano 先在每个原始 Query 地址内平均重复测量的误差/正确性，再对唯一地址等权平均；临时基线直接平均 8 个 mask 分数。多次出现的地址因而获得不同权重，平均 mask R² 也不等于正式 R²。
4. **NMSE 定义不同。** Nano 使用各 episode 可见 target 的标准差归一化；临时 XGBoost 的 `nmse` 是除以 Query 方差，实际等于每 mask 的 `1-R²`。两者不能混用同一指标名。上表只保留 R²/accuracy，仍需修正聚合。
5. **名义类别表示尚未公平约定。** 原始类别编号的大小关系对 nominal 没有语义。正式基线应明确 one-hot/native categorical 或其他处理，记录未知类别策略，并区分表示差异与模型能力。

相应复核代码：[evaluation.py](../../src/tabu_lab/curriculum_v53/evaluation.py) 的 `evaluate_probe`、`_finish`，以及 [data.py](../../src/tabu_lab/curriculum_v53/data.py) 的 `build_episode`。

## 4. scm_mixed_v1_040：数据与训练迟滞案例

这是本轮登记为 synthetic 的混合类型表；名称暗示 SCM 系列，但具体生成机制、噪声和因果图尚未在此次核查中追溯，不能仅凭名称认定。共有 12 列：5 numeric、6 nominal、1 ordinal。target index 11 即第 12 列，是 nominal，声明 9 个类别，实际 train 只出现 4 类。

| target 类别 | train 行数 | train 比例 |
| --- | ---: | ---: |
| 4 | 106 | 51.96% |
| 7 | 56 | 27.45% |
| 3 | 26 | 12.75% |
| 1 | 16 | 7.84% |

全表 256 行的类别计数先前已被描述性查看过：4→133、7→70、3→33、1→20。这里明确留痕：reserved 标签分布已经被人工诊断查看；它们没有被本次临时基线用于拟合或计分，但不能再声称这 52 行标签完全未被查看。以后若围绕这些统计反复调方案，需要另设独立的最终测试集。

本次仅对 train 204 行复核：列 4 的各类别都对应唯一 target，映射为 `{0:3, 1:1, 2:4, 3:3, 4:1, 5:4, 6:1, 7:3, 8:7, 9:4, 10:7}`。这说明已观察训练样本中存在很强的分类线索；不证明生成分布上的确定性或因果关系。

先前“连续列互信息接近 target 熵，所以都几乎确定 target”的解释也需撤回：把每个唯一连续值当成离散类别估计经验互信息，会在有限样本上得到虚高结果，不能证明可学习的预测关系。上面的 train 类别映射是更直接的有限样本证据。

| r0 update | 固定 Query accuracy |
| ---: | ---: |
| 0 | 15.75% |
| 1,024 | 46.36% |
| 2,048 | 52.43% |
| 3,072 | 48.78% |
| 4,096 | 49.56% |
| 5,120 | 52.43% |
| 6,144 | 57.95% |
| 7,168 | 88.11% |
| 8,192 | 95.93% |

8,192 update 的 discrete encoding MSE 为 0.0037163，固定 Query 覆盖 185 个唯一地址。早期准确率接近多数类参考，之后迅速改善，支持“早期尚未充分优化”的解释；尚不能单凭这条曲线确定迟滞根因是 codec、读出、梯度、学习率或类别不均衡，也不能保证继续训练必然达到 100%。

建议以后同时记录逐类别 recall、混淆矩阵、预测类别占比、编码误差与 decode margin，用于区分多数类塌缩、少数类未学会，以及连续表示进步但离散解码尚未越过边界。

## 5. 公平比较建议：三种评估分别命名

### A. 训练行拟合能力——用户当前关心的主比较

三者均可使用同一 204 行的训练标签学习。MLP/XGBoost 在全部 train 行拟合 target；Nano 通过 episode Query loss 学习这批行。都报告这批行上的训练拟合，并让 Nano 推理时隐藏被评分 target，禁止直接复制答案。

为让评分点可对齐，普通监督基线也在 Nano 固定 mask bank 的 Query 地址上输出预测，复用同一 scorer；同时报告普通全 train 分数并注明它与固定 Query 子集的差异。各模型表示机制不同，但训练标签范围及评分地址明确一致。

以充分训练后的可达质量、达到预设误差门槛的时间、不同初始化的稳定性为主要结论。设置相近参数量的 MLP 可以作为补充；树数量与神经网络 update 不具备一一对应关系。不得仅以相同步数声称公平。

### B. 每 episode 的 context-only 恢复——补充诊断

每个 episode 仅用当前可见支持行拟合，预测其 Query；这就是当前临时基线接近的协议。若 Nano 已在该表其他 episode 中见过 Query 标签，必须标为“有表内训练历史”，不能与从头拟合的基线称为同信息比较。

严格比较 B 时，应控制所有模型的历史标签访问，例如固定目标隐藏集合并在训练历史中一致排除，或单独研究没有该表训练历史的模型。该任务与 A 的训练拟合不同，不应为了做基线而偷偷改变当前主任务。

### C. Reserved 行评估——未来阶段

先固定训练配置、checkpoint 选择规则与预处理，再评估未参与优化的 test 52 行。同表 reserved 行预测与未见表泛化分别报告。

现有 `build_episode(partition='test', evaluation=True)` 支持把 train 支持与 heldout 行放在同一 episode，heldout 非 target 特征可见、target 真值仅 scorer 可见。这是可见 heldout predictors 的 transductive 协议；需说明模型能否利用其他 heldout 行的特征。MLP/XGBoost 也应具有相同的可见信息权限，实际使用方式可以不同。

训练预处理、类别映射和超参数选择不得用 heldout target。支持集不存在的 target 类、常数 target、超出支持范围的数值应作为显式状态报告，不能静默丢弃。现有 nominal 路径遇到无训练支持的 heldout 类会报 `no-answer-code`，应计入覆盖率。

当前 split 没有 validation：日常挑 checkpoint 不能反复挑 test 最优。可按训练拟合/预定步数选择 checkpoint；若需要调参，应在 train 内另定 validation，清楚说明这会改变训练数据规模。最后只对冻结方案报告 test，并说明此前已有的分布查看。

## 6. 固定评分与最小证据

正式 baseline runner 应直接复用正式 episode 构造与 scorer 的口径，而不是另写近似 mask。建议每次保存以下最小信息：

| 类别 | 必需记录 |
| --- | --- |
| 实验身份 | run/replicate、parent 或初始化来源、source/manifest/data SHA、split、target、codec、recipe、所有 seed、依赖版本 |
| Episode 身份 | probe 名、partition、mask index、原始 Query 地址、mask bank 摘要、visible 支持数、唯一地址数、总测量数 |
| 学习条件 | 可用于拟合的标签集合、是否有同表历史训练、预处理/类别策略、优化器、超参搜索与停止规则 |
| 输出证据 | 每地址每次预测、真值、误差、正确性、normalizer；持久化到实验结果目录，并与 checkpoint 绑定 |
| 汇总 | 相同地址/列/表聚合；current、historical best、final 分开；注明评估 update |
| 资源 | host/device/dtype、训练与评估耗时、updates/s、数据规模、峰值内存、可用时的算力预算 |
| 完成性 | terminal receipt、退出原因、final checkpoint/评估摘要、有限值检查、覆盖率与跳过原因 |

设唯一 Query 地址为 $i$，它在多个 mask 中共出现 $v_i$ 次，正式 accuracy 应先算每地址的平均正确率，再对地址平均：

$$
\mathrm{Acc}=\frac{1}{|U|}\sum_{i\in U}\frac{1}{v_i}\sum_{j=1}^{v_i}\mathbf{1}[\hat y_{ij}=y_i].
$$

回归 MSE 对平方误差作同样聚合；R² 使用唯一 Query 地址真值的总体方差，NMSE 使用每次 episode 可见支持 target 的方差。先平均误差，不是先平均预测后计算误差，也不是平均各 mask 的 R²。多 target 列先等权平均列，多表再等权平均表，同时始终保留逐表结果。

正式评分输出需先与已有常数参考在同一个 mask bank 上对齐，验证地址、权重、归一化一致，再加入模型预测。r1/r2 的 evaluation seed 不同，应分别重建相应基线；重复 mask 不是独立训练种子，不能把 8 个 masks 当作 8 次独立实验估计模型方差。

## 7. 监控与停止判断：待审阅建议

保留当前每 1,024 update 固定评估的主轴，训练日志另报近期 loss 和吞吐。监控面板的最小一行应包含：表、模型、replicate、实际 update/预算、最新已完成评估步数、Query 主指标、常数参考、历史最佳及其步数、耗时、进程/terminal 状态。日志 update 与评估 update 可能不同，应分别显示。

拟合诊断优先显示 decoded accuracy 或原始单位 MSE/R²，并辅以归一化/编码 loss。retained 指标独立显示，不能用 retained 接近满分掩盖 Query 不会恢复。分类补充逐类别 recall；回归补充误差分位数和尾部错误；需要解释局部恢复失败时，再看 kernel ESS/最大权重等已有诊断。

比较速度时，对同一张表分别记录纯训练、评估、编译/预热时间，以及达到相同质量门槛所需时间。当前不同主机跑不同表且精度不同，仅凭 updates/s 不能断定某设备或架构更快。CUDA/MPS 数值轨迹允许不同，当前探索更关注可用、快速、稳定达到拟合目标；仍记录 dtype/backend 以便解释差异。

停止判断同时考虑预算、固定曲线与数值稳定性。`040` 说明短期多数类平台不够支持提前宣判失败。比较最新、最好、终态及独立重复；若出现回退，再查梯度/裁剪、mask 难度、类别预测分布和支持覆盖。延长预算或改学习率属于下一项实验决策，不由监控记录自动触发。

## 8. 接入训练系统前的最小待办

- [ ] 用户确认 A 为主比较，B 为补充诊断，C 为后续阶段；分别命名，不合并成单一 leaderboard。
- [ ] 复用 evaluator 的命名 seed、地址聚合与 NMSE，建立持久化的 baseline 回执；重跑 MLP/XGBoost 后替换第 3 节的临时比较。
- [ ] 主拟合基线允许三者使用同一 train 标签集，明确 nominal 编码与预算；保存收敛曲线与达到质量门槛的时间。
- [ ] 读取并核验 r0/r1/r2 的实际 terminal、checkpoint 和固定结果，再给逐表重复统计；本文不把调度计划当完成结果。
- [ ] 在预定阶段冻结方案，补 reserved 报告；若根据 reserved 反复调参，将其视为 validation，并另留独立 final test。
- [ ] 决定将哪些最小字段接入现有 runner/报告入口，再实现；本文件本身不新增监控服务、自动任务或训练系统规范。

审阅重点：希望把“已知训练行最终能否拟合”“同等资源下拟合多快”“reserved 行预测如何”放在同一份报告中分开呈现，并让每个结论都能追溯到对应学习条件、固定评分和运行回执。
