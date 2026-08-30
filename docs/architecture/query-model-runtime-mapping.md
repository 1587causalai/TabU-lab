# Query-model runtime mapping

状态：Checkpoint 0（math → runtime mapping）。

本文件是 Axis C `table-cell-as-query-models` 数学合同到 `tabu-lab` runtime 的实现桥梁，不是新的数学 authority，也不是训练或 evidence 结果。语义 authority 是只读的
`latex/model-factory/table-cell-as-query-models/`，首个 concrete 合同是
`TabUBase/main.tex` 中的 `tabu.query.base@0.1.0`。

## 1. 身份边界

Axis C 不是 Axis B `table-cell-as-unit-models` 的改名。旧 runtime 的
`tabu.cell.base@0.2.0`、ModelSpec bytes、component IDs、trace identity、checkpoint、receipt
和实验结论都不能被新模型继承。低层 tensor operator 可以复用，但复用必须经过 query-specific
composition/adapter，并产生新的 source、factory 和 composition identity。

| 数学角色 | QueryBase runtime 角色 | 不允许的重解释 |
| --- | --- | --- |
| cell $\omega=(r,a)$ | semantic query object；每个 cell 都可成为 query | 不能把 cell 写成 semantic Unit |
| row / column | 两根抽象 axis role；Base 都是 `HOMOGENEOUS` | 不能在 Base 声称任务级 row-Unit/column-Unit 语义 |
| $W$ | shared response mechanism；Base 的 global fallback，$z=Wc$ | 不是 Unit，也不是 Unit token |
| column axis | 计算轴角色 | 不能把 column unit 当作 feature/entity identity |
| `query_target_mask` | supervised label-query 的 raw origin marker | 不能用它表达“全部 cell 都是 query” |
| truth | `TruthSidecar` 中的 loss-only target | 不能进入 tokenizer、source plan、dynamics 或 readout |

## 2. 五步合同映射

| 数学 Step | QueryBase 输入/输出 | 可复用 runtime | 当前缺口 |
| --- | --- | --- | --- |
| 1. typed query cells | `EvidenceEpisode` 编译为 value、raw role、mask、query、null 的无真值输入 | `EpisodeCompiler`、`DenseModelInput`、context-only statistics、`Symbolizer` | 需要 query-specific trace 字段，不能沿用 `unit=cell` |
| 2. tokenized queries | 共享 Fourier/ordinal/episode-local nominal token；query marker 不含 $y$；null 为 exact zero | `CellTokenizer` 的数值、类别、mask/null 算子 | 旧 tokenizer 的 class/interface 名称带有 Axis B 语义，必须由新 adapter 绑定 |
| 3. token dynamics | column source 只取 context-visible cells；row source 保留 query-row visible Feature；query/mask/null 为 receiver-only；可选 label-column broadcast | `InducedCarrierBlock`、OMAB/MAB、label-broadcast operator | source topology 必须成为 typed、hash-bound `AxisSourcePlan`，不能藏在 orchestration 分支 |
| 4. readout | $c_{ra}=h^{(L)}_{ra}$，Base 使用 $z_{ra}=Wc_{ra}$；same-feature/same-label support；typed LL/NW terminal | `PairUnitReadout` 的低层 routing/terminal 逻辑、`PredictionBundle`、trace 基础设施 | 新 readout/geometry identity 不能使用旧 `tabu.cell-*` interface 或 `unit=cell` metadata |
| 5. loss | prediction 与 `TruthSidecar` 配对；固定 input-side evidence 后替换 truth 不改变 Steps 1–4 | 现有 evaluator/loss-sidecar boundary | 首批只验证边界，不运行训练 |

## 3. Base 的 family plan

首个 query runtime 的 typed plan 固定如下：

```text
cell_role       = query
row_axis.mode   = HOMOGENEOUS
column_axis.mode= HOMOGENEOUS
geometry        = GLOBAL_W
response        = z = W c
```

`W` 是两根 homogeneous axis 的全局 fallback；未打开的 axis 不单独生成坐标或私有 token。
未来 family probe 只允许以下生成元：

| Probe | row axis | column axis | geometry |
| --- | --- | --- | --- |
| Base | H | H | global $Wc$ |
| R | X | H | row heterogeneous geometry |
| C | H | X | column heterogeneous geometry |
| RC | X | X | concat$(z_R,z_C)$ |

这里 X 只表示显式打开 heterogeneous axis；它不是本批已注册的 concrete ModelSpec。

## 4. Source topology 与语义标记

编译后的 source policy 必须可逐格检查：

| cell 状态 | column axis K/V | row axis K/V |
| --- | --- | --- |
| context-visible cell | 是 | 是 |
| query-row visible Feature | 否 | 是 |
| query label marker | 否 | 否 |
| artificial mask | 否 | 否 |
| null / natural missing | 否 | 否 |

`query_target_mask` 保留现有兼容字段；runtime 内部和 trace 使用明确的 supervised-target 语义，
不得新增一个含义混杂的 `is_query` 布尔值。`e_query` 不含目标 truth，natural missing、mask、null
也不能被当作 feature identity。

## 5. 可复用与不可继承清单

可以复用：

- `DenseReferenceModel` 的 public `EvidenceEpisode → PredictionBundle` boundary；
- `EvidenceEpisode`、`TruthSidecar`、split-before-compile 和 truth-isolation；
- `DenseModelInput` 的 truth-free transport；
- Symbolizer、OMAB/MAB、typed numeric/categorical terminal 的低层算子；
- 旧 tokenizer/readout 的数值行为，前提是经新 query adapter 重新绑定身份。

不可继承：

- `TabUCellBaseModel` 的 class/model identity；
- `tabu.cell.base@0.2.0` ModelSpec、builder、checkpoint/variant namespace；
- `TabUBaseComponentManifest`、`tabu.cell-*` component authority 和 `unit=cell` trace；
- 旧 Axis B receipt、local evidence 或性能结果；
- 把 Axis B 的 parameter-token 叙述直接转写成 Axis C 的 heterogeneous Unit token 叙述。

## 6. Checkpoint 0 gap list

1. 当前 registry 只表达 `tokenizer/dynamics/readout` 三类 Axis B 组件，不能表达 axis mode、source topology、geometry 和 query role。
2. 当前模型 constructor/trace/spec 把 cell 写成 Unit，必须新增独立 query contract，不改写旧 contract。
3. 当前 source mask 逻辑散落在模型 orchestration 中，必须提升为可验证、可 hash 的 typed plan。
4. 当前 `TabUBaseProfile` 是 Axis B 类型；profile 字符串可复用，但 typed binding 必须独立。
5. 当前 verification 结果是单一 pass/fail，无法区分 harness、run、evidence 和 claim；query ladder 需要四维状态。
6. Base/R/C/RC 的 family growth 目前没有可组合的 executable seam；Base 先用 test-only probes 证明生长关系，
   R/C/RC 的合同可以登记，但在 carrier/token/source policy 未冻结前保持 `design_open`，不提供 builder。

## 7. 本 checkpoint 的停止条件

本文件通过路径、角色、source table 和 gap review 后，才进入 Checkpoint 1。Checkpoint 0 不创建
模型、checkpoint、训练 run 或 formal receipt；任何本地验证结果都只能标为 `local_unissued`。

## 8. 实施状态（2026-08-30）

- Checkpoint 0：mapping 已冻结；Axis C 与 Axis B 身份边界保持分离。
- Checkpoint 1：`tabu.query.base@0.1.0`、`QueryFamilyModelBase`、`TabUQueryBaseModel`、typed
  component manifest/registry、public builder 和 query-specific trace/checkpoint identity 已实现。
- Checkpoint 2：Base/R/C/RC 以 test-only `QueryFamilyPlan` probes 验证；只允许目标 axis 变化，未打开的
  homogeneous axis 不产生坐标或 token；匿名/旧 Axis B component 不能伪装为 query component。
- Query family registration：`tabu.query.row@0.1.0` 已作为首个可构建的异构 row family
  （TabUR）登记；`tabu.query.column@0.1.0`、`tabu.query.row_column@0.1.0` 仍是独立
  `design_open` ModelSpec，不继承 Base 的 checkpoint、receipt 或 claim。
- 评估阶梯 1–6 的 bounded harness 均已实现并通过本地 CPU 回归：每一层为
  `implemented / passed / local_unissued / none`。这包括 F0/S1 合成拟合、scratch-only
  真实数据诊断、无 optimizer 的 frozen ICL，以及同一 supervised profile 的 paired
  synthetic-pretrain → real fine-tuning lift。它们仍不是 formal receipt、benchmark 或
  accepted capability claim。
- 运行入口为 `scripts/run_tabur_evaluation_ladder.py`；`--device cpu|mps|cuda|auto` 会绑定
  到同一份协议。`dgx2` CUDA 环境已确认可用，但长跑须避开当前正在运行的 SGLang 服务。
