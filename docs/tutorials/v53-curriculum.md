# V5.3 可复用课程实验

这条路径把表、阶段、采样比例、优化器转换、固定评估和预算写入一个 manifest。
表数由 manifest 决定，没有固定的 8 / 120 / 12 表执行分支。实现使用独立的
`restoration_v53` 模型身份；legacy TAR 与以前 restoration 的结果保留各自含义。

当前范围是本地、单进程的 FP64 参考训练与恢复流程。小预算 CPU 验证不表示
GPU、大表、大模型吞吐或分布式训练已经合格。`regression_width=4` 的演示是显式降维配方，
与默认全宽共享 LL 的性能不能直接比较。

## 先固定每阶段回答的问题

| 阶段 | 可以回答 | 还不能回答 |
|---|---|---|
| 单表分别拟合 | 架构在每种固定表上的基本拟合能力 | 同一参数能否同时容纳这些表 |
| 多表共享拟合 | 固定训练表是否共同可训练 | 未见行、未见表泛化 |
| 相同表继续训练 | 在同一配方下增加优化预算的影响 | 新任务学习和遗忘 |
| 新表持续学习 | 新表适应与旧表保留的变化 | 没有匹配对照时的预训练增益 |
| 已知表监督适应 | train support 到 target Query 的适应 | 未见真实表能力或课程顺序收益 |

历史 8 表独立拟合与 8 表共享拟合是两组诊断。历史 old120 从随机初始化开始，
没有承接 8 表权重；old120 延长训练和 recent120 新表持续学习也应分开。
历史 recent120 同时改变过优化器，真实 12 表经历过 old3、新9、mixed replay 等阶段。
这些经历可以贡献协议和负结果，不能整体视为已证明有效的单一课程。
相关证据保留在 `archive/tar-unified-synthetic-nearzero-20260907/`、
`archive/tar-unified-joint-fit-20260907/`、`archive/tar-diverse-120-fit-20260907/` 和
`archive/tar-new120-muon-20260908/`；目录位于 TabU 项目根，而非此代码仓库内部。

## 一个可以完整检查的小例子

在当前代码仓库根目录，用包含 PyTorch 和项目依赖的 Python 环境执行。
下面的 `PYTHONPATH=src python -m tabu_lab.cli` 与当前源码安装出的 `tabu-lab` CLI 等价；
它明确从本 checkout 导入，避免误用其他 worktree 的可编辑安装。

```bash
python examples/curriculum_v53_fixture.py --output-root /tmp/v53-curriculum-data
PYTHONPATH=src python -m tabu_lab.cli curriculum-v53 plan --manifest /tmp/v53-curriculum-data/manifest.json
PYTHONPATH=src python -m tabu_lab.cli curriculum-v53 preflight --manifest /tmp/v53-curriculum-data/manifest.json --output-root /tmp/v53-curriculum-preflight --device cpu
PYTHONPATH=src python -m tabu_lab.cli curriculum-v53 run --manifest /tmp/v53-curriculum-data/manifest.json --output-root /tmp/v53-curriculum-run --device cpu
```

所有输出目录都应是新的目录；重跑时换一个路径，保留前一次产物。
fixture 生成三张各 12 行的混合类型合成表，每表固定 train 8 / validation 2 / test 2。
`old_a`、`old_b` 组成旧表组，`new_a` 组成新表组。三张表都不是现实数据；
`new_a` 的 `kind=real` 只演示按 kind 分派 recipe，manifest 和原始表文件都标注了模拟来源。

演示依次执行共享拟合 4 updates、新表持续学习 3 updates、同一新表监督适应 3 updates，
全部使用 AdamW。默认不配置质量 gate，10 次更新只检查流程，不设人为拟合通过线。
每阶段 `question` 说明所问问题，`max_updates` 与 `max_seconds` 同时约束预算。
`preflight` 的数值检查仍不能替代正式实验；以实际命令产物为证，不预填测试数字。

最终 test 必须单独指定冻结 checkpoint 与 probe，写入另一目录：

```bash
PYTHONPATH=src python -m tabu_lab.cli curriculum-v53 evaluate --manifest /tmp/v53-curriculum-data/manifest.json --checkpoint /tmp/v53-curriculum-run/checkpoint-progress.pt --probe final_test --output-root /tmp/v53-curriculum-final-test --device cpu
```

可重复传 `--probe` 选择多个已登记探针。独立评价不执行 optimizer update。

## Manifest 的关键关系

顶层 `schema` 为 `tabu.curriculum.v53.v1`，包含 `experiment_id`、六个独立 seed stream
（`model/order/masks/codes/windows/evaluation`）、`model`、`optimizer`、`tables`、
`probes` 与 `stages`。JSON 和 YAML 均可；以 fixture 生成的完整 JSON 为可运行模板。
未知字段会报错，不能靠未解析的备注字段改变执行。

默认 `model.codec_version=unit_gaussian_v2`、`numeric_scaling=zscore`；组合码设为
`constant_weight_v1`。两者的 ordinal 都是类别身份 $q_{a,c}$ 加归一化秩方向
$r_a(c)b_a$，按完整声明域做最近码解码。上一版 `unit_gaussian_v1` 保留共享基点方案，
更早的 `legacy_v53` 也只作为显式候选；不能把旧 checkpoint 按新 codec 严格续训。

阶段 `loss.state_weights` 的顺序为 retained、Query、Null、corrupted，默认 `[0,1,0,0]`。
要加入 restore loss，设 `[lambda_restore,1,0,0]`；显式 `null` 才采用历史混合均值。
预检仍要求全部原始真值地址和合法支持，不因 retained 系数为零而缩小契约。

每张表登记 `id/path/sha256/cohort/kind`，并可指定 `role`、`target_column`、`window_rows`。
`path` 相对于 manifest 目录，`sha256` 绑定文件原始字节。数据沿用已有
`values[N,M] + features + splits` JSON；每列类型显式声明，离散值是 declared-domain 索引，
ordinal 可显式声明 `order`。train、validation、test 必须唯一、互斥并覆盖所有原始行。
train 至少三行；`role=probe` 的表不能进入阶段采样。

一个阶段的 `sampling` 按 cohort 登记 `episodes`，这是每个调度周期的更新数权重，
不是 Query 单元格数或算力权重。cohort 内各表按固定顺序轮换。`recipe` 按实际使用的
`synthetic` / `real` kind 选择 `random_cell` 或 `supervised_row`；监督目标来自每表
`target_column`，默认为末列。表类别标签不提供真实数据来源证明。

训练只使用 train 张量。随机 cell mask 保留每列至少两个支持值，numeric 还须至少两个
不同值；nominal 保留各可见类别代表。新 ordinal 码由完整 schema 建立，因此允许 Query
等级未在支持中出现；监督行 mask 使用同样的类型边界。无法满足精确Query预算时整个episode报错，不静默降低预算、
删掉难例或收窄loss。`numeric_query_guard` 默认为显式 `{"kind":"none"}`；如要排除numeric
尾部Query，需把已有guard写入配方，并报告被保护数量。这会改变学习问题。

`window_rows` 只对train窗口生效。固定seed可复现窗口，但若只评估少数mask，不能声称
覆盖所有训练行；实际 `row_ids/query_addresses` 和曝光统计才是覆盖证据。
本版validation/test使用完整train support加完整对应holdout，不支持holdout窗口抽样，
因此有reserved probe的表必须让 `window_rows` 为null。

## 固定探针与决策边界

`probe` 声明 `name/cohorts/partition/purpose/recipe/masks`。同一个命名probe的表地址、
mask与code实现固定，跨阶段比较同一评估问题；不同训练步骤不重抽一个更容易的评估集。
numeric、discrete、Query、retained分开报告，同时保留逐表数据与曝光，避免总loss掩盖退化。

- `partition=train` 的 `fit/retention/transfer` 仍评价登记的训练池；目的名称不会自动赋予泛化含义。
- `partition=validation` 的 probe 才能控制阶段 gate。gate 指定已列入本阶段的 probe、
  `metric`、`mode=min|max` 和预注册 `threshold`；支持 `query_numeric_mse`、
  `query_numeric_normalized_mse`、`query_discrete_accuracy`。阈值应来自任务标准与基线，
  不从本轮test分数倒推。
- `purpose=final_test` 只能用于独立 `evaluate`，不能列入 `stage.probes`。已经反复查看的
  历史保留集如用于训练期间比较，须显式写成 `purpose=retrospective`，不能再叫未接触测试。

validation/test只允许 `supervised_row`、`masks=1`：heldout目标列完全隐藏，真值只交scorer；
heldout其他特征可见，所以这是显式transductive协议。所有模型统计仍从forward-visible
证据求取。未被 train 支持的 nominal 目标类别会整次评价失败；新 ordinal 的声明等级仍可评分。
可评分不代表模型已学会恢复这些等级，不能跳过难例后再报告平均分。

## 恢复、初始化与优化器转换

用一次短暂停检查恢复路径，不需要修改正式预算：

```bash
PYTHONPATH=src python -m tabu_lab.cli curriculum-v53 run --manifest /tmp/v53-curriculum-data/manifest.json --output-root /tmp/v53-curriculum-part1 --device cpu --max-updates-this-invocation 2
PYTHONPATH=src python -m tabu_lab.cli curriculum-v53 run --manifest /tmp/v53-curriculum-data/manifest.json --output-root /tmp/v53-curriculum-part2 --device cpu --resume-checkpoint /tmp/v53-curriculum-part1/checkpoint-progress.pt
```

`--resume-checkpoint` 是严格续训：核对数据、源码、model、配方和完整checkpoint身份，
继承optimizer、RNG、阶段cursor、累计更新数与预算状态。不要改动manifest后把它叫resume。
`--initialize-from` 是显式weights-only派生实验：使用兼容模型的父权重，重新建立本次调度和
optimizer状态，记录父checkpoint；它不与strict resume等价，也不会自动证明预训练增益。
这两个入口互斥，均使用新的输出目录。

阶段字段 `optimizer` 可以显式写 `adamw` 或 `muon`。例如在另一个新manifest中，把
`fit.optimizer` 保持 `adamw`，将 `continual.optimizer` 与后续 `adapt.optimizer` 都设成
`muon`，就是预注册AdamW→Muon转换。转换按现有混合优化器的参数分组规则执行，
其余参数保留AdamW；不支持Muon→AdamW反向转换。Muon需要当前环境提供对应PyTorch API；
小fixture默认全AdamW以减少流程演示的变量。换优化器与换数据同时发生时，结果无法单独归因。

## 产物和下一次正式实验

`resolved.json`保存解析后的配置和身份，`updates.jsonl`保存成功更新与实际曝光，
checkpoint及其校验文件保留可恢复状态，`terminal.json`记录completed/stopped/failed等结束原因。
`report.md`是这些记录的可读投影：并列阶段问题、预算与实际verdict，比较同名固定probe的
首末指标，并列出按表训练曝光、结束原因和原始记录链接。报告只读取此次attempt的评价文件；
strict resume继承的累计更新与曝光会明确标为累计链，不补写父attempt的起始评价。
numeric normalized MSE 采用 episode-visible codec 尺度，报告 $(\hat z-z)^2$，
不包含组合码训练损失的方向范数因子 4；也不能直接替换历史固定方差 NMSE。

checkpoint以不可变代际存放在`checkpoints/<sha256>.pt`，`checkpoint-progress.pt`和阶段末
checkpoint是原子切换的相对符号链接。搬移实验时携带整个attempt目录，保留`checkpoints/`
及相对链接关系；只复制`.pt`指针或只复制JSON侧车会丢失可恢复状态。侧车用于查询，
实际载入仍核验已提交代际的内容地址和内部身份。

失败产物和最后有效checkpoint应保留；有限forward也不代表梯度与optimizer状态安全。
进程结束、局部smoke通过、完整拟合、泛化结论是不同证据层次。

迁移到历史表时，新建一个实验manifest，引用已冻结JSON及其实际SHA，不改历史数据、
注册表或checkpoint。先明确哪些旧reserved已被观察，再决定验证与确认集合。把原来
“先8张、再120张、再12张”的叙述拆成上述问题；登记准确表集合和预算后才启动。
缺失的冻结数据先找齐并核验，不凭名字重造一批数据却继续使用旧实验身份。

要解释课程或预训练的收益，还需要相同数据曝光/更新预算下的scratch、联合训练或顺序对照，
并固定codec、readout、优化器等比较条件。这些对照应成为各自可重放的manifest与结果。
