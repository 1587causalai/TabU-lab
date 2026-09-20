# 模型无关课程接口：先绑定证据，再选择探针

[独立设计规范](../design/pretraining-curriculum.tex)（[PDF](../design/pretraining-curriculum.pdf)）
解释三层边界：不变的语义合同、可修订的经验默认、可替换的实现。
这里说明首个最小实现 `tabu_lab.curriculum` 的实际调用范围。

核心只做冻结数据绑定、分离校验、探针选择与 episode 请求。
`select_stage` 支持任意命名、表数和顺序；`reference_stage` 提供当前
fit8 / multi120 / continual120 / real12 默认，并允许显式覆盖。
它们不启动训练、不加载 checkpoint，也不安装自动质量 gate。
默认四阶段是证据阶梯，不能用前一层通过替代后一层验证。

## 复用原位 old120 / new120

从 `tabu-lab` 根目录执行，只读检查 manifest 和数据：

```bash
PYTHONPATH=src .venv/bin/python examples/curriculum_catalog.py \
  --old-corpus experiments/local/tar-diverse-120-fit/corpus/manifest.json \
  --new-corpus ../archive/tar-new120-muon-20260908/source/experiments/local/tar-new120-muon/corpus/manifest.json
```

这是本地历史路径，不是新发布的数据集。其他环境应传实际原位 manifest 路径。
`bind_legacy_corpus` 核对各文件的原 SHA，并保存来源 manifest SHA、record、
生成 seed、split seed、可解析 world ID；不会复制或补造数据。
old/new 复用相同历史表名，因此消费 ID 加 cohort 前缀，来源 ID 保持原样。

`level="table"` 拒绝原字节或 values 内容重叠，以及已知 world 冲突。
values 指纹忽略行顺序、split、schema 和表名，保留列顺序与重复行数量，
因而会保守拒绝相同 values、不同 schema；不保证识别列置换或任意变换。
`level="world"` 还要求每张表都拥有能从已支持 manifest record 核实的 world ID。
不同 seed、任意自填 ID 或一般来源引用不能完成这项认证。

本轮本地核验的 240 个 SHA 均匹配，old/new 内容分离通过。
当前 importer 每组只提取 48 个 `realized.world_hash`，其余 144 个记录不能通过
此 importer 的完整 world 分离认证。其他历史机制证据应单独适配，不能填字符串绕过。

## 组合默认与自定义探针

```python
from tabu_lab.curriculum import bind_legacy_corpus, reference_stage, select_stage

old = bind_legacy_corpus("path/to/old/corpus/manifest.json", cohort="old")
new = bind_legacy_corpus("path/to/new/corpus/manifest.json", cohort="new")

joint = reference_stage("multi120", old)
continual = reference_stage("continual120", new, previous=old)
probe = select_stage(
    "two-table-capacity", old[:2], question="Can this mechanism fit these two tables?",
    fit_mode="independent",
)
revised = reference_stage(
    "fit8", old[:2], name="fit2-control", expected_count=2, fit_mode="independent",
)
selection_record = probe.manifest()  # planned metadata, never a training verdict
```

这里的二表是新诊断，不继承历史 fit8 的身份。`previous` 是显式分离/保留参照，
不自动传递权重；真正 continual 实验仍必须在 execution manifest 绑定父 checkpoint。
同一表的重复曝光应在采样权重里记录，不用复制别名增加独立表数。
`BoundTable` 只有 metadata，没有 raw values；`manifest()` 输出解析后的政策和内容身份，
不嵌入本地路径或原始真值。数据路径仍由调用方以原 manifest 解析，它不是独立执行配置。

## 可调用 episode，适配器显式选择

```python
from tabu_lab.curriculum import EpisodeRequest, EpisodeSeeds
from tabu_lab.curriculum.v53_adapter import V53EpisodeFactory

factory = V53EpisodeFactory(device="cpu")
request = EpisodeRequest(
    seeds=EpisodeSeeds(masks=17, codes=23, windows=31, evaluation=41),
    index=0, kind="random_cell", fraction=0.25,
)
inputs, targets, truth, audit = factory(old[0], request)
# Only inputs and targets may be passed to forward; truth belongs to the scorer.

reserved = EpisodeRequest(
    seeds=request.seeds, kind="supervised_row", partition="test",
    purpose="retrospective",  # these historical reserved rows have been observed
)
reserved_inputs, reserved_targets, reserved_truth, reserved_audit = factory(old[0], reserved)
```

该 bridge 惰性调用现有 V5.3 `load_table/build_episode`，不新写随机 sampler 或 codec。
训练 episode 只有 train 行；reserved 请求须显式指定用途，并使用完整 train context、
隐藏的 heldout 目标与可见的 heldout 非目标特征（transductive）。
`final_test` 是调用者对使用经历的声明，接口无法证明该数据此前无人看过。
只有新确认集可以据此获得确认用途；历史已查看 reserved 必须保留 retrospective 解释。

episode audit 记录完整请求、原始地址、四路实际种子、数据/split 和源码文件摘要。
model/order 两路种子、模型/优化器配置、依赖环境、父 checkpoint 与预算属于外层执行合同。
episode audit 不是完整训练回执，完整重放仍需这些执行信息。
当前 adapter 只支持 complete typed-fit JSON 和无 tail guard 的两种现有 recipe；
这不是对未来数据、missingness、损坏方式或 factory 的通用限制。

## 与现有执行器的关系

继续通过 [V5.3 runner](v53-curriculum.md) 执行已有 V5.3 实验，保留它的
`resolved.json / updates.jsonl / terminal.json / report.md` 与严格恢复身份。
新包没有改动 runner 或给它偷偷加 gate；在使用历史表构造新 runner manifest 前，
可先调用此接口校验选择和分离。尚未实现跨模型通用 runner、完整新结果 schema、
自动 stage advance 或 TFM-Data 持续供给接入。

检验范围：metadata/episode 契约与现有 data/protocol 回归，没有启动训练实验。
测试命令：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/unit/test_curriculum_catalog.py \
  tests/unit/curriculum_v53/test_data.py tests/unit/curriculum_v53/test_protocol.py
```
