# V7 行内 64→64→lift：实现与极短迁移对照

2026-10-08。已在本机 canonical 源码实现显式 `value_encoder="row_reversible64"`；默认 `phi_lift` 与昨天 `row_dual_stream` 保持原有身份和行为。提交基底为最新主线 `f6411375`，不重复导入昨天的 V7.3 实现。该分支尚未成为共享默认。

## 计算与迁移

路径为 `C → φ(C) → split(32,32) → 四块行内加性耦合 → concat(64) → 原 lift(64→128) → token dynamics`。LL 使用升维前的 64 维响应。所有 Query 的 LL 结果同步装回整行，先逆过四块耦合，再逐列经过 φ 的逆映射；只回写 Query。观测、结构 Null、源资格、跨轮梯度与 V7.3 数值边界沿用既有合同。

复用昨天的 Attention/FFN 增量和正逆耦合实现，未修改其 128 维复制双流路径。新模块宽度为每支 32，四头，FFN 32→64→32；Attention 输出投影及 FFN 末层为零初始化，因此初始行内变换精确为恒等。沿用旧路径的 active-cell lift 批形状及逐列 φ 逆回批形状，避免仅由批形状引入的浮点差异。

固定供体为七表 `phi_lift` 第 7580 步，SHA-256：`8e4de8a7af1fb597ff2caabac3f05f20750933c19401902e677a0c6c7fe416a9`。新分支复制 94 个兼容张量，包括 φ、lift、骨干及原 seeds；36 个新行内模块张量独立初始化。供体模型张量没有遗漏；优化器、RNG 与采样状态不继承。逐项清单、形状和供体身份记录于迁移回执，完整目标状态严格加载。昨天分支从同一供体复制 86 个兼容张量，新编码器按昨天的 Xavier 起点初始化。

## 验收

- 本机新分支 15 项测试通过，覆盖 CPU/FP64、真实 MPS/FP32 的正反向、扰动后任意行状态的双向逆、Null、行隔离、数值方向导数、Query-source 两种设置、隐藏真值隔离、优化更新及 checkpoint 严格重载。
- 最新主线检出上的新旧分支合计 **78/78** 通过，无跳过。补齐三份历史 runner 后，其16项测试与新分支15项测试的最终批次 **31/31** 通过；合计94个不同用例，不表示全仓测试已运行。新测试 fixture 沿用主线的 MPS determinism 隔离方式。
- 修改前后 V7、V7.3、昨天双流三条旧路径，CPU/FP64 与 MPS/FP32 共六组：配置、初始化权重、前向、损失和参数梯度逐位一致。
- 子朝 Mini 使用既有 `train` profile（Python 3.12.14 / Torch 2.13.0、MPS/FP32）。首批 77/78 通过，唯一失败是异常文本未匹配旧测试；恢复兼容文本后，对应 8 项参数化测试全部通过。没有安装或替换训练环境，测试使用该机已有的隔离测试依赖。
- 真实供体迁移后，新分支在三个固定样本的四轮输出与旧 φ+lift 模型逐位一致；两个对比分支的 86 个骨干/seed 张量逐位一致。双方第 0、15 步均完成原生 checkpoint 严格重载及输出重放。

## 已停止的极短对照

用户在执行期间要求缩短，未跑满原定每分支 30 步。进程停止前日志记录新分支 21 步、昨天分支 20 步；**报告仅使用双方已保存且已评估的第 15 步**。第 15 步之后的更新没有 checkpoint，不作为可恢复终点，也不重新启动补跑。

同一供体、MPS/FP32、完整供体骨干/K4、fresh AdamW、学习率 `3e-5`、梯度裁剪 1；三个既有合成任务各 5 步，双方完全相同的训练行、Query mask、code/donor seed 与更新顺序。每个任务固定 64 个训练 Support、24 个历史同表留出 Query；不读取留出标签建立支持类目。它是迁移起点的工程对照，不是独立测试、未见表评估或充分训练后的架构比较。

| 指标 | 今天 64→64→lift：起点→15步 | 昨天复制双流：起点→15步 |
|---|---:|---:|
| 三表平均加权 code loss | 0.03587 → 0.02150 | 4.18475 → 3.00717 |
| numeric `discoscm_048` R² | 0.9482 → 0.9677 | −80.5741 → −4.5782 |
| nominal `discoscm_049` accuracy | 0.9167 → 0.9167 | 0.6250 → 0.6250 |
| ordinal `discoscm_051` accuracy | 0.7917 → 0.7917 | 0.2500 → 0.2500 |
| 前15步最大裁剪前梯度范数 | 23.15 | 4.53×10⁹ |

双方这15步的 loss/梯度均为有限值。新分支的迁移起点较平稳，支持保留“恒等行内模块＋原 φ/lift”的初始化优化；新分支仍触发梯度裁剪，极短试验不能证明长期稳定性。两个分支可继承的权重和初始化不同，因此本结果不能把收益单独归因于 64 维结构，也不比较昨天已经训练数小时的 checkpoint。12 组保存的原值预测/参考指标已独立复算。

## 使用与证据位置

```python
import torch
from tabu_lab.models.restoration_v7 import (
    V7Config, RowReversible64Config, row_reversible64_from_checkpoint,
)
config = V7Config.v73(
    value_encoder="row_reversible64",
    row_reversible64=RowReversible64Config(),
)
# Existing phi_lift checkpoint: explicit weights-only migration, fresh optimizer.
model = row_reversible64_from_checkpoint(
    donor_path, expected_sha256=donor_sha256, device="mps", dtype=torch.float32,
)
```

完整本地实验材料位于项目 `preparations/v73-row-reversible64-impl-20261008/`；不把私有数据、权重、采样行或训练产物纳入源码提交。版本化验收摘要见同目录 `verification.json`。仅修改源码的默认入口不等于采用新的共享 checkpoint。

历史 runner 的最小补齐范围见 [记录](HISTORICAL-RUNNERS.md)。
