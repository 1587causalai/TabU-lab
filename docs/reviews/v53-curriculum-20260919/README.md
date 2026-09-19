# V5.3 curriculum runner 验证记录

日期：2026-09-19。工作分支：`codex/restoration-v53`。
首次验证时 HEAD 为 `6d078b2359e484dbf96ce7a5ca6744f0f83184e3`，curriculum
新增代码当时尚未提交；重建需使用包含本记录和 runner 的后续提交，不能仅使用该基线。
实际执行源码身份为
`17cedb6a5058be87444c625a33b382dfe3661bd5765a46b32a3b95410030b12d`，
完整逐文件 SHA 位于验证产物的 `resolved.json` / `terminal.json`。

结论：单进程 FP64 课程的配置、数据隔离、固定评估、预算、恢复和报告已通过本地 CPU
检查，可以作为下一轮有预算研究实验的执行基础。GPU、大表内存、分布式、混合精度和
长程训练资格尚未取得。训练效果和课程收益需要独立实验，运行完成不表示质量达标。

## 这次固定下来的合同

- 阶段必须登记研究问题、表组、采样权重、recipe、优化器和 updates/seconds 双预算；
  单表拟合、共享拟合、同表继续优化、新表学习和监督适应分别解释。
- 六路 seed、数据原始字节及分割、解析后配置、源码共同定义实验身份。表数没有硬编码。
- train/validation/test 隔离；只有 validation probe 可以控制 gate。final_test 不进入阶段，
  显式独立冻结评估。历史已观察测试集只能声明 retrospective。
- 固定命名 probe 比较相同 mask/code 问题；逐地址、逐列、逐表统计后等权汇总，分开
  numeric/discrete 与 Query/retained，同时保留原始地址曝光和 Query kernel ESS。
- strict resume 继承 optimizer、RNG、cursor、累计耗时和实际曝光；weights-only 初始化
  另建实验身份并记录父 checkpoint。AdamW→Muon 是显式协议变更。
- checkpoint 通过不可变内容地址代际及原子相对符号链接提交。失败不能覆盖上一有效
  checkpoint；恢复从原生代际路径也读取所属 attempt 终态的已消耗时间。
- 自动中文 `report.md` 汇总实际阶段 verdict、同名 probe 起止、表曝光和结束原因。

V5.3 先前审查的两项缺陷也已修复并纳入回归：共享 LL 用逐中心中心化矩避免二阶矩相减
丢失信号；公共 `forward` 在 `torch.inference_mode()` 下可建立带版本保护的 owned snapshot。
稳定 LL 的计算与反传内存成本仍需实测，中心分块不等于反传内存已经有界。

## 实际验证

在本 checkout 设置 `PYTHONPATH=src`，使用现有 `tabu-lab/.venv/bin/python`：

```bash
python -m pytest -q tests/unit/curriculum_v53 tests/unit/restoration_v53 tests/unit/restoration tests/unit/test_program_cli.py tests/unit/test_tabur_cli.py
python -m ruff check src/tabu_lab/curriculum_v53 src/tabu_lab/cli.py tests/unit/curriculum_v53 examples/curriculum_v53_fixture.py
```

- **504 passed，18.46 秒**；Ruff 通过，`git diff --check` 通过。这是上述指定范围，未声称全仓测试。
- 四个中断断点覆盖待执行 periodic/final probe、阶段切换、AdamW→Muon 初始化，以及
  Muon 运行中恢复。最终 model、完整 optimizer state、RNG、计数与曝光逐项一致。
- 故障注入覆盖 sidecar 写入前/后失败、非有限更新、最终评估超时、最终保存超时后的
  completed cursor 恢复、preflight 最后探针超时、probe-only 表排除及中断回执。
- CLI 以独立进程实跑 `plan → preflight → run`：三阶段 4+3+3，共 10 updates 完成。
  另一次第 5 update 停止后恢复到 10，model/optimizer/RNG/exposure 与完整运行完全一致。
- 独立 `evaluate --probe final_test` 完成，checkpoint 不变。另用
  `regression_width=null` 的全宽共享 LL 做小表 CPU preflight，下一更新恢复一致性通过。

演示全部使用微型合成表，每表 12 行，train/validation/test 为 8/2/2。
主要流程 fixture 使用显式 `regression_width=4`；单独全宽检查仍只是小表资格，
不能外推到真实 120 表或大模型。示例无性能 gate，validation 也并未因运行完成而被标为改善。

可重建模板与完整命令见 [课程教程](../../tutorials/v53-curriculum.md)。
本地完整产物保留在仓库根 `results/raw/v53-curriculum-20260919-verification/`，其中：

- `verification.json`：命令、实验身份、实际结束状态及精确恢复比较。
- `uninterrupted/report.md`、`resumed/report.md`：完整与恢复课程报告。
- `final-test/report.md`：独立冻结评价报告。
- `full-width-preflight/terminal.json`：全宽小表资格回执。

## 扩展实验前的具体工作

1. 将历史 8/120/12 表的冻结文件、SHA、已观察分割和实验关系录入新 manifest；旧 TAR
   checkpoint 不直接进入 V5.3。配套 scratch、joint、sequential 对照，匹配预算及数据曝光。
2. 对目标设备及实际最大行数、列数、全宽回归配置执行 `preflight`，记录峰值显存与耗时。
   本版使用单进程 CPU 或 `cuda:0` FP64；未实现 DDP/FSDP、混合精度及 activation checkpointing。
3. 正式预算同时考虑固定 probe、checkpoint IO 与存储。时间限制在完整步骤和探针边界检查，
   不会中途杀死一个 kernel。checkpoint 代际保留且不自动清理，需按模型大小与保存频率预留磁盘。

首次实现验证未启动历史数据长程训练、GPU 作业、发布或合并。

## 主分支合并验证

同日，owner 明确授权合并主分支并推送远程。runner 提交为
`3266d1b28c22bf8d3f8cf9dbb4aced2f75cd5f62`，合并目标原为
`465398685207d4c33a59eb2c8d0ab6e0f0cad0ec`；保留目标分支的 TAR repository snapshot
测试修正和设计文档订正。自动合并无冲突，合并后的执行源码 SHA 与上文完全一致。

合并工作树全仓回归：**1067 passed，81.88 秒**。两条 warning 来自已有的 W&B 缺失/失败
降级行为测试。初次全仓运行因环境缺少 `scikit-learn` 出现两个 classical baseline 失败；
补齐临时目录中的可选依赖及其依赖后通过，未改变项目环境或通过删改测试规避失败。

```bash
PYTHONPATH=src:/private/tmp/tabu-v53-merge-test-deps-20260919 .venv/bin/python -m pytest -q
```

临时测试依赖：scikit-learn 1.9.1、joblib 1.6.0、threadpoolctl 3.7.0、cloudpickle 3.1.2、
narwhals 2.26.0。Ruff 和 staged whitespace 检查通过；独立发布审查未发现阻塞项，
提交范围不包含 checkpoint、实验原始产物或凭据。
