# 课程包证据索引

核查日期：2026-09-19。以下是来源和结论边界，非新的训练收据；历史状态不当作当前运行状态。

| ID | 来源 | 证据与边界 |
|---|---|---|
| E1 | [V5.3 教程](../../tutorials/v53-curriculum.md) | manifest、固定探针、resume、final-test 隔离合同；示例运行不证明模型效果 |
| E2 | [模型无关课程](../../tutorials/model-independent-curriculum.md)及[设计规范](../../design/pretraining-curriculum.tex) | 数据冻结与 episode 接口；world 来源不完整时不可补造身份 |
| E3 | [历史 TAR 共享拟合](../../research/tar-shared-fit-20260907/README.md) | old120 768 rounds/92,160 updates；57 数值表 median NMSE 0.038238、63 离散表 median accuracy 99.265%；训练行证据，属于 TAR |
| E4 | [历史新120](../../../../archive/tar-new120-muon-20260908/REPORT.md) | 3,840 updates，新表 median accuracy 60.48%→90.99%，旧表99.26%→80.88%；同时换 Muon，无 scratch 对照，不能证明预训练增益 |
| E5 | [real12 历史报告](../../../../research-proposals/restoration-real12-rescue-20260917/REPORT.md) | 不同训练阶段和读出对照；历史 reserved 已观察，比较预算不匹配；TAR/旧 Restoration 均不等于 V5.3 |
| E6 | [V5.3 编码对照与单表验收说明](../../../../research-proposals/v53-sparse-codec-20260919/README.md) | 短对照、充分拟合和用户撤销统一 1e-3 门槛分别记录；单表验收不等于八表通过 |
| E7 | [V5.3 八表共享协议](../../../../research-proposals/v53-eight-joint-fit-20260919/README.md) | 当前一个共享模型、均衡曝光、raw4 混合类型校准；当前状态需读取 DGX2 原始收据 |

## 六级阶梯的资格状态

课程主轴为 C1 单表强拟合、C2 多表联合拟合、C3 超多表联合拟合、C4 持续学习与旧表保持、C5 预训练后真实数据微调、C6 未见表泛化。每关通过是下一关资格。E3–E5 是历史方法与负结果参考，不提供 V5.3 通关资格。E6 仅支持一张线性表的人工验收，完整 C1 仍有缺口；E7 为 C2 探索，尚未验收，不能回填 C1。C3–C6 均未取得本次 V5.3 通过证据。早期预留的数据隔离不等于提前执行或通过泛化评价。

## E7 本次只读观察

DGX2 原始根：`/home/cms/experiments/tabu-v53-eight-independent-20260919-6293e95/joint-eight-raw4-20260919-r2/`。

来源：`summarize_joint.py` 读取 `experiment/run/updates.jsonl` 与 `evaluation-000-000012288-periodic.json`。2026-09-19T22:29:41+08:00 另查容器 `tabu-v53-joint8-20260919` 为 running；GPU 利用率 74%、62°C。未修改训练或复制数据/权重。

| 表 | checkpoint 12,288 的 Query 指标 |
|---|---:|
| discoscm_1729 | NMSE 0.278805 |
| linear_numeric | NMSE 0.108658 |
| nonlinear_additive_numeric | NMSE 0.104869 |
| interaction_numeric | NMSE 0.087256 |
| gated_numeric | NMSE 0.260322 |
| discoscm_2718 | NMSE 0.292360 |
| discoscm_31415 | NMSE 0.381210 |
| xor_classification | accuracy 0.998048 |

七数值表宏平均 0.216212，最差 0.381210；全部 train 行在固定探针中被覆盖。读取时日志已到 13,338 更新，但上述评估对应每表 1,536 次更新。尚无 terminal，不据此单独推断存活；存活结论来自另查容器。

基础 commit：`6293e959438a1febd85dcf59f8bd54b976b5fab8`。完整 overlay、镜像、数据与计划身份以远端 `launch.json`、`experiment/qualification.json`、`experiment/joint-manifest.yaml` 为准；仅基础 commit 不足以恢复本实验。
