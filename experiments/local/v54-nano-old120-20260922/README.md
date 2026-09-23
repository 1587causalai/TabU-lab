# TabU V5.4 Nano：五表权重接续 old120 拟合

2026-09-22 用户授权在 dustinstudio 从已完成的五表检查点开始 old120，每表平均 1,024 次更新。已启动唯一正式尝试，当前进度和逐表指标见 [CURRENT.md](CURRENT.md)，实际命令见 [launch-dustinstudio.json](launch-dustinstudio.json)。本目录与五表、单表实验分开记录。

- **预算**：120 表共享一个 Nano 模型、一个 AdamW；每120步每表恰好一次，共122,880次新增更新，每表恰好1,024次。只授权这一次，不自动追加重复或预算。
- **父模型**：dustinstudio `joint5.r1`，正常完成81,920步，原五表各16,384次。父 checkpoint SHA-256 `192974130e8b90ddcb16859dcb7a7918e2b6a98a9f33a370a74a3ad7dec95e46`。复制到新根 `parents/joint5.r1.completed.pt`，并保留对应JSON校验记录；原件不变。
- **接续含义**：`--initialize-from` 加载模型权重，优化器、RNG、schedule重置；不是strict resume，也不是随机初始化的独立重复。新增阶段结束时，五张旧表累计17,408次/表，其余115张表1,024次/表；模型历史共204,800次更新。
- **训练设置**：Nano width128、layers2、heads4、slots256、ff256、unit0、subtokens1；`constant_weight_composition_v1` 组合编码，zscore，`supervised_row` 25%，仅目标列Query计loss，AdamW lr1e-4。冻结训练源码SHA `8db026b30a6a2b071f0c0bfc8745ed686dcf4f0f68db886d61f69393d106ba87`。
- **运行时**：`train-20260920`，Python3.12.13 / Torch2.13.0，MPS FP32；MPS CPU fallback关闭；承接已有MPS优化路径。未改训练代码、环境、矿工或无关服务。
- **数据**：原 `experiments/local/tar-diverse-120-fit/corpus/data` 的120份冻结文件，目标列及SHA来自 `restoration-v54-single-table-20260921/table-panel-full120.json`。57张numeric目标、43张nominal、20张ordinal。每表204训练行、52保留行；保留原split，训练和评估只使用train。
- **Query评估**：每表固定8个掩码，每掩码51个target-only Query，每次评估408次Query曝光/表；逐地址平均后聚合，报告唯一Query地址数、逐目标列R²/NMSE或Accuracy。不是全表随机遮挡，也不使用reserved/final_test。
- **评估频率**：开始前记录120表初始指标；每15,360总步（每表128次新增更新）评估全部120表，共8个新增训练评估点；每7,680步保存checkpoint。48小时wall上限是防失控界限，不是增加更新预算。
- **配对观察旧五表**：保留父r1的masks/codes/windows/evaluation seeds与`train_fit`探针名称，旧五表fixed Query bank完全一致；新的`old120_fit`训练stage命名会独立派生训练mask/code/window随机流。model/order seeds为542000/542100。原五表起始读回可与r1终态配对比较；其余115表初始指标只作拟合起点，不作泛化结论。

## 核验与调度

`preparation.json` 保存120表hash/split/target、首个episode静态检查以及完整122,880步调度计数。`preflight-dustinstudio.json` 为现有MPS普通预检：选中最大行数代表及最大单表cell数代表，均通过；预检不是所有120表拟合通过的证明。`identity-and-runtime.json`、`initial-checkpoint-readback.json` 分别记录源/模型兼容性与实际初始权重、优化器清空、FP32读回。`naive-query-reference.json` 是同一fixed Query bank的可见支持均值/多数类参考，CPU数据诊断，没有模型训练。

用户新指令已替代dustinstudio的五表r2：该run在9,357步安全保存并停止，检查点hash已验证；见 `../v54-nano-joint5-20260921/r2-stop-for-old120.json`。禁止恢复或迁移r2。dgx2的五表r0随后已正常完成，按用户更新授权转入 `../v54-small-old120-20260922` 的标准Small old120；其余三台仍按各自单表主线执行。

远程冻结根：`/Users/dustinstudio/tabu-v54-nano-old120-20260922-8db026b`。

- manifest：`experiments/local/v54-nano-old120-20260922/manifests/old120.from-joint5-r1.json`
- output：`runs/old120.from-joint5-r1`
- receipt/stdout：`receipts/launch.json`、`receipts/run.stdout`
- 单次正式launch身份：`bafbc8d5183b1c3578b72a7f7a26b68f21691efa5e3417b6ce7d14fcb3f282ac`

在本地仓库运行 `python3 experiments/local/v54-nano-old120-20260922/monitor.py` 会通过SSH只读采集，保存带时间戳JSON和逐表Markdown，更新CURRENT及decisions。它绝不启动或重启任务。自动回查继续监测更新增长、逐表当前/最佳/终态及曝光数；异常先定位，不机械重启。以实际训练步数和终态为准，不把进程存在或平均指标当成拟合通过。
