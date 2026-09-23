# TabU V5.4 Small：old120 从头联合拟合

用户 2026-09-22 授权 dgx2 使用标准 Small 训练 old120，每表平均8,192次更新。已于北京时间2026-09-22 01:26启动唯一正式尝试。当前状态以 `decisions.json`、`launch-dgx2.json` 和 `CURRENT.md` 为准；自动回查只读监测，不能重复启动。

- **单次预算**：120表共享一个Small和一个AdamW，每120步每表恰好一次，共983,040次更新，每表恰好8,192次。只授权一个run，不自动增加重复或预算。
- **初始化**：从头随机初始化，没有父checkpoint，没有加载Nano参数或优化器。model/order seeds为543000/543100。
- **标准Small**：width128、3层backbone、8heads、ff256、256slots、3层Unit、subtokens1；2,082,688参数。与新版V5.4文档附录及`V54Config(size="small")`一致，没有保留Nano的显式结构覆盖。
- **既有设置**：`constant_weight_composition_v1`组合编码、zscore、`supervised_row`25%、Query-only state_weights=[0,1,0,0]、AdamW lr1e-4。预测列以外的值可见；不是全表随机遮挡。
- **运行时**：dgx2 GB10，既有`wehub-python --profile train-20260920`固定GPU容器，CUDA FP64。源码SHA `8db026b30a6a2b071f0c0bfc8745ed686dcf4f0f68db886d61f69393d106ba87`；不变更核心代码、运行环境、矿工或无关服务。
- **数据**：与`v54-nano-old120-20260922`完全相同的120份冻结old120数据、SHA、目标列和split；每表204训练行、52保留行。57张numeric目标、43张nominal、20张ordinal。train-only拟合，不使用reserved/final_test，不开展泛化。
- **固定评估**：与Nano old120使用相同`train_fit`名称及masks/codes/windows/evaluation seeds；每表8掩码、每掩码51个target-only Query，共408次曝光。初始评估后，每15,360总步（每表128次更新）评估120表，直到983,040步，共64次训练后固定评估。每7,680步保存checkpoint。
- **时间上限**：14天wall保险上限，不改变983,040步的更新上限；实际耗时按运行速度更新，避免沿用短试验的48小时限制导致预算提前结束。

逐表报告当前/最佳R²、NMSE或Accuracy、Query唯一地址数、真实表曝光以及可见支持均值/多数类参考。`naive-query-reference.json`复用Nano同一数据评估bank的CPU数据诊断值，已验证数据指纹、probe及四条相关seed一致，并登记本manifest身份。它不是拟合模型。

Small和dustinstudio的Nano任务同时运行，各目录独立。Nano来自五表r1权重且MPS FP32，Small从头且CUDA FP64，预算也不同，因此不是只改变尺寸的因果对照；相同Query bank仅保证指标针对相同训练行任务。

## 启动与事实源

远程冻结根：`/home/cms/tabu-v54-small-old120-20260922-8db026b`；从其`src`目录运行。

- manifest：`experiments/local/v54-small-old120-20260922/manifests/old120.small.r0.json`
- output：`runs/old120.small.r0`
- 运行身份：`a1d36db65c2eb6830bd2883308d98fc052f9ef953baaed4c15be72cbef3ba7db`
- `preparation.json`：完整983,040步调度计数、每表episode静态检查、模型结构与参数量；没有执行本地模型训练。
- `preflight-dgx2.json`：CUDA普通预检；只证明所选代表的有限训练步骤和checkpoint读写，不代表120表拟合通过。
- `launch-dgx2.json`：唯一正式启动回执与完整命令。
- `initial-checkpoint-readback.json`：实际初始模型结构、FP64参数、空优化器和空lineage。
- `status-*.json`、`status-*.md`、`CURRENT.md`：带时间戳的实际命令、update增长、逐表固定曲线和终态。

本地执行 `python3 experiments/local/v54-small-old120-20260922/monitor.py` 只通过SSH读回并更新本目录报告，不启动或重启训练。异常先定位；自动回查不追加第二次尝试。

dgx2旧五表`joint5.r0`已正常完成81,920步，每表16,384次，checkpoint hash已验证并保留，见`prior-joint5-r0-terminal.json`。不能恢复旧五表r2或旧单表续跑。dustinstudio及其余三台现有主线不受本次主机切换影响。

## W&B 外部监控

2026-09-22接入独立只读日志镜像，全程未停止/续跑/更改本训练进程。启动前后PID一致、update继续增长，见同级v54-old120-wandb-20260922/training-after-monitoring.json。

在线地址：[v54-small-old120-dgx2-20260922](https://wandb.ai/zj3712/restoration/runs/v54-small-old120-dgx2-20260922)。镜像脚本/config位于同级v54-old120-wandb-20260922；本目录wandb-observer-process.json、wandb-launch.json、wandb-mirror-status.json、wandb-cloud-readback.json分别记录观察进程、在线run、已消费游标和实际云端接收。

横轴train/update和fixed/update为本任务新增步，另有global_update表示old120累计；每表new_exposure/cumulative_exposure分列。Nano接续的122880步父边界是本轮新增0步；当前/最佳包含该起点评估。每100新增步采样训练标量、每次完整上传120表固定Query；训练源码/环境/预算没有因监控改变。

训练趋势新增 `train_cycle/mean_loss`：严格120表各一次的完整循环等权平均，历史回填并持续更新；原 `train/loss` 每100步原始抽样及本地完整逐步日志保留。只重启独立镜像，训练不重启。实际云端验证见cycle-cloud-readback.json；定义及共享代码见同级v54-old120-wandb-20260922/README.md。

同一120表完整循环另加 `train_cycle/median_loss`、`train_cycle/p05_loss`、`train_cycle/p95_loss`；采用线性分位数，保留均值与原始单步抽样，并回填历史。

## 2026-09-22 已由 loss replay 接续取代

本旧均衡任务已在118,030步安全保存并停止；使用原冻结代码strict resume的独立bridge仅补50步至118,080。全部旧证据保留，禁止自动恢复旧任务或bridge。当前唯一标准Small训练已迁移至 [v54-small-replay-old120-20260922](../v54-small-replay-old120-20260922/README.md)，保持正常累计983,040上限并对剩余正常轮增加top6+top24的30次extra。新模型、AdamW、RNG和曝光保留，新的source/策略身份见新panel。旧CURRENT和历史曲线只代表旧任务时点，不是当前训练。
