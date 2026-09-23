# 既有 old120 两个任务接入 W&B

用户2026-09-22授权将dgx2标准Small和dustinstudio Nano接续任务加入W&B。采用独立本机观察进程，只通过SSH读取已落盘训练日志和固定Query评估，补录当前任务的历史并持续跟踪。训练进程保持运行，源码、环境、优化器和预算不变。

- dgx2：标准Small（3层、8heads、Unit3），从头训练，当前任务起点0步，983040步预算。
- dustinstudio：Nano（2层、4heads、Unit0）接续，当前任务起点old120累计122880步（每表1024次），本次新增983040步。不导入已完成父任务的早期曲线；接续目录内122880步重评估作为本轮起点。父模型更早的五表训练单列，不混作old120曝光。
- gongqian-mini Small-H4保留独立镜像及run；后续完整循环loss功能已统一加入三路镜像。

W&B entity/project为zj3712/restoration。每100次新增更新采样训练loss/gradient/seconds，完整导入每次120表固定Query评估及常数参考。train/update和fixed/update是本次新增步数，另记录global_update；每表new_exposure与cumulative_exposure分列。best包含各自初始化/接续起点。

观察环境：/Users/cms/.wehub/envs/tabu-wandb-monitor-20260922。新mirror与两个config在本目录；每个任务的outbox、启动回执、镜像状态及云端读回放回对应实验panel。SDK自动使用本机已有登录；不传输密钥、表格内容、权重或代码。镜像失效不影响训练。

身份与语义以config、schema审查和启动前后实际进程/训练增长读回为准。W&B页面或SDKenqueue不能代替云端history已收到指标的验证。固定Query只来自训练行，不开展泛化；三种模型结构、初始化与后端并非完全配对对照。

## 在线入口

- [dgx2标准Small](https://wandb.ai/zj3712/restoration/runs/v54-small-old120-dgx2-20260922)
- [dustinstudio Nano接续](https://wandb.ai/zj3712/restoration/runs/v54-nano-old120-extend8192-dustin-20260922)
- [既有gongqian-mini Small-H4](https://wandb.ai/zj3712/restoration/runs/v54-small-h4-warm-old120-gq-20260922)

UTC02:19:38启动的两个本机观察PID分别77849/77851。training-before-monitoring.json与training-after-monitoring.json验证原训练PID未变，两机update继续增长。cloud_readback.py通过SDK只读核对云端元数据、训练/固定Query history、完整120表集合及global/new横轴和曝光。tabu-nano自动回查已纳入三个镜像；异常先定位，不以监控故障为由重启训练。

UTC2026-09-22 02:22:55–59云端核验：Small训练采样已到99200步，完整固定bank0至92160共7点；Nano本次新增33200步/累计156080，完整固定bank累计122880、138240、153600共3点。两者均收到精确120表集合，global/new轴及每表新增/累计曝光核验通过，历史补录已追过接入前进度。最新状态以各自mirror-status和cloud-readback为准。

## 完整120表循环的训练 loss

2026-09-22按用户要求增加 `train_cycle/mean_loss`，横轴 `train_cycle/update` 为本次新增更新的循环结束位置；另有global_update和cycle index。从每一条原始更新的loss计算，按各任务起点0/122880/0分组，每120步必须覆盖120张表各一次，等权均值为sum(loss_i)/120。只有完整循环出点，部分循环跨poll保留。不是滑动120步平均，也不是对每100步抽样点做平均。

原有 `train/loss` 仍保留每100步的原始单步抽样；全部单步loss保存在各自未改动的updates.jsonl。两条曲线分别反映瞬时单表难度和一整轮120表的训练趋势。cycle mean跨越120次参数更新，不能替代同一checkpoint的逐表fixed Query评估，也不因曲线变平滑就证明拟合改善。

共享cycle_loss.py与cycle_mirror.py负责纯CPU聚合和独立原始日志游标；旧outbox与W&B run ID保留，历史cycle以独立横轴追加后持续更新。只重启本机W&B镜像，未向训练进程发送信号。备份、停止和新PID回执为cycle-observer-stop.json、cycle-observer-restart.json及各panel/monitor-history。cycle-audit-*.json核对真实均衡循环；cycle-cloud-readback.json核对实际云端均值与全量日志聚合相符。

## 循环内分布：中位数、P05、P95

随后按用户追加要求，三路同一完整120表循环新增 `train_cycle/median_loss`、`train_cycle/p05_loss`、`train_cycle/p95_loss`，同时保留 `train_cycle/mean_loss` 和原始 `train/loss`。分位数采用排序位置(n-1)*q的线性插值；n=120，q=0.05/0.50/0.95。它们是120张表各一次原始训练loss的分布，不是Query样本误差分位数或均值置信区间。

v2历史回填仅重置cycle统计读取游标；已上报mean事件按原唯一ID保留，新distribution事件另有唯一ID，不覆盖过去数据。未满120步的尾部仍不出分位数点。原训练进程不变；只重启三个本机观察进程，见quantile-observer-restart.json。云端校验同时检查四类统计、完整连续cycle索引及保留的原始抽样曲线。

UTC2026-09-22 02:49:56–02:50:05最终云端读回：Small均值/分位数871轮，Nano接续373轮，Small-H4 63轮；四项统计均逐点匹配全量原始日志聚合，无缺轮，原始单步抽样保留。三台训练PID与原启动一致且update继续增长，loss/gradient有限。完整回执cycle-statistics-upgrade.json。

## 主动停止与 W&B 结束状态

2026-09-22 修复旧镜像把所有非 `completed` 终态传为非零 exit 的问题。`terminal_lifecycle.py` 区分训练终态、预算完成与监控会话关闭：已保存的 `interrupted` / `stopped`，只有无异常、update 等于 durable_update、终态身份匹配且检查点实际字节 SHA256 验证通过，才以 exit 0 关闭 W&B，并写入 `telemetry/lifecycle=closed_after_stop`、`training_intentionally_stopped=true`、`training_budget_completed=false`。真实错误和缺失持久化证据继续以非零退出。检查点字节校验不等同于 Torch 加载验证。

旧均衡阶段和 replay v1 共六个 run 已核实为用户授权迁移的主动停止并修正为 Finished，添加 `intentional-stop`、`superseded`、`budget-incomplete` 标签和明确后继链接；原 `terminal/outcome=interrupted` 保留。Finished 表示监控正常关闭，不能用作预算完成或拟合达标证据。旧检查点、训练记录与预算没有修改。

服务端禁止通过 upsertBucket 直接把 state 改成 finished，已拒绝的尝试未改变 run。修正采用原 ID 的 `resume="must"` 加 `finish(exit_code=0)`，不调用 log，再恢复全部原 summary 并追加生命周期说明。六份 `lifecycle-repair-*.json` 逐项确认原 storage ID、配置、名称、summary 和 history 末步保持一致；SDK 会话文件及云端会话元数据会更新。独立终态审计为 `old-runs-intentional-stop-audit-20260922T045552Z.json`。代码验证回执 `terminal-lifecycle-fix-readback.json`：38 unittest + 8 pytest 通过。

当前 replay v2 三路使用同一共享镜像。升级只滚动本机 observer，保留各 panel 的 run ID 与 SQLite outbox；回执写在各 panel 的 `wandb-lifecycle-refresh.json` 和 `monitor-history/*-terminal-lifecycle/`。不因监控升级、异常或云端状态操作远端训练。

UTC 2026-09-22 05:08 最终验收：六个旧 run 逐项通过 `lifecycle-cloud-final.json`；当前三路 W&B 均 Running，每路恢复边界前24点、后48点共72个连续事件逐字段匹配 outbox，零缺失、零差异、零重复（见各 panel `wandb-resume-cloud-readback.json`）。三台远端原训练 PID 2371967 / 1226 / 32698 及命令不变，actual 更新分别由127294→128984、209320→211332、28771→30357，均无终态。全部操作收口于 `lifecycle-resolution.json`。

当前训练入口：[dgx2 Small v2](https://wandb.ai/zj3712/restoration/runs/v54-small-replay-v2-old120-dgx2-20260922)、[dustinstudio Nano v2](https://wandb.ai/zj3712/restoration/runs/v54-nano-replay-v2-old120-dustin-20260922)、[gongqian-mini Small-H4 v2](https://wandb.ai/zj3712/restoration/runs/v54-small-h4-replay-v2-old120-gq-20260922)。旧 run 的后继链已写在 `telemetry/superseded_by_url` 中。

生命周期行为依据 W&B 官方 [resume](https://docs.wandb.ai/models/runs/resuming) 与 [finish](https://docs.wandb.ai/models/ref/python/functions/finish)；已有训练指标的真实性依赖上述本地和云端读回。

## 当前三路训练的专用工作区

用户反馈通用项目入口仍显示三十多个实验后，已另建具名工作区 [restoration5.4](https://wandb.ai/zj3712/restoration?nw=brg2pegawui)。后续给用户监控入口应使用这个带 `nw` 的专用链接，通用 `/restoration` 链接可能打开查看者的默认工作区。新工作区精确过滤 Small/Nano/Small-H4 三个 replay-v2 run，最多显示三条 run 曲线，关闭自动生成指标图，只保留置顶的 P99、P95、median、mean loss 四图（2×2）。全部使用 `train_cycle/update` 横轴。只新建视图，不移动、复制或删除 run，不影响训练或日志上传。通过专用 URL 重新加载并核对过滤及布局配置，回执为 `restoration54-workspace-receipt.json`。
