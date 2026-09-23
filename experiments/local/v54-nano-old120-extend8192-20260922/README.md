# Nano old120：每表再训练8,192次

> **当前状态：已由Nano replay接续替代。** 2026-09-22本任务在194,029步安全终止，原代码补11步到194,040边界后转入[`v54-nano-replay-old120-20260922`](../v54-nano-replay-old120-20260922/README.md)。本目录保留历史证据；禁止自动恢复、重启或重复启动此旧任务。当前训练和预算以新panel为准。

用户2026-09-22授权从已完成的Nano old120 checkpoint继续，每表新增8,192次。唯一正式接续尝试在dustinstudio既有MPS FP32环境运行；状态以decisions.json、launch回执和CURRENT.md为准。preparing/launching期间自动回查只读，不自行启动。

- 本次新增983,040步；old120从122,880接至1,105,920步。120表每表从1,024到9,216次；五张旧表另有父joint5.r1的16,384次历史，最终25,600次/表，其余115表9,216次/表。含五表历史的模型总更新1,187,840。
- 在原已完成checkpoint上作显式budget-only迁移，保留模型、AdamW状态、已有RNG、训练episode游标及曝光。不是从头训练，也不清空优化器。原checkpoint、终态与输出保持不变，新输出runs/old120.continue-plus8192。
- 源码仍为8db026b，train-20260920 MPS FP32，lr1e-4，组合编码constant_weight_composition_v1，supervised_row 25%，Query-only。训练与固定评估只用train行，不用reserved/final_test。
- 每15,360累计步评估120表，每7,680步保存checkpoint。fixed Query bank和所有seeds不变；在接续起点重评估一次核对父终态。stage名称和episode游标保持，避免重复旧训练随机流。
- manifest仅max_updates和max_seconds改变，累计wall保险上限为14天。继承的experiment_id、description和question属于父合同元数据，当前授权预算以这里和manifest数值字段为准。
- extend_budget.py沿用已验证预算迁移逻辑，仅适配V5.4 plan入口和显式路径。不会修改冻结训练包。保存实际模型、优化器、RNG、曝光和序列化校验回执，浮点训练轨迹细差仅诊断。
- 本任务与dgx2 Small从头old120不同；初始化、后端、预算不同，不能称为纯尺寸控制实验。不恢复其他旧任务或矿工，不自动追加后续预算。

远程复用冻结根：/Users/dustinstudio/tabu-v54-nano-old120-20260922-8db026b；新manifest位于experiments/local/v54-nano-old120-extend8192-20260922/manifests/old120.plus8192.json。自动回查只监测本接续，不恢复已完成父任务。

## 启动回执与接续验证

北京时间2026-09-22 08:48:14已启动，PID67674，power guard67675。新任务以--resume-checkpoint接续预算迁移checkpoint，不使用--initialize-from。

- preflight-dustinstudio.json：普通MPS预检通过，204行代表与最大32列代表均有限、读写恢复通过。
- migration.json：61份模型张量、180份优化器张量、已有RNG、曝光与episode游标逐项保留，序列化读回通过；父checkpoint不变。
- initial-resume-readback.json：实际新任务起点仍为122880步，60个AdamW参数状态的step均为122880，表曝光均为1024；读取到的模型/优化器/RNG与迁移状态一致。
- launch-dustinstudio.json：真实命令、环境、父/接续身份、PID和新增预算。
- monitor.py：只读SSH检查，保存status-*.json/.md和CURRENT.md。报告区分本次新增与old120累计；最佳值范围为父old120终态及本接续曲线，不混用旧五表的最佳值。启动时先重评估固定120表bank，父任务同一评估约245秒，评估后开始新增训练。
- automation-update-receipt.json：原tabu-nano自动回查已转向本接续和dgx2 Small主线；其他三台单表完成后保持空闲。

启动读回：startup-readback.json 已确认累计步数超过122880，新增loss/gradient有限；120表固定Query起点重评估与父终态各指标最大绝对差0.0。

## W&B 外部监控

2026-09-22接入独立只读日志镜像，全程未停止/续跑/更改本训练进程。启动前后PID一致、update继续增长，见同级v54-old120-wandb-20260922/training-after-monitoring.json。

在线地址：[v54-nano-old120-extend8192-dustin-20260922](https://wandb.ai/zj3712/restoration/runs/v54-nano-old120-extend8192-dustin-20260922)。镜像脚本/config位于同级v54-old120-wandb-20260922；本目录wandb-observer-process.json、wandb-launch.json、wandb-mirror-status.json、wandb-cloud-readback.json分别记录观察进程、在线run、已消费游标和实际云端接收。

横轴train/update和fixed/update为本任务新增步，另有global_update表示old120累计；每表new_exposure/cumulative_exposure分列。Nano接续的122880步父边界是本轮新增0步；当前/最佳包含该起点评估。每100新增步采样训练标量、每次完整上传120表固定Query；训练源码/环境/预算没有因监控改变。

训练趋势新增 `train_cycle/mean_loss`：严格120表各一次的完整循环等权平均，历史回填并持续更新；原 `train/loss` 每100步原始抽样及本地完整逐步日志保留。只重启独立镜像，训练不重启。实际云端验证见cycle-cloud-readback.json；定义及共享代码见同级v54-old120-wandb-20260922/README.md。

同一120表完整循环另加 `train_cycle/median_loss`、`train_cycle/p05_loss`、`train_cycle/p95_loss`；采用线性分位数，保留均值与原始单步抽样，并回填历史。
