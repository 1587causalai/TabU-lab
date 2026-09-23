# Small-H4 old120：均衡覆盖后按loss加训

> **已由replay-v2接续替代。** 当前任务为[`v54-small-h4-replay-v2-old120-20260922`](../v54-small-h4-replay-v2-old120-20260922/README.md)，保留normal及已发生extra。禁止自动恢复/重复启动此旧output。

2026-09-22用户确认采用每轮120表均衡覆盖，再给最高6表和最高24表各一次额外训练；重叠的6表各有两次额外更新，完整轮150次实际优化。先接入gongqian-mini已有Small-H4变体，dgx2标准Small与dustinstudio Nano保持当前合同。

当前状态以decisions.json为准。preparing/launching时仅只读回查，不启动或自动重启。

正常预算仍是Small-H4从其Nano初始化起累计983,040步（每表8,192次）；切换前均衡更新保留，切换后每完整120正常步新增30次extra。不追补历史轮。实际总上限由干净的120步边界父checkpoint计算，写入manifest与launch回执。

正常loss排名一次固定，按loss降序、表ID升序确定名单。extra使用同一supervised_row Query-only配方和组合编码，独立episode namespace、逐表extra index；优化器持续更新。normal与extra日志分别标记，train_cycle mean/median/P05/P95只取正常120条；extra原始loss与曝光单列。固定Query仍用原训练行bank，每15,360个实际optimizer steps评估一次，最终预算点评估；checkpoint每7,680实际steps及停止点保存。

从旧任务最新合法停止点迁移，保留模型、AdamW、RNG、曝光与游标。若旧任务在半轮停止，用旧冻结源码在不可覆写bridge目录补齐至120边界，最多119次，计入原正常预算。再用新冻结源码和明确strategy-change lineage继续。新增源码/策略身份，不称未改变配方的strict resume，也不重置为weights-only。

MPS FP32，沿用train-20260920。不改模型结构、loss权重、lr1e-4或数据；不访问reserved/final_test。父Nano及joint5历史保留在祖先链，不冒充此次新增曝光。此试验尚不证明新策略优于均匀训练。

## 已启动及实际预算

UTC2026-09-22 03:29:35启动，PID29954，MPS FP32/train-20260920。源代码eae13514fda04c067cb7f111a56b76e66c0a3efa1dc110cbb2dc4ecf44664306；manifest identity060e06f3d6efd567a7fc8d1d2a8c51845343b598ae74a6b92ccfa6f774a3c668。父均衡H4在13,814步保存，bridge在旧源码下补106步至13,920（每表116次），全部计入原正常预算。

接续剩余正常969,120步＝每表8,076次，新增extra242,280步，共1,211,400次实际更新；H4累计正常上限983,040，实际总上限1,225,320。不是每张表平均再加8,192次；已完成正常更新保留。extra曝光按动态排名不均匀，必须逐表记录。

migration.json确认116个模型entries、345个optimizer tensors、RNG、历史exposure/cursor保留；普通MPS preflight通过。新的W&B run：[Small-H4 balanced120 + loss replay](https://wandb.ai/zj3712/restoration/runs/v54-small-h4-replay-old120-gq-20260922)，和原均衡run分开以标明策略变化；云端实收以wandb-cloud-readback.json为准。旧observer停在原任务终态，bridge完整日志另存旧根/runs/old120.small-h4.bridge-to-replay，不覆盖旧文件。

monitor.py只读真实训练并保存逐表CURRENT/status。normal/extra原始日志均保留；train_cycle横轴为本次新增normal步，train/replay横轴为本次新增actual步，另记录累计actual和normal。fixed沿用actual步与固定train Query。当前/最佳仅本output已有固定评估，不继承父最佳；无评估时明确未产生新拟合结果。

首轮真实150步审计与初始checkpoint保留读回均通过，见first-cycle-audit.json及initial-resume-readback.json。W&B固定快照实收3个完整normal轮的mean/median/P05/P95、90个extra、5个raw采样，全部匹配本地outbox；约一个上传周期延迟。此为调度与监控验收，不是拟合优胜结论。完整交付回执见implementation-and-launch-receipt.json；tabu-nano自动回查已切换至本panel，仍每120分钟，仅监测。
