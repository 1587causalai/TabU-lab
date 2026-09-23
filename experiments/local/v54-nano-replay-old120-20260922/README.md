# Nano old120：120表正常覆盖后对高损失表加训

> **已由replay-v2接续替代。** 当前任务为[`v54-nano-replay-v2-old120-20260922`](../v54-nano-replay-v2-old120-20260922/README.md)，保留normal及已发生extra。禁止自动恢复/重复启动此旧output。

用户2026-09-22明确授权dustinstudio的既有Nano old120训练接续采用相同120 normal + top6 + top24策略。本实验是一项新的训练策略接续，保留模型、AdamW、RNG及曝光，不是从头训练，也不是配方未变的strict resume。

## 起点与不回滚切换

- 原任务：v54-nano-old120-extend8192-20260922，runs/old120.continue-plus8192。
- 唯一旧训练PID67674在194,029步收到SIGTERM，当前更新完成后写合法checkpoint和interrupted终态，未丢弃已训练步。
- 原冻结8db026b代码、原manifest和train-20260920环境进行11步strict bridge，全部计入原normal预算；边界194,040步，120表各1,617次old120曝光。
- 边界checkpoint SHA `a439db2141803af42ed8dda1ad889ae002c380419057d18d20fff2ff4cd30e89`。显式strategy migration checkpoint SHA `3b7c7220e53268a61d59cb3f7fb8909fdcf60ade7e4098ee92c47af7fa3ad7b7`。
- 初始实际训练checkpoint已核验：61个模型条目、180个AdamW张量、RNG、runtime、normal游标和所有曝光精确保留。partial及extra queue为空，extra=0。只有运行计时按执行继续增长。

## 预算与调度

- 正常训练累计上限保持 **1,105,920** 步，即120表各9,216次；它包含最初122,880步和已授权每表另8,192次normal训练。
- 从194,040边界起尚有 **911,880 normal步 = 7,599个完整循环**。
- 每轮先覆盖120张表各一次，冻结其正常单步loss排序；按loss降序、表ID打破并列，top6各加训一次，然后top24各加训一次。top6在本轮合计3次、接下来18表2次、其他96表1次。
- 仅对剩余7,599轮新增 **227,970加训步**；本接续最多 **1,139,850实际优化步**，old120累计实际上限 **1,333,890**。
- 加训使用单独episode随机流和每表extra游标；normal随机流保持原游标。额外曝光按实际逐表记录，不均摊给120张表。
- 五张原joint5表各另有16,384次父历史。这里只统计old120累计，父joint5历史不混入normal/extra轴。完成时每表old120曝光为9,216 normal加其实际extra，五张旧表另外加joint5历史。

## 固定模型、数据与运行时

Nano：2层backbone、4 heads、Unit0、width128/FF256/slots256。使用已验证冻结源码eae13514fda04c067cb7f111a56b76e66c0a3efa1dc110cbb2dc4ecf44664306，仅新增策略调度；不改模型源码、优化器或既有运行时。dustinstudio既有train-20260920，MPS FP32，lr1e-4，组合编码constant_weight_composition_v1，supervised_row25%，Query-only。

同一120表、每表204训练行、固定8掩码/408次target Query曝光；不使用reserved/final_test，不开展泛化。普通MPS preflight通过仅表示运行资格，不代表拟合成功。原14天wall保险上限保持。固定评估与checkpoint沿用实际step周期15,360/7,680。

## 启动与监控

UTC2026-09-22 04:02:06唯一正式启动，PID94419，caffeinate94420，原任务和bridge均已退出，无重叠训练。

远端根：`/Users/dustinstudio/tabu-v54-nano-replay-old120-20260922-eae1351`；输出：`runs/old120.nano.replay`。真实命令、runtime和预算见launch-dustinstudio.json。新identity `8a947d1b39b25b715399a46165509979fed82e6a0af932ef0f665a16dae2ee75`，data `f14ca644085f62c06a2c172f20773d6f4fe8dbdd6b1481478ff3f62d2e81c8dd`。

- parent-live-audit.json：旧source40文件/data120文件、唯一PID、runtime和空间核验。
- parent-stop-request.json / parent-stop-readback.json / bridge-launch.json / bridge-readback.json：安全终止与11步边界补齐证据。
- preflight-dustinstudio.json / migration.json / initial-resume-readback.json：运行资格、显式策略迁移与真实新训练初始状态核验。
- first-cycle-audit.json：首次150步确为120不同表normal、top6和top24按冻结排序加训，6表得到2次extra、18表1次extra，全部loss/gradient有限。
- monitor.py / status-*.json / CURRENT.md：SSH只读真实命令、增长、normal与extra逐表曝光、固定Query当前/本输出最佳/可见支持参考；本输出最佳不表示父实验最佳。
- 远端receipts/run.stdout为本次训练日志；bridge.stdout/preflight.stdout/migration.stdout保留对应动作日志。

train_cycle均值/中位数/P05/P95只统计正常120表；extra单独监控。W&B观察进程及automation由主线程统一处理。本任务不自动增加预算/重复，不恢复旧任务或矿工，不改变其他服务。
