# 标准 Small H8 old120：120正常更新＋30次loss replay

UTC2026-09-22 04:04:55 在 dgx2 启动唯一接续；实际训练宿主 PID2329081，容器 PID1，固定 train-20260920 CUDA FP64。当前运行与逐表结果见 decisions.json、CURRENT.md 和带时间戳状态文件。

用户授权沿用现有标准 Small H8（backbone3、Unit3、2,082,688参数），每轮120表各一次正常训练，再按本轮正常loss降序、表ID升序冻结排名，top6和top24各一次extra；重叠6表各加两次，共30次extra。normal namespace及episode index延续，extra独立namespace并按表从0计数，loss排名不在extra后重新计算。train_cycle只统计120条正常loss，不能当作固定checkpoint评估。

正常累计上限仍983,040，即每表8,192次；没有重置或加倍正常预算。旧任务在118,030安全保存并退出，原冻结代码strict resume只补50步到118,080（每表984次），这些均计入正常预算。切换后剩余正常864,960，即每表7,208次；仅对这些未来正常轮新增216,240次extra。接续实际1,081,200次更新，累计actual上限1,199,280。extra逐表曝光依排名而不同，不追补历史轮。

旧source8db026b3保留；新冻结source eae13514fda04c067cb7f111a56b76e66c0a3efa1dc110cbb2dc4ecf44664306仅改变protocol/runner并新增loss_replay，模型源码不变。新manifest identity1f9032fb2f5e30028595718b8f41e7453a9d94e69098489605de850a89f73bb4。数据120份及原固定Query seeds/recipe不变，组合编码、supervised_row25%、Query-only、AdamW lr1e-4。仅训练行拟合，无reserved/final_test。

显式迁移保留116模型entries、345 AdamW tensors、CPU/CUDA RNG、runtime、历史曝光与调度状态；只新增replay状态及strategy-change lineage，不能称不改配方的strict resume。普通CUDA preflight、初始checkpoint逐张量CPU读回、首个真实150步调度检查均passed；见preflight-dgx2.json、migration.json、initial-resume-readback.json、first-cycle-audit.json。首次120正常步每表一次，normal index984，额外更新名单与原始loss排名一致，top6各两次、extra index0/1；已观测loss/gradient finite且唯一任务持续增长。

固定训练行Query仍每15,360个actual更新评估，checkpoint每7,680actual更新及停止点保存；新输出尚未到首个固定评估点，不能据调度验收声称拟合改进。旧固定结果保留原panel；新指标当前/最佳只算本output已有固定评估。父终态、bridge、预检、迁移和launch完整stdout/回执在receipts/。

远端根 /home/cms/tabu-v54-small-replay-old120-20260922-eae1351，输出runs/old120.small.replay。manifest位于manifests/old120.small.replay.json；preparing模板保留为准备证据，不用于运行。monitor.py仅只读回查，不启动或重启。W&B由主线程独立配置；每台最多一个训练，不追加重复或预算，不改无关服务。

## 2026-09-22 已由 replay v2 接续取代

此v1任务已在U123602/S122520/E1082安全保存停止；原v1冻结源码bridge仅补28次待完成extra，边界U123630/S122520/E1110。旧output和bridge保留为证据，禁止自动恢复。当前唯一标准Small训练见 [v54-small-replay-v2-old120-20260922](../v54-small-replay-v2-old120-20260922/README.md)，保留模型/AdamW/RNG/历史extra，未来轮采用120正常＋42extra。旧CURRENT仅是历史时点。
