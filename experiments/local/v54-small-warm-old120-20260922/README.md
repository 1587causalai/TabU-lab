# Small-H4 变体：Nano权重扩深后的old120拟合

> 2026-09-22最新状态：原均衡任务已在13,814步安全保存，旧代码bridge补106步至13,920。之后保留模型/AdamW/RNG，转入同级 `v54-small-h4-replay-old120-20260922` 的120＋30策略。以下为原任务合同与历史证据；不要恢复旧output。

用户2026-09-22明确选择：保留Nano的4 heads，只增加层数，并标注为Small变体；不是标准Small。gongqian-mini运行唯一正式任务，old120每表新增8,192次（共983,040步），W&B在线监控。preparing/launching阶段回查只读，不重复启动。

## 结构与初始化

- Nano：2层backbone、4 heads、Unit0，width128、FF256、slots256。
- 本实验Small-H4：3层backbone、4 heads、Unit3，其余维度相同，2,082,688参数；标准Small的8 heads没有采用。
- 从dustinstudio UTC2026-09-22 01:42:01当时最新已落盘的Nano old120 checkpoint冻结权重：138,240步，每表1,152次；SHA870efb270d8a735a0a209e6225c31598fee8f55e089c2bc5082614e0cd328e09。原训练继续，不读取不断变化的指针作为父文件。
- 复制Nano全部61个state entries（包含codec buffer）。新增55个entries；新增axial层3个OMAB及3个Unit OMAB共12个out/FF残差出口张量清零，其余新增权重按seed初始化。保持4 heads且新增残差为零，设计目标是保留Nano起点函数；实测范围与误差以conversion及初始固定Query读回为准。
- 显式架构转换后用普通--initialize-from进入新Small-H4任务：AdamW、RNG和训练游标重置，当前任务从step0计数，不能称为严格resume。
- 原五表另外各有joint5.r1的16,384次历史。本任务结束后，旧五表累计25,728次/表，其余115表累计9,344次/表；这些是父系曝光历史，不混入新任务8,192次/表的横轴。

## 固定实验合同

源码沿用冻结8db026b，gongqian-mini既有train-20260920、MPS FP32；不重装或修改训练环境。组合编码constant_weight_composition_v1、supervised_row 25%、Query-only、lr1e-4，120表均衡交替。每表204训练行，固定8掩码/408次target Query曝光；不使用reserved/final_test，不开展泛化。每15,360步完整评估120表，每7,680步checkpoint；14天wall保险上限不增加983,040步预算。

数据、固定Query bank、训练seeds/order及stage名称沿用dgx2 Small合同，以便对照。dgx2是标准Small8 heads从头CUDA FP64；本任务是Small-H4 Nano初始化MPS FP32，因此同时存在结构、初始化、设备精度和父历史差异，不能称为纯初始化控制实验，也不能把warm start直接当成每步更快或最终更优。

## 事实源与监控

远程根：/Users/gongqian/tabu-v54-small-warm-old120-20260922-8db026b；output runs/old120.small-h4.from-nano。manifest manifests/old120.small-h4.from-nano.json。

- parent-snapshot.json / parents：父SHA、时间戳和对应138240步固定Query银行。
- grow_from_nano.py / conversion-small-h4.json：逐张量复制、新增/零出口、配置差异和实际验证回执。
- preflight-small-h4-gongqian-mini.json：普通MPS资格检查，不能替代拟合结果。
- decisions.json / launch-gongqian-mini.json / initial-checkpoint-readback.json：当前状态、真实命令及实际初始化。
- monitor.py / CURRENT.md / status-*：只读远程读回，逐表当前/最佳/常数参考、实际新增曝光。
- wandb_mirror.py：独立本机观察环境，SSH只读已落盘训练/evaluation日志；训练每100步、固定Query每次完整120表上报。稳定run ID v54-small-h4-warm-old120-gq-20260922，entity/project zj3712/restoration。无权重、数据行或代码上传，镜像失败不阻塞/重启训练。
- wandb-launch.json及cloud-readback：真实W&B URL与云端实际收到指标的证据。

prepared-standard-small-not-launched/仅保存用户作出4-head选择前的标准8-head预备检查，未启动正式训练，不能混入本实验结果。

## 实际启动

UTC2026-09-22 01:58:01（北京时间09:58:01）正式启动，PID26778，MPS FP32，manifest identity dce4a4cc964d16bafd2d185c9eb6658df3c30d716136400678c4ef678174eee0。转换checkpoint SHA3a3ab361ad866e37a8cc9d608349f4bb4df36b838774974c5a16c1e75944cce2。MPS数值/名义/序数3个固定train episode中全部carriers、units、Query encoding/decoded输出差异0；这不是对全部120表训练效果的结论。

W&B：[Small-H4 warm old120](https://wandb.ai/zj3712/restoration/runs/v54-small-h4-warm-old120-gq-20260922)。实际指标上传以wandb-cloud-readback.json为准。

UTC2026-09-22 02:08:54启动读回：405步，120表均已更新，已记录loss/gradient全部有限；完整initial固定评估为step0，耗时513.58秒。初始checkpoint全部116个entries与转换件相同，fresh AdamW与RNG已核验。完整120表父/子Query地址和曝光一致；63/63分类accuracy相同，数值最大R²差9.06e-7、NMSE差8.22e-7。以逐表比较回执为准。

UTC02:08:55云端读回确认W&B收到train/update100、200、300及fixed/update0全部120表；训练曲线代表新增更新，起点指标不是本轮训练改善。现有tabu-nano每120分钟回查已纳入本任务，和dgx2标准Small、dustinstudio Nano接续分开监测；不自动扩大预算。

训练趋势新增 `train_cycle/mean_loss`：严格120表各一次的完整循环等权平均，历史回填并持续更新；原 `train/loss` 每100步原始抽样及本地完整逐步日志保留。只重启独立镜像，训练不重启。实际云端验证见cycle-cloud-readback.json；定义及共享代码见同级v54-old120-wandb-20260922/README.md。

同一120表完整循环另加 `train_cycle/median_loss`、`train_cycle/p05_loss`、`train_cycle/p95_loss`；采用线性分位数，保留均值与原始单步抽样，并回填历史。
