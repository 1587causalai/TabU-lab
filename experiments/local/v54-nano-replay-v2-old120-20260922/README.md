# Nano backbone2,4 heads,Unit0 old120 replay-v2

用户授权既有任务采用120正常更新后top2三轮、top6两轮、top24一轮额外训练，共42 extra，每周期162实际步。排名冻结自同轮120个normal loss，按loss降序、表ID升序打破并列。top2各6 extra、接下来4张各3 extra、接下来18张各1 extra；正常覆盖每表一次。

起点actual=204690，normal=202560，已发生历史extra=2130。正常上限保持1105920；剩余normal=903360；未来V2 extra=316176；最终累计extra=318306；实际总上限=1424226，本接续最多1219536步。旧extra不归零、不重算，也不追补既往normal轮。base_counts包含normal和旧extra，base_extra_counts单列，逐表曝光不均摊。

旧任务在204660安全停止；冻结v1代码strict bridge 30个actual步，补齐normal及本轮旧30 extra队列后迁移，无回滚。parent/model/AdamW/RNG/runtime/累计normal+extra及逐表曝光均保留；显式策略变更，不称配方不变的strict resume。

既有train-20260920 MPS FP32、模型结构/数据/optimizer不变；源码冻结0dfd1f3a6e350d5a5434f128ec25ce4277699f1a240f9891444b31374534c421。组合编码、supervised_row25%、Query-only，固定训练行Query，未使用reserved/final_test。普通preflight、不可变初始checkpoint CPU精确读回、实际首162步排序和finite检查通过。

唯一实际PID1226，启动UTC2026-09-22T04:41:28.773810+00:00。远程root `/Users/dustinstudio/tabu-v54-nano-replay-v2-old120-20260922-0dfd1f3`，output `runs/old120.nano.replay-v2`。当前normal/extra/actual分别看decisions和带时间戳monitor结果；尚未以这次调度验收宣称拟合提升。train_cycle均值/中位数/P05/P95仅120正常loss；extra单独报告。W&B与automation由主线程统一接入。

证据：parent-live-audit、deployment-readback、parent-stop、bridge、boundary-readback、preflight、migration、launch、initial-resume-readback、first-cycle-audit。旧任务标记superseded禁止恢复，不追加重复预算，不改无关服务。
