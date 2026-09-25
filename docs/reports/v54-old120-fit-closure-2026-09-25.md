# V5.4 old120 拟合阶段收口（2026-09-25）

用户确认已亲自停止 dgx2 标准 Small 与 gongqian-mini Small-H4，认为本阶段训练已经足够。两项 runner 的技术终态是 `interrupted`，但均无训练错误，且有可校验的持久 checkpoint。这里的含义是**主动提前收束**，不是训练失败，也不是跑满原预算。dustinstudio Nano 此前已另行[主动停止封存](../../experiments/local/v54-nano-replay-v2-old120-20260922/STOP-ARCHIVE.md)。这三路 V5.4 old120 当前均不再训练，旧任务不自动恢复或追加预算。

| 任务 | 终态累计 actual / 上限 | normal / 上限 | extra 累计 | 最后完整固定 Query bank | 本地封存 checkpoint SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| [dgx2 标准 Small H8](../../experiments/local/v54-small-replay-v2-old120-20260922/CLOSURE.md) | 902,240 / 1,285,332（70.2%） | 699,278 / 983,040（71.1%） | 202,962 | 890,880，距停止点 11,360 actual | `a129f9aded559862c5416e57589c271e98316588c542c80b339cd6ff68af58f7` |
| [gongqian-mini Small-H4 变体](../../experiments/local/v54-small-h4-replay-v2-old120-20260922/CLOSURE.md) | 710,158 / 1,321,272（53.7%） | 530,392 / 983,040（54.0%） | 179,766 | 706,560，距停止点 3,598 actual | `fae80fcfcb7f571df7b92a2675edfa5b1a85c82efa539fbd97d9453b076dff41` |

两路均核对了真实终态、唯一内容寻址 checkpoint、source/data/identity、无活跃 trainer、有限日志和 W&B 云端 `finished` / `closed_after_stop`。W&B 的 `finished` 只表示监控已收口；`training_budget_completed=false` 保留预算未完成的事实。检查点本地副本、terminal、哈希及云端回执在各自 `frozen-owner-stop-20260925/`，远端原输出未覆盖。权重和原始 terminal 留在本地/远端实验存储，不随 Git 分发；Git 只保存小型封存索引、逐表收口指标和解释。共同冻结源码 SHA-256 为 `0dfd1f3a6e350d5a5434f128ec25ce4277699f1a240f9891444b31374534c421`，old120 数据 SHA-256 为 `f14ca644085f62c06a2c172f20773d6f4fe8dbdd6b1481478ff3f62d2e81c8dd`。两任务都是组合编码、`supervised_row` 25%、Query-only、120 张已有训练表；未用 `reserved` / `final_test`。

## 最后完整固定训练行 Query 评估

下表的“当前”是**最后完整 bank**，并非停止点权重的独立评估；“最佳”仅取该 v2 输出中各表历次固定 bank 的最佳值，不能当作终态。每个单元依次列出当前指标中位数 / 逐表最佳指标中位数 / 当前严格超过可见支持参考的表数。

| 模型 | Numeric R²（57 表，支持均值） | Nominal accuracy（43 表，支持多数类） | Ordinal accuracy（20 表，支持多数类） |
| --- | --- | --- | --- |
| dgx2 标准 Small H8 | 0.99419 / 0.99504 / 57 | 0.94722 / 0.95109 / 40（另 1 持平） | 0.61227 / 0.69490 / 15 |
| gongqian-mini Small-H4 | 0.99324 / 0.99421 / 57 | 0.98670 / 0.99335 / 41（另 1 持平） | 0.70112 / 0.71034 / 16 |

逐表当前/最佳/最佳步数、支持参考、normal/extra/累计曝光见 [dgx2 收口指标](../../experiments/local/v54-small-replay-v2-old120-20260922/closure-metrics.json)与 [Small-H4 收口指标](../../experiments/local/v54-small-h4-replay-v2-old120-20260922/closure-metrics.json)；原工作区另保留完整 [dgx2 状态](../../experiments/local/v54-small-replay-v2-old120-20260922/status-20260925T010042Z.json)与 [Small-H4 状态](../../experiments/local/v54-small-h4-replay-v2-old120-20260922/status-20260925T010041Z.json)。两路仍有低于可见支持多数类的尾表：`scm_mixed_v1_045` nominal 为 0.0481 / 0.0507，对应参考 0.1215；`discoscm_087` ordinal 为 0.2207 / 0.3012，对应参考 0.3536（前数为 H8，后数为 H4）。不能用宏中位数宣布 120 表全部拟合。

加训机会仍明显偏向数值表：本 v2 阶段新增 extra，dgx2 的 numeric / nominal / ordinal 为 200,584 / 1,268 / 0；Small-H4 为 177,270 / 96 / 0。早期“extra 全给 numeric”只是当时快照，此处以终态累计为准。两模型的结构、初始化历史、训练量与 CUDA FP64/MPS FP32 均不配对，不能拿中位数做公平尺寸或策略优胜判断；这些数值只描述训练行拟合，不能称泛化。后续 V5.5 若复用权重，明确选择 checkpoint SHA，并将严格续跑与 weights-only 新任务初始化分开。

原 [2026-09-23 阶段报告](v54-old120-fit-handoff-2026-09-23.md)及 [训练与监控手册](../guides/v54-training-and-monitoring.md)继续保留历史快照和方法，不再作为当前运行状态入口。V5.4 专用 [W&B 工作区](https://wandb.ai/zj3712/restoration?nw=brg2pegawui)保留供回看；两条 run 的云端数据已结束同步。自动回查在此收口后关闭，不会因旧 `running` 文本继续轮询或重启训练。
