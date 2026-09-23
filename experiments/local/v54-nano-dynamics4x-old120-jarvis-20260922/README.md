# Nano-Dynamics4x · old120 · jarvis

状态：UTC 2026-09-23 02:16 已在 jarvis 唯一启动（PID 2519190）。截至 UTC 02:30 的[只读状态](status-20260923T023019Z.json)，已真实完成 actual 717、normal 549、extra 168；本输出 step0 120 表固定 Query 完整保存，source/data/manifest 身份匹配。用户授权每表新增8192次正常更新，后续曲线及终态继续按时间戳读回，不能凭启动或 step0 宣称拟合恢复。

从dustinstudio当前已落盘Nano检查点只复制Token dynamics：原两层按[0,1,0,1,0,1,0,1]扩为8层，四份独立可训练参数，width128/heads4/Unit0。编码器、读出各保留一份，不置零或缩放残差；不是标准Small，不保证起点输出不变。MPS FP32父权重转入jarvis CUDA FP64，weights-only，AdamW/RNG/调度重新开始。

120张既有训练表，组合编码、supervised_row25%、Query-only；固定训练行8个mask，不用reserved/final_test。新normal预算983040=120×8192；沿用P99×3/P95×2/P80×1叠加加训，每周期120normal+42extra，总actual1327104。父训练历史单列，不计作本实验新增曝光或收益。

冻结源码与原v2相同0dfd1f3a6e350d5a5434f128ec25ce4277699f1a240f9891444b31374534c421。父 Nano 固定在 actual 307200 的不可变检查点 SHA `a1dc7ce1…`；四倍新模型 checkpoint SHA `edbc4890…`，正式 manifest identity `0741cc16…`。普通 CUDA FP64 preflight、严格层复制、独立参数与初始状态读回均通过。代表性首步耗时约 0.9–1.4 秒，因此将继承的14天时间护栏延至30天，以便执行用户给出的更新预算；normal/extra/actual 上限未变。最初14天 manifest 的转换与起点评估只作诊断，正式任务已用30天 manifest 重新 preflight/转换并从新身份启动。

同一 Jarvis CUDA FP64 和同一 120 表训练行 Query bank 的诊断起点比较：父 Nano 的数值表 R² 中位数0.9786、nominal准确率0.8978、ordinal准确率0.5625；原样四倍复制后分别为-0.0837、0.5700、0.4203。全部57张数值表均退化，说明复制未直接保留已有拟合。正式 step0 逐表固定 Query 与诊断四倍模型的评估一致；训练是否恢复尚未获证据。详细原始逐表结果在 `parent-fixed-cuda.json` 与 `grown-fixed-cuda.diagnostic-14d.json`，正式 step0 另存于训练 output。

执行证据依次为`parent-freeze.json`、`source-freeze.json`、`preflight-30d-jarvis.json`、`conversion-30d-jarvis.json`、`launch-jarvis.json`及远端唯一输出。正式初始化 checkpoint 已按 SHA 内容地址另存于 [`initialization-freeze/`](initialization-freeze/)；其 SHA 与转换回执一致。矿工两次让出 GPU 的可恢复证据见`jarvis-miner-stop-receipt.json`与`jarvis-miner-second-stop.json`；不恢复矿工，不改其他服务。原三台训练继续，用户的 W&B 三路专用工作区保持原过滤。

独立的 [W&B run](https://wandb.ai/zj3712/restoration/runs/v54-nano-dynamics4x-old120-jarvis-20260922) 已由本地只读 observer 接入；[云端读回](wandb-cloud-readback.json)证实正式 fixed step0 的 120 表、前三个完整 normal cycle 的 mean/P99 与本地 outbox 相符。它是监控副本，训练真实进度仍以远端命令、journal 和本 panel 状态为准。

2026-09-23 03:02 UTC 的[故障诊断](failure-diagnosis-20260923.md)：页面短暂显示 `Failed` 是 observer 的 SSH 读取超时；训练 PID 仍推进，但裁剪前梯度与 loss 已明显失稳。observer 已用原 outbox/原 run ID 恢复，[云端再读回](wandb-recovery-readback.json)为 `running`，未重启训练。四倍模型能否恢复仍须看之后的固定 Query，不把监控状态当作训练终态。
