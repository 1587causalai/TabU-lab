# V5.4 Nano 五表联合拟合 — 2026-09-21

**2026-09-22 用户新指令覆盖下文旧 r2 计划：dustinstudio 转入 old120，使用已完成 r1 检查点。r2 已在 9,357 步保存并停止，禁止恢复或迁移重启。dgx2 的 r0 已正常完成81,920步，主机按用户新指令转入标准Small old120，每表8,192次。见 `../v54-small-old120-20260922/README.md`。** 见 `r2-stop-for-old120.json` 及 `../v54-nano-old120-20260922/README.md`。

用户授权从单表探索进入五表联合拟合。每个 run 只有一个模型和一个 AdamW 优化器，五张表均衡交替，每五步各表一次；不是五个单表模型并行。

- 表：discoscm_076、discoscm_095、scm_mixed_v1_001、scm_mixed_v1_017、scm_mixed_v1_040。沿用原始冻结数据，各表204训练行；不访问 reserved/final_test。
- Nano width128 / layers2 / heads4 / slots256 / ff256 / unit0 / subtokens1；组合编码、zscore、supervised_row 25%、Query-only、AdamW lr1e-4。冻结代码和现有加速路径不变。
- 每重复81,920次全局更新=每表16,384次。三个独立种子重复总计245,760次=每表49,152次。r0在dgx2 CUDA FP64，r1在dustinstudio MPS FP32，r2只在其中首个正常完成且空闲主机启动一次。两台设备和种子同时变化，因此不作纯后端性能因果对比。
- 每5,120总步评估一次，即每表1,024次曝光；每2,560总步保存检查点。固定8个训练行Query掩码，逐表当前/最佳/终态；不以五表宏观均值替代逐表拟合。
- 新 evaluation seed，旧单表曲线不能直接当成同掩码配对对照。旧MLP/XGBoost报告在训练标签量、掩码和聚合上未完全配对，不能宣称公平排名。
- dgx2和dustinstudio停止旧单表接续并保留结果；其他三台单表任务继续。禁止恢复用户已删除的学习率分支。

运行事实源 decisions.json、launch-*.json、带时间戳status文件。准备期间state=preparing，自动回查禁止启动。每台最多一项训练；只用train-20260920环境；不改矿工或无关服务。第三重复完成后不自动增加预算。异常先定位。

## 启动与核对

每台独立冻结根 `~/tabu-v54-nano-joint5-20260921-8db026b`，进入 `src`。manifest 路径 `../experiments/local/v54-nano-joint5-20260921/manifests/joint5.rN.json`，preflight 输出 `../preflight/joint5.rN`，正式输出 `../runs/joint5.rN`。现有 `~/.local/bin/wehub-python --profile train-20260920 -u -m tabu_lab.cli curriculum-v54 preflight|run --manifest ... --output-root ... --device cuda:0|mps`；preflight 可传 `--max-seconds 180`。禁止传 resume/initialize-from，禁止覆盖输出。MPS 必须在进程导入前设置 `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0`。

`naive-query-reference.json` 为三个重复各五张表的可见支持均值/多数类参考，按对应固定Query bank重新计算，仅CPU数据诊断，没有模型训练。对应manifest identity可在decisions.json核对。

`single-table-terminal-*.json` 保存旧单表停止/完成回执和已核实的checkpoint hash。旧单表结果不能与联合结果混为同一曲线。

## r2 已分配（北京时间 2026-09-22 00:43）

dustinstudio 的 r1 已正常完成81,920步，每表16,384次，final固定评估与checkpoint hash已核实。r2已在dustinstudio独立从头启动，见 `launch-dustinstudio-r2.json` 和 `decisions.json`。禁止在dgx2或其他主机再次启动r2；dgx2完成r0后保持空闲，本项目不追加第四重复。
