> 后续验证优先使用 [Small 验证入口](../tar-small-validation/README.md)。本目录保留其原始 Standard 或尺寸对照协议，不作为默认后验验证入口。

# TAR 学习率配对短诊断（整批 ≤30 分钟）

Iris（120 train / 30 test）和 Diabetes（353 / 89）分别比较恒定学习率 1e-4 与 3e-5。其他训练设置相同：完整 54,071,520 参数、初始化 seed 1729、effective_episode_batch=1、无 warmup、每步重新采样 mask/codebook、相同数据与 episode 随机流。每条最多 128 更新或 330 秒训练循环，每个命令墙钟上限 420 秒，四条共同硬截止 1800 秒。

每 32 步重放同样的 8 个训练 mask episodes，记录 loss、accuracy / normalized MSE、支持权重平均最大值与熵。Test 仅作初始/最终联合评估，不选择 checkpoint。训练裁剪前梯度超过 1e6 或连续三步高 loss 且零梯度会提前停止并记录 stability_alarm，仍尽力执行最终评估与保存。该预警是诊断门槛，不是数学定理；不替代已有 clip_norm=1。

这轮只比较早期拟合效率，不足以证明避免历史第 1040 步附近的崩溃。时间或预警导致不同更新数时，只在共有更新步比较固定评估，末步结果另列。不得同时宣称 batch、warmup 的因果贡献。

GPU/W&B 源码快照启动：`python3 experiments/local/tar-lr-diagnostic/run_batch.py --output-root ../batch --command tabu-lab`。认证由既有启动器读取，不写入源码或产物。W&B 项目 tabu-lab，组 tar-lr-diagnostic-30m。

本仓库保留协议与输入作为可复现配置。原实验的结果与 checkpoint 继续绑定原始归档；当前集成的来源字段改变 checkpoint source identity，新的运行会产生新身份。
