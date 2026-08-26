# Percy Liang 是怎么做的：Marin 开放科学预训练研究复盘

> 整理日期：2026-08-26
> 来源：research-wehub / garry_tan 在 Discord #percy-liang 系统理解子区的分析 + 公开资料
> 用途：作为 Percy Liang 数字分身计划的认知基线材料（v1 基线 thread: 1541738219337678949）

> 文档边界：这是一份历史综合稿，不是 Marin 当前状态的唯一事实源。参考工作的总索引、独立研究卡和待核验问题见 [`docs/references/`](references/README.md)；涉及最新项目状态、模型结果或外部贡献情况时，以对应 canonical source 和独立 receipt 为准。

## 相关网站与基本信息（2026-08-26 核验，链接均 200）

**人物**
- Percy Liang 个人主页：<https://cs.stanford.edu/~pliang/>
- X/Twitter：<https://x.com/percyliang>
- Stanford CRFM（HELM 所在中心，Percy 任 director）：<https://crfm.stanford.edu/>

**Marin 项目（本文档主角）**
- 官网：<https://marin.community>
- GitHub 主仓：<https://github.com/marin-community/marin>
- 实验预注册 issue 列表（`label:experiment`）：<https://github.com/marin-community/marin/issues?q=is%3Aissue+label%3Aexperiment>
- 博客（站内 `/#blogs` 锚点；示例文章：2025-05-19 announcement、2026-05-21《Async RL from Scratch on TPUs》 <https://marin.community/blog/2026/05/21/async-rl-from-scratch/>）
- 实验 = PR 的实例：`experiments/exp163_bert.py` 等，全部在主仓 `experiments/` 目录下可追溯

**相关开放科学项目（对比对象）**
- HELM（ holistic evaluation）：<https://crfm.stanford.edu/helm/>
- LLM360 / DataComp：一次性放全套的「完成时」路线，与 Marin「进行时」路线对比见文末

**我们自己的投影**
- TabU-lab canonical source：`wehub-system-msp/causal-superintelligence/TabU/tabu-lab`
- TabU-lab GitHub 公开投影：<https://github.com/1587causalai/TabU-lab>

> 注：Simile（Percy 2026-08 新创公司）尚无稳定公开官网，暂不给链接；Speedrun 独立域名未验证存在，入口以 marin.community 站内为准。

## 一句话总结

Percy Liang 团队主导的 Marin 项目讲的故事是：**把预训练科研变成一场完全公开、可参与的连续剧**——它不发布模型，而是发布*过程本身*。核心叙事可以叫 **"Process as a Product"（把过程当产品）**。

## 背景：为什么这个故事有杠杆

- 开源软件成功了，但开源 AI 还没到那一步。大家有 open weights（Llama / DeepSeek / Gemma），却没有 recipe——代码、数据、实验过程都不开放。
- 几乎所有巨头都在把大模型训练变成绝对黑盒。Marin 选择在明显劣势（算力拼不过大厂）下寻找不对称优势：**激进的透明度本身就是护城河和号召力**。
- 他们不仅开源赢的局，连踩坑、失败的实验、所有蠢错误都实时、全量公开。

## 叙事展开的四个关键手法

### 1. 把科学规范移植到开源流程里

每个实验是一个 GitHub issue，充当「迷你预注册」（preregistration）：先声明假设和目标，再跑实验，杜绝先出结果再圆逻辑的 p-hacking。这是叙事上最聪明的一步——把自己和「发 arXiv 论文的 lab」区分开来。

### 2. 实验即代码，过程即证据

实验在 PR 里以代码声明，跑出来的 provenance graph 自动沉淀成 WandB 报告。叙事单位不是博客文章，而是可复现的 **issue → PR → report 链条**。官网只是这条链的橱窗。

### 3. 邀请参与而非仰望成果

首页 CTA 不是「下载模型」，而是 install 代码、跑你的第一个实验、加 Discord。故事的主角是社区，Percy Liang 团队只是发起人。

### 4. 节奏感靠博客 + 仪表盘维持

博客很少（一年两篇，最近一篇 2026-05-21《Async RL from Scratch on TPUs》），但 Perplexity Gap Dashboard、Scaling Laws Explorer 这些分析视图让数据自己持续讲故事。

## 执行飞轮（The Execution Flywheel）

1. **强制预注册建立 Trust**：Issue 声明假设 → PR 实现 → 自动聚合到 WandB。过程正义保证科学严谨性。
2. **游戏化与算力杠杆（Speedrun）**：把架构和训练效率优化变成极客圈的速通竞赛——谁的算法效率高，就给谁免费算力去 scale up。用最小预算撬动野生 hacker 社区，形成自造血飞轮。
3. **拿结果说话**：Marin-8B 和 32B 在同等规模上实打实 beat Llama 3.1 8B 和 OLMo 2 32B。「过程正义且结果能打」，叙事才立得住。

## 活跃度与成色判断（2026-08-26 快照）

- GitHub 主仓最新 commit 为当天，自动化 bot 在同步外部 runtime（evalchemy / harbor / MarinSkyRL），日常流水线在跑。
- 统计真实开发活跃度时要区分「人力 commit」和「runtime-updater bot commit」，别被表面 commit 频率骗了。
- 弱点：「公开」不等于「有人真的在外部复用它的实验」。验证故事真实成色的指标是**外部贡献者的 PR 占比**，而不是 star 数。

## 对 WeHub 的启示

1. **Agent 能力的完美沙盒**：`Issue -> PR -> 代码 -> WandB` 构成高度标准化的科学流水线，上下文结构化、结果机器可读（provenance graph），是验证数字分身科研能力的最佳演练场。
2. **社区激励机制可借鉴**：Speedrun 模式可 mapping 到 WeHub 社区——不只做任务分发，而是建竞技场机制，让做出优秀 workflow 或效率优化的开发者直接获得资源倾斜（算力放大）。
3. **透明度作为定位武器**：在防御性的赛道里，「进行时」比「完成时」更有号召力。WeHub 的系统进化也可以考虑公开过程而非只公开成品。

## 与其他开放项目的对比

Marin 比 LLM360 / DataComp 这批「一次性放全套」的项目更进一步：它卖的是**进行时**，不是完成时——开放科学作为基础设施，而不是开放产物作为终点。

---

*相关：Percy Liang 2026-08 创办 Simile（模拟人类行为），Series B $200M @ $2B（Greenoaks + Index），15 岗招聘；其 confidence model + 每周真人验证机制可另行借鉴。*

## 从复盘到执行：Marin 手法 → TabU-lab 约定（2026-08-26 追加）

> 背景：gong 已确认工作重心转向亲自训练表格模型（USL01 理论 / OAttention 架构均完成构造期）。本文档从「Percy 分身认知基线」升级为 **TabU-lab 开放训练的操作规范来源**。canonical source：`wehub-system-msp/causal-superintelligence/TabU/tabu-lab`；GitHub `1587causalai/TabU-lab` 为公开投影。

### 逐条映射（Marin 做法 → 我们的落地）

| Marin 手法 | TabU-lab 落地约定 |
|---|---|
| Issue = 实验预注册 | 每个正式实验先开 issue：声明假设、目标指标、成功判据，再跑；杜绝先出结果再圆逻辑 |
| PR 即实验代码 | 训练代码只经 PR 进 main；PR 描述必须含 config 摘要与预期影响 |
| WandB provenance 报告自动沉淀 | 每个 run 记录 git SHA + Hydra config + dataset version + seed；预训练与微调 run 建 lineage |
| Speedrun 竞赛机制 | v1 先不做社区竞赛，但把「同预算下效率」作为内部评测维度之一 |
| Dashboard 让数据自己讲故事 | 公开入口页放 gate 进度看板 + wandb 关键曲线，随 gate 推进更新，不做空壳宣传页 |
| 外部贡献者 PR 占比 = 真实成色 | 我们同样用这个指标自检开放性，而不是 star 数 |

### 防漂移条款（源自 2026-08-26 设计讨论）

1. 数学内核先行：方案文档第一层是 unit/机制/噪声的形式化定义（继承 USL01 的 $Y=f(U,\varepsilon)$ 视角）；任何新组件必须能映射回第一层数学对象，否则不进架构。
2. 归因原则：**"预训练带来增益"必须通过同架构随机初始化、相同微调预算的对照证明**。
3. Gate 化推进：Gate 0（100 条数据过拟合 sanity check）→ Gate 1（端到端一条命令出 checkpoint）→ Gate 2（优于随机初始化对照）→ Gate 3（固定协议下 vs XGBoost/CatBoost/FT-Transformer，多种子 mean±std）。
4. 先 vertical slice（1 语料 × 1 模型 × 1 目标 × 2–3 数据集 × 3 seeds），后 sweep。

### 待办

- [ ] 方案设计文档（数学内核 + 表示层 + 预训练 objective 证明义务）起草
- [ ] 公开入口页面接入 dgx + Cloudflare 流水线，内容=安全投影（愿景/gate 看板/repo 链接）
