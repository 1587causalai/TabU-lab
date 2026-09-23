# 标准 Small H8 old120：replay v2 接续

UTC2026-09-22 04:41:11在dgx2启动唯一正式接续，实际Python宿主PID2371967，容器PID1。沿用标准Small H8/backbone3/Unit3、CUDA FP64与train-20260920固定镜像；模型源码、数据、optimizer超参及固定训练行Query bank不变。当前状态以decisions.json、CURRENT.md和时间戳回执为准。

每120表各一次正常更新后，按这些原始normal loss降序、table ID升序冻结排名。3个完整top2 pass、2个完整top6 pass、1个top24 pass，共42extra、每轮162actual；最高2表各额外6次，接下4表各3次，其后18表各1次。normal namespace/顺序/episode索引不变；extra使用loss_replay_v2独立namespace，但逐表episode index继续旧extra累计。train_cycle统计只取正常120条，extra另记，不能当固定checkpoint评估。

原v1训练在U123,602/S122,520/E1,082安全保存；原冻结v1源码strict resume只补28次尚未完成的extra到U123,630/S122,520/E1,110，队列空、正常整120。没有回滚或丢弃旧extra。正常累计上限N983,040（每表8,192次）保持；新起点每表正常1,021次，剩余正常860,520（每表7,171次），未来v2 extra301,182。本接续新增actual1,161,702，累计actualcap1,285,332=N+历史extra1,110+未来extra301,182。不追补历史正常轮；base_counts包含正常及历史extra，base_extra_counts记录各表历史extra，实际extra曝光取决于动态排名。

显式迁移保留116模型entries、345 AdamW tensors、CPU/CUDA RNG、历史exposure和调度状态，新增kind及base_extra_by_table，并追加strategy-change lineage；不是未改变配方的strict resume。新source0dfd1f3a6e350d5a5434f128ec25ce4277699f1a240f9891444b31374534c421，仅3scheduler文件改变，41文件与120数据均核验。新identity d7b5289412225a44da9b4bfcd726a6703da70a32a1df9e99888569033215cc03；完整回执见migration.json、initial-resume-readback.json。

普通CUDA preflight、真实初始checkpoint CPU读回、首162真实调度与finite检查全部passed。首120每表一次，normal index1,021；额外更新顺序/重复数与同一冻结排名完全一致，extra索引保留旧累计。此为运行与策略执行验收，不是拟合改善证据；尚未到新output首个固定Query评估。固定评估沿原训练行bank，每15,360actual步及最终点评估；每7,680actual步及停止点保存。组合编码、supervised_row25%、Query-only、AdamW lr1e-4，无reserved/final_test。

远端根/home/cms/tabu-v54-small-replay-v2-old120-20260922-0dfd1f3，输出runs/old120.small.replay-v2。旧v1及bridge保留为祖先证据并禁止自动恢复。所有stop、bridge、preflight、migration、launch stdout在receipts/。每台唯一训练，不自动追加重复/预算；W&B由主线程独立配置，不因监控重启训练。preparing模板仅保留准备证据，不用于实际执行。
