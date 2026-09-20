# 模型无关 curriculum 最小实现验证

2026-09-19，在 saved `tabu-lab` 的 `main` 基线 `6293e95` 上完成。
本轮是本地未提交的 additive 实现，没有启动训练、移动数据或改写历史记录。

交付为 [6 页独立规范](../../design/pretraining-curriculum.tex)
（[PDF](../../design/pretraining-curriculum.pdf)）、
[Python 调用说明](../../tutorials/model-independent-curriculum.md)，以及
`src/tabu_lab/curriculum`。文档将不变语义、可修订经验政策、可替换实现分开；
四参考阶段不构成强制 pipeline。结果 schema、自动质量 gate 与通用 trainer 仍是设计边界，
本轮只实现 metadata/selection 与 episode bridge。

验证记录见 [verification.json](verification.json)：

- 新增 25 个 metadata/episode 测试；与既有 V5.3 data/protocol 范围合计 **89 passed**。
  包含 truth intervention、reserved 拒绝路径、别名/重排/重新划分、真实来源 record 认证、
  自定义探针与默认覆盖、import 不加载 Torch。没有 optimizer update。
- Ruff 和 `git diff --check` 通过。独立审查找到的未验证 world 声明与 adapter 源码身份缺口
  已修复并复核；不是只依据实现者自报通过。
- 原位 old120/new120 共 240 个文件 SHA 匹配，跨组 values 内容分离通过。
  当前 importer 认证 96 个 world_hash；144 个未取得这一层认证。
  这不否定来源的其他机制身份，也不等于完整 world separation。
- 既有 V5.3 源码摘要仍为
  `17cedb6a5058be87444c625a33b382dfe3661bd5765a46b32a3b95410030b12d`，
  与原 runner 验证记录一致；本轮未更改它的 resume/source identity。
- XeLaTeX/latexmk 编译成功，6 页逐页渲染与视觉检查通过；无溢出、缺字或未定义引用。
  系统 Poppler 中文 CMap 不完整，视觉验收使用 bundled PDFium；有 Fandol script 提示。

历史事实另经本地回执核对：V4 fit8 为八个独立模型，72/72 artifact hashes 匹配；
V4 multi120 的统计和 real12 reserved 宏平均 R² 从逐表结果复算一致。
历史 fit8/old120/new120/real12 并非已验证的单一 checkpoint 链；V4/V5 的 mask、
objective、训练预算不同。新的 V5.3 八表远端运行状态不在本轮验证范围。

验证命令：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/unit/test_curriculum_catalog.py tests/unit/curriculum_v53/test_data.py tests/unit/curriculum_v53/test_protocol.py
.venv/bin/python -m ruff check src/tabu_lab/curriculum tests/unit/test_curriculum_catalog.py examples/curriculum_catalog.py
git diff --check
```

TeX 源码为主要交付；PDF 遵循仓库既有 ignore 规则，作为本地可读产物保留。
本轮没有修改相邻模型设计、既有训练入口或保护的 canonical TeX。
