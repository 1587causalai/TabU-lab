# TabU V5.4 — 统一列身份组合编码

这是 owner LaTeX 设计包在 `TabU-lab` 中的归档副本；下文提到的其他版本、设计笔记与研究提案路径均指原 `TabU/latex` 工作区，不是此 Git 仓库的相邻目录。本包 TeX 与 owner 冻结源完全相同；此 README 仅将跨仓库链接改成原路径说明。

2026-09-22 起本包冻结。后续改写在原工作区 `TabU/latex/model-factory/table-restoration/TabU_V5p5/`，不在本目录改写。

这是第五代内的 **V5.4**：先整份复制原工作区 `TabU/latex/model-factory/table-restoration/TabU_V5p3_Refined_Complete/` 主 TeX，再改 value 编码，把 Unit Transformer 收进默认前向，并把 Unit/Feature 的 \(K\) 个 subtokens 收成可选扩展（默认 \(K=1\)）。V5.3 整包冻结，不在本目录改写。

本包**不是** `table-restoration` 的 canonical 主稿。唯一主稿仍是原工作区 `TabU/latex/model-factory/table-restoration/end-to-end-design.tex`。

设计笔记：原工作区 `TabU/brainstorm/unified-column-value-compositional-encoding.md`（2026-09-21）。

## 相对 V5.3 的最小改动

V5.3 默认模式 C 里，numeric / ordinal 已经是共享原点加仿射直线；nominal 仍是每类一个独立 \(128/8\) 码。V5.4 把三类收成同一外层：

\[
e_a(x)=q_a+v_a(x),\qquad
v_a=\begin{cases}
z_a(x)\,b_a & \text{numeric}\\
r_a(x)\,b_a & \text{ordinal}\\
b_{a,x} & \text{nominal}
\end{cases}
\]

默认基础向量一律从 \(\mathcal H_4\)（\(128/4\)）抽取。列身份 \(q_a\) 是 episode 内随机标识。nominal 不再等范数，解码改为 \(\arg\max_c(\widehat e-q_a)^\top b_{a,c}\)。独立 \(128/8\) 身份码降为附录候选。

第二项默认不同：V5.3 ModelSpec 写「最小 \(L_U=0\)」，Unit Transformer 是可选组件。V5.4 默认开启 Unit Transformer；命名默认档 **V5 Small** 取 \(L_U=3\)。**V5 Nano** 是另一档：2 层 backbone、无 Unit Transformer。把 Small 的 Unit 关掉不等于 Nano。

第三项接口不同：V5.3 正文只按单 token 书写。V5.4 允许把每个 Unit 与每个 Feature 扩成 \(K\) 个同宽 subtokens，**默认仍 \(K=1\)**。Cell 不随 \(K\) 复制。\(K>1\) 还不是完整实例，须另写槽位编译、供源资格、Unit Transformer 入口和匹配读出。

列共享 LL、Query 系数 1、restore 系数 0、OTransformer 算子、训练准入、损伤与多轮候选保持原定义。episode 的当前优先级见下节。既有内部标签仍用 `v53:` 前缀，避免无谓改引用。

## Episode 与竞争选项的当前定位

当前默认 **supervised-row Query**：只遮去所选 Query 行的目标标签，特征观测仍可见。**random-cell 是第一备选**，也是训练稳定、实验经验充分后希望采用的默认方向；当前尚未切换，不预设自动切换阈值或固定日程。该优先级适用于当前基础拟合课程，不按合成/真实数据来源自动分流。

episode 协议决定哪些 Cell 成为 Query；Query-only loss 决定哪些误差参与训练。两种 episode 当前都使用 Query 系数 1、restore 系数 0。附录新增「Episode 的当前默认、第一备选与未来方向」，原有「合成 random-cell、真实 supervised-row」课程保留为历史配置，不覆盖当前默认。

许多选项仍相互竞争、同等重要，默认只是当前执行与复现起点。**V5 Nano 是必须纳入基础拟合实验的竞争方案**；Small 的默认地位不表示它已胜出。研究重要性、定义完整性和实验验证状态分别说明，尚未闭合的接口仍须补齐定义。

## 第五代尺寸档

第五代尺寸档独立定义，本稿默认前向仍按 **V5 Small**。五档的尺寸字段如下；定义完整不代表已完成训练验证。

| 档位 | Backbone 层数 | 宽度 | Heads | FFN | Unit 层数 | Inducing 槽数 |
|---|---:|---:|---:|---:|---:|---:|
| V5 Nano | 2 | 128 | 4 | 256 | 0 | 256 |
| V5 Small | 3 | 128 | 8 | 256 | 3 | 256 |
| V5 Medium | 6 | 192 | 8 | 384 | 3 | 256 |
| V5 Standard | 12 | 256 | 8 | 512 | 6 | 256 |
| V5 Large | 24 | 384 | 12 | 768 | 6 | 256 |

从 Small 起，backbone 深度逐档翻倍、宽度逐步增加，FFN 保持为宽度的 2 倍；Unit 深度为 3 / 3 / 6 / 6，控制独立行间精炼的额外开销。各档保持 $S=256$、$K=1$、原始 codec 宽度 $p=128$；共享输入映射为 $W_{\rm enc}\in\mathbb R^{d\times128}$，默认 LL 特征宽度为 $d_R=d$。Medium 的 6 / 192 / 384 是按此递增原则显式选定，虽与 TAR medium 的三个字段相同，但不继承 TAR 配置及实验结论。

## 阅读

- `TabU_V5p4_Unified_Composition.tex`：本包主源，自包含编译。
- `TabU_V5p4_Unified_Composition.pdf`：与主源配套的编译稿；附录「第五代模型尺寸档」含全部五档。
- 冻结基线：原工作区 `TabU/latex/model-factory/table-restoration/TabU_V5p3_Refined_Complete/README.md` 与同目录主 TeX。
- 最小 nominal 对照实验身份：原工作区 `TabU/research-proposals/small128-unit3-nominal44-dustinstudio-20260921/README.md`。该实验效果待验证；dgx2 Small-128＋3 Unit 仍是旧 \(128/8\) nominal，不能当作本编码的效果证据。

## 编译

需要包含 XeLaTeX、ctex、Fandol、Latin Modern、TikZ、tcolorbox 的 TeX 发行版。无需私人字体、额外图文件或 `.bib`。

```sh
latexmk -xelatex -synctex=1 -interaction=nonstopmode -halt-on-error TabU_V5p4_Unified_Composition.tex
```

也可以连续运行 XeLaTeX 直到交叉引用稳定。不要只编译一次便假定所有引用已有编号。

## 证据边界

本稿是设计文档。统一公式、几何恒等式和解码规则以主 TeX 为准。

- 未把主代码默认值改为 \(q_a+b_{a,c}\)，也未把训练代码的 `unit_layers` 默认从 0 改为 3，也未实现 \(K>1\) subtokens。
- 未把 dustinstudio 预检或任何未完成拟合写成性能结论。
- 未重跑 V5.3 的算子检查；那些报告仍只覆盖 V5.3 身份。
- 编码范数和类别间距离会变，不能单凭编码 MSE 下降判断拟合改善。

## 修订记录

| 日期 | 内容 |
|---|---|
| 2026-09-21 | 明确 episode 当前默认 supervised-row、第一备选 random-cell，以及条件成熟后希望切换默认的方向；区分 Query-only loss 与采样协议，注明历史分流不覆盖当前选择，并将 Nano 明确列为必须实验验证的同等重要竞争方案。 |
| 2026-09-21 | 整份复制 V5.3 主 TeX，仅将默认编码改为 \(e_a=q_a+v_a(x)\)。 |
| 2026-09-21 | 登记第五代尺寸档：V5 Small 为 3/128/8/256 + 3 层 Unit Transformer；不沿用 TAR small 的 96/192。 |
| 2026-09-21 | 尺寸档独立成附录：一览表、V5 Small 闭合定义、Medium/Standard/Large 待写小节。 |
| 2026-09-21 | 默认开启 Unit Transformer（V5 Small 为 \(L_U=3\)）；V5.3 的 \(L_U=0\) 降为可选旁路。 |
| 2026-09-21 | 登记 Unit/Feature 可选 \(K\) 个 subtokens，默认 \(K=1\)；\(K>1\) 尚未闭合。 |
| 2026-09-21 | 登记 V5 Nano：2 层 / 128 / 4 heads / FFN 256 / \(L_U=0\)；与 Small 的 \(L_U=0\) 消融不是同一档。 |
| 2026-09-21 | 补齐 Medium / Standard / Large 六项尺寸字段及递增依据：6/192/8/384/3、12/256/8/512/6、24/384/12/768/6，均取 $S=256$、$K=1$；默认仍为 Small，新增档位尚待训练验证。 |
