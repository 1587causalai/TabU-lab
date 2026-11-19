# 表格基础模型与评测参考

从模型所属版本的论文、代码和权重读起。比较时同时记录可见样本、数据划分、训练预算和输出指标。[TabU 研究问题](../../research-questions/README.md)

| 参考对象 | 公开入口 | 与 TabU 的关系 |
| --- | --- | --- |
| TabPFN | [历史阅读卡](tabpfn.md) | 合成先验、Support 与 Query 的条件化、表内推理。 |
| LimiX | [历史阅读卡](limix.md) | 统一预测和补全任务、缺失信息与特征交互。 |
| TabICLv2 | [论文第二版](https://arxiv.org/abs/2602.11139v2) · [官方代码与权重入口](https://github.com/soda-inria/tabicl) | V10.1 同任务量比较的参考。作者报告沿原论文查阅，本项目的证据沿[匹配任务报告](https://research.wehub.us/tabu-lab/experiments/research-update-20261029/)查阅。 |
| NVIDIA Kumo Tabular Small | [官方代码](https://github.com/NVIDIA/structured-data-models) · [模型卡](https://huggingface.co/nvidia/Kumo-Tabular) | V10.1 比较参照和 CellRidge-1024 的父模型。衍生版沿[发布清单与本地评测](https://github.com/1587causalai/kumo-small-cellridge-1024)查阅。 |
| TabArena | [基准仓库与 protocol](https://github.com/autogluon/tabarena) | 定义独立的任务、划分、指标与提交。TabU 的首次提交和本地 Lite 复现材料从[公开评测记录](../../reports/README.md#published-results)进入。 |

## 阅读时核对什么

- 模型学习哪类任务，数据先验如何构造，待预测目标何时可见。
- 表内上下文包含哪些行和特征，模型怎样区分缺失和已知值。
- 对比涉及已见表的留出行、未见表还是新的生成机制，选择检查点时看过哪些数据。
- 分类和回归分别用了什么指标，模型及基线的计算预算如何匹配。
