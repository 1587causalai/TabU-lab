# TabUBase 第 2 阶：组件可组合性门

这一阶回答的不是“模型效果好不好”，而是一个更基础的问题：

> 当我们只替换 TabUBase 的一个组件时，其他组件和公开 forward 接口能否保持不变，并且模型身份会不会如实变化？

## 当前验证的三个替换轴

| 轴 | 基准组件 | 替代组件 | 必须保持不变 |
|---|---|---|---|
| Tokenizer | `cell-tokenizer.v1` | `cell-tokenizer.v2` | dynamics、readout、profile、输出结构 |
| Dynamics | `cell_unit_three_omab` | `cell_unit_three_mab` | tokenizer、readout、profile、输出结构 |
| Readout | `same_column.local_linear` | `same_column.nadaraya_watson` | tokenizer、dynamics、profile、输出结构 |

每次检查只允许一个轴变化。若一次同时改变两个轴，即使 forward 可以运行，也不能通过这一门，因为那样无法判断接口是否真正解耦。

## 怎样才算通过

一次替换同时满足以下条件才记为 `pass`：

1. 只有声明的组件轴发生变化，其他 semantic config 必须完全一致；
2. prediction entries、status、tensor shapes、auxiliary shapes 和 trace 有无保持一致；
3. `variant_ref.semantic_hash` 必须变化，避免新组件伪装成旧模型身份。

`src/tabu_lab/verification/composability.py` 只读取已经存在的模型与预测契约，不修改 `tabu.cell.base@0.2.0` 的语义、ModelSpec 或 checkpoint identity。

## “可扩展、可生长”目前具体意味着什么

- 内部已有三个有名字、可单独切换、受模型身份约束的组件轴；
- 模型级 `BuilderRegistry` 可以增加新的 namespaced builder；
- canonical `tabu.cell.base` builder 受保护，扩展不能覆盖它冒充基准实现。

这一结果是**有边界的可组合性通过**，不是任意第三方组件注入能力。未来若需要开放自定义 tokenizer/dynamics/readout factory，应使用新的、可验证的 component identity contract，而不是向 `0.2.0` 构造器塞入任意 callable。

## 不属于这一阶的结论

本门不比较预测数值优劣，也不证明训练拟合、真实数据效果、frozen ICL、微调提升或 foundation-model 能力。那些分别属于后续第 3–6 阶。

运行：

```bash
uv run pytest -q tests/contract/test_tabubase_composability.py
```
