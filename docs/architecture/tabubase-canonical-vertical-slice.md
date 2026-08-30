# TabUBase 治理第一站：把“它是谁”闭合起来

这一站先不问模型效果。它只回答一个更底层的问题：

> 我们写在 ModelSpec 里的 TabUBase，是否就是 builder 实际构建出来的那个模型？

闭环如下：

```text
cell 是 Unit
  → tabu.cell.base@0.2.0 ModelSpec
  → canonical builder
  → tokenizer / dynamics / readout / supervision route
  → Step 1 正确性 + Step 2 单轴替换验证
  → local_unissued evidence
```

## 这次具体做了什么

1. 保持 `tabu.cell.base@0.2.0` 和现有 ModelSpec 字节不变；
2. 增加一个 resolved component binding，把 ModelSpec、profile 和实际组件绑定为同一个可哈希身份；
3. 每次 public build 返回前，都核对实际 tokenizer、dynamics、readout、supervision route、cell-Unit 语义和 truth sidecar boundary；
4. 把组件正确性与单轴可替换性结果写成严格的 `local_unissued` 对象。

因此，YAML 不是装饰文档，builder 也不能只返回“某个能跑的模型”。两者必须落到同一组实际组件，并共享同一个 ModelSpec 哈希。

## 为什么没有新建可任意注入的 component registry

`0.2.0` 目前声明的是一组内置组件轴及其有界替代项。开放任意 callable 注入会新增公共扩展协议，也会改变错误隔离、checkpoint identity 和证据解释方式。这不是第一站必须做的事。

当前先证明三件事：

- canonical builder 不能悄悄偏离 ModelSpec；
- 当前 ModelSpec 已声明的 tokenizer、readout alternatives 可以一次只替换一个轴；
- 替换后公开 forward 接口保持稳定，同时 variant identity 必须变化。

MAB dynamics 仍可作为代码级 non-O ablation 构建，但在 ModelSpec 正式声明前，验证结果必须保持 `fail`，不能被称为 declared component substitution。

未来若开放第三方组件，应建立新版本 component identity contract，而不是把它偷偷塞进 `0.2.0`。

## 证据边界

本阶段结果固定标记为 `local_unissued`。它只支持“组件身份闭合”和“有界可组合性”结论，不是正式 receipt，也不支持训练拟合、真实数据预测、frozen ICL、微调提升或 foundation-model claim。

定向检查：

```bash
PYTHONPATH=src python -m pytest \
  tests/contract/test_tabubase_vertical_slice.py \
  tests/contract/test_tabubase_composability.py \
  tests/contract/test_tabubase_v020_contract.py -q
```
