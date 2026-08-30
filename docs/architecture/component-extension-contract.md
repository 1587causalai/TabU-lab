# TabUBase 治理第二站：让组件可以安全生长

第一站证明了“YAML 声明的 TabUBase，就是 builder 实际构建的模型”。第二站继续回答：

> 如果以后增加一个新组件，怎样保证它不是匿名 callable，也不会冒充 canonical TabUBase？

现在的扩展链是：

```text
ComponentSpec
  → protected ComponentRegistry
  → typed ComponentRef
  → TabUBaseComponentManifest
  → builder resolves concrete modules
  → composition hash
  → variant / checkpoint / trace / local verification
```

## ComponentSpec 记录什么

每个组件必须声明：

- namespaced component ID 与 semantic version；
- `tokenizer`、`dynamics` 或 `readout` 角色；
- 对应角色的固定 interface ID；
- concrete `nn.Module` 类型、factory 及两者的直接源码 SHA-256；
- fixed config、允许开放的 config 字段、兼容模型；
- `canonical` 或 `experimental` maturity。

registry 注册时会重新读取 concrete runtime type、factory 源码，以及 factory 直接引用的 concrete types、常量、defaults 与 bytecode 身份。factory 不得捕获 nonlocal closure，不得通过 helper function 或 module object 间接取得行为。构建前会再次核对这些依赖，并要求 factory 返回 exact concrete type。ID/version 重复、角色接口不符、源码身份漂移、factory 返回错误类型都会 fail closed。

这里的 component source hash 绑定直接 runtime class 与 factory；正式 run provenance 仍须使用仓库 commit/source-tree identity，不能把局部哈希当成完整源码 closure。

## canonical 与 experimental 的边界

canonical component entries 受保护，不能覆盖。builder 会要求传入 registry 完整继承内置 canonical root；fresh registry 不能用相同 ref 自封 canonical。公开注册面只接受 namespaced `experimental` entries，`model_spec_declared` 只能来自 exact canonical spec hash。

builder 接受的是 typed manifest 和 registry，不接受临时 callable。只要显式 manifest 出现，旧的 `numeric_terminal`、`nominal_tokenizer` 等选择参数就不能同时出现，避免双重配置权威。

ComponentRef、ComponentSpec 和 resolved composition 会递归复制并冻结嵌套 config。inspection 与 verification 重新执行 registry/spec → exact runtime type/source 的闭合，并再次核对 prediction/trace 中的 component hashes；attached metadata 本身不是可信依据。

为了保持已有 `tabu.cell.base@0.2.0` checkpoint/variant identity，默认 legacy build path 不改变。任何显式 composition manifest 都把以下身份写入运行边界；experimental manifest 还明确标出替换轴：

- `component_manifest_hash`；
- `component_composition_hash`；
- 每个角色的 `component_spec_hash`；
- `experimental_component_axes`。

这些字段进入 variant semantic config、checkpoint identity、forward trace/metadata 和 `local_unissued` extension verification。扩展不能因此升级成 ModelSpec alternative、formal receipt 或 accepted claim。

## 当前最小证明

定向测试注册一个 `PairUnitReadout` subclass，只替换 readout 轴，并验证：

1. 实际构建得到该 subclass；
2. 其他 component axes 与 forward interface 不变；
3. variant 与 composition identity 变化；
4. prediction 分别绑定对应模型，并使用同一 input hash；
5. 结果只记作 `component_extension / local_unissued`。

MAB 仍是 `code_only_non_o_ablation`，不在本次 registry 中获得 experimental ComponentSpec，也不提升为 ModelSpec alternative。

定向检查：

```bash
PYTHONPATH=src python -m pytest \
  tests/contract/test_component_extension_contract.py \
  tests/contract/test_tabubase_vertical_slice.py \
  tests/contract/test_tabubase_composability.py \
  tests/contract/test_tabubase_v020_contract.py -q
```
