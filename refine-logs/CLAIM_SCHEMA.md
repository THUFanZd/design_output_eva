# Claim Schema

**目标**  
把自由文本假说 `H` 压缩成少量可检验的 `atomic claims`，让后续评分器可审计、可比较。

## 1. 原子 claim 模板

每个 atomic claim 只表达一个输出效应，格式如下：

```text
Claim = {
  target: ...
  polarity: promote | suppress
  scope: focused | moderate | diffuse
  context_condition: always | positive_only | conditional
  manifestation: logits | generation | both
}
```

## 2. 字段定义

### `target`

表示该 feature 影响什么输出对象。只允许以下 4 类：

1. `token_family`
2. `semantic_direction`
3. `style_or_function`
4. `task_behavior`

要求：

- `target` 必须具体，不能只写“相关内容”“某种主题”
- 如果是宽主题，必须进一步压缩到 token family / semantic direction / style function

### `polarity`

- `promote`: 干预后目标更强
- `suppress`: 干预后目标更弱

不允许在单个 claim 中写“既增强 A 又抑制 B”。这种情况必须拆成两个 claims。

### `scope`

- `focused`: 影响集中在较窄对象
- `moderate`: 影响存在，但会带少量邻近变化
- `diffuse`: 假说明确声称该效应较弥散

默认不鼓励 `diffuse`。如果 claim 写得很宽，后续会触发 vagueness penalty。

### `context_condition`

- `always`: 大多数相关 prompt 都应出现
- `positive_only`: 只在一类 prompt 中应出现
- `conditional`: 只在明确条件下应出现，必须补一行短描述

### `manifestation`

- `logits`: 效应主要应出现在 logits shift
- `generation`: 效应主要应出现在最终生成
- `both`: 两层都应能看到

## 3. 假说分解规则

把自由文本 `H` 分解成 `C = {c_1, ..., c_k}` 时，遵守：

1. `k <= 3`
2. 每个 claim 只能有一个 `target`
3. 每个 claim 必须显式给出 `polarity`
4. 如果 `context_condition = conditional`，条件必须短而具体
5. 不允许把 input-side 内容混进 target

## 4. 对照假说生成约束

为同一 feature 构造 hard negatives 时，只允许以下 5 类变体：

1. `broad`
2. `partial`
3. `wrong_polarity`
4. `adjacent`
5. `correlational`

生成约束：

1. 每次只改一个主维度
2. 长度与原假说保持接近
3. 不允许明显稻草人
4. 生成后过 validator，检查：
   - 是否仍是 output-side 句子
   - 是否只偏移一个维度
   - 是否没有显式自相矛盾

## 5. 为什么这样设计

- 不先把 `H` 压成原子 claims，judge 容易奖励套话
- 不限制 claim 数量，假说越长越容易“总有一点说中”
- 不控制 hard negatives，pairwise 比较就会失真
