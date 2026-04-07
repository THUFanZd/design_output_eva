# Evidence Format

**目标**  
统一记录一个候选假说 `H` 在某个 feature 上的可用证据，供 logits 模块、generation 模块和 controls 共用。

## 1. 基本单位

一个 evaluation item 定义为：

```text
E = (feature_id, hypothesis_id, prompt_id, trace)
```

其中 `trace` 记录该 prompt 在无干预和有干预下的输出差异。

## 2. Trace 字段

最小字段集：

```text
trace = {
  prompt_text,
  intervention_spec,
  generation_before,
  generation_after,
  logits_before,
  logits_after,
  meta
}
```

## 3. 字段说明

### `intervention_spec`

至少记录：

- feature id
- intervention type：boost / suppress / replace
- intervention strength
- intervention position：next-token only / multi-step / full decode

### `generation_before`, `generation_after`

同一 prompt、同一解码配置下：

- `generation_before`: 不干预生成
- `generation_after`: 干预后生成

这两个对象组成 `generation pair`。

### `logits_before`, `logits_after`

建议记录：

1. 干预位置的 full logits，或至少可恢复指定 token indices 的 logits
2. top-k 上升 tokens
3. top-k 下降 tokens

### `meta`

至少包括：

- decoding config
- prompt split id
- seed
- 是否属于 control item

## 4. Prompt Bucket

每个 prompt 相对某个 hypothesis 要落在 3 类之一：

1. `expected_positive`
2. `expected_negative`
3. `decoy`

注意：这个 bucket 是相对 `H` 定义的，不是 feature 的绝对属性。

## 5. Controls

评分时必须额外构造 3 类控制证据：

### `wrong_feature_control`

把 `H` 拿去解释另一个 feature 的 traces。

### `mismatched_trace_control`

把 `H` 和乱配的 prompt/trace 配起来。

### `expected_negative_context_control`

在按 `H` 不应出现效应的 prompt 上做正常评分。

## 6. 建议的数据组织

```text
feature_id/
  hypotheses.jsonl
  traces.jsonl
  pairwise_items.jsonl
```

其中：

- `hypotheses.jsonl`: 原假说与 hard negatives
- `traces.jsonl`: 正常 traces 与 control traces
- `pairwise_items.jsonl`: `(H_a, H_b, trace_set)` 比较任务

## 7. 为什么这样设计

- 同一份 evidence 要同时喂给 absolute score 和 pairwise score
- 不把 controls 作为同等格式数据保存，后面就容易不可复现
- 不显式记录 prompt bucket，`context selectivity` 就无法定义
