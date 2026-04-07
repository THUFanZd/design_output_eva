# Rubric

**目标**  
给定候选假说 `H`、其 claim 分解、以及一组 intervention evidence，输出：

1. `absolute score`
2. `pairwise preference`

## 1. 评分结构

```text
Total(H) =
  0.35 * Targeted_Logit_Capture
  0.20 * Polarity_Accuracy
  0.15 * Context_Selectivity
  0.15 * Generation_Agreement
  0.15 * Control_Robustness
  - Vagueness_Penalty
```

所有主分项归一到 `[0, 1]`。  
`Vagueness_Penalty` 取值 `[0, 0.15]`。

记号：

- `C(H)`: 假说 `H` 分解后的 claim 集合
- `T_pos(c), T_neg(c), T_decoy(c)`: claim `c` 对应的正例 / 负例 / decoy traces
- `Compile(c) -> (S_strong(c), S_weak(c))`: 把 claim 编译为 token 集
- `p_before(t), p_after(t)`: 由 `logits_before`, `logits_after` 经过 softmax 得到的 token 概率
- `Δp(t) = p_after(t) - p_before(t)`

如果 `Compile(c)` 失败，视为该 claim 不够具体：

- `Targeted_Logit_Capture(c) = 0`
- `Polarity_Accuracy(c) = 0`
- 同时在 `Vagueness_Penalty` 中额外加 `0.03`

对单个指标 `M`，统一采用：

```text
M(H) = mean_{c in C(H)} M(c)
```

## 2. 分项定义

### A. `Targeted_Logit_Capture`

**原材料**

1. claim `c`
2. `Compile(c)` 产出的两层 token 集：
   - `S_strong(c)`: 明确预期会变化的 tokens
   - `S_weak(c)`: 可能会变化但不那么核心的 tokens
3. `T_pos(c)` 中每条 trace 的 `logits_before`, `logits_after`

**评价对象**

评价的是：  
该 claim 预测的 target，是否真正吃到了干预后最主要的 logits 变化质量。

**计算步骤**

对每条正例 trace `τ in T_pos(c)`：

1. 计算 `Δp_τ(t) = p_after(t) - p_before(t)`
2. 如果 `c.polarity = promote`，定义：

```text
Gain_strong(τ,c) = Σ_{t in S_strong(c)} max(Δp_τ(t), 0)
Gain_weak(τ,c)   = Σ_{t in S_weak(c)}   max(Δp_τ(t), 0)
TopMass(τ,c)     = Σ_{t in TopK_pos(τ)} max(Δp_τ(t), 0)
```

其中 `TopK_pos(τ)` 是 `Δp_τ(t)` 上升最多的前 `K` 个 token，默认 `K = 20`。

3. 如果 `c.polarity = suppress`，把正向质量改成负向质量：

```text
Gain_strong(τ,c) = Σ_{t in S_strong(c)} max(-Δp_τ(t), 0)
Gain_weak(τ,c)   = Σ_{t in S_weak(c)}   max(-Δp_τ(t), 0)
TopMass(τ,c)     = Σ_{t in TopK_neg(τ)} max(-Δp_τ(t), 0)
```

4. 计算覆盖率和遗漏惩罚：

```text
Coverage(τ,c) =
  (Gain_strong(τ,c) + 0.5 * Gain_weak(τ,c)) / (TopMass(τ,c) + ε)

OmissionPenalty(τ,c) =
  1 -
  [Σ_{t in (S_strong(c) ∪ S_weak(c)) ∩ TopK(τ,c)} |Δp_τ(t)|] / (TopMass(τ,c) + ε)
```

5. 单条 trace 的分数：

```text
TLC(τ,c) = clip(Coverage(τ,c) - 0.5 * OmissionPenalty(τ,c), 0, 1)
```

6. 对 claim 聚合：

```text
Targeted_Logit_Capture(c) = mean_{τ in T_pos(c)} TLC(τ,c)
```

含义：

- 高分：主要变化确实集中在假说预测的 target 上
- 低分：要么没打中 target，要么 top changes 里有大量未被假说覆盖的 token

### B. `Polarity_Accuracy`

**原材料**

1. claim `c` 的 `polarity`
2. `Compile(c)` 产出的 `S_strong(c), S_weak(c)`
3. `T_pos(c)` 中每条 trace 的 `Δp`

**评价对象**

评价的是：  
假说对 target 的方向判断是否正确。

**计算步骤**

对每条正例 trace `τ in T_pos(c)`：

1. 定义 target 上的“顺向质量”和“反向质量”：

如果 `c.polarity = promote`：

```text
Correct(τ,c) =
  Σ_{t in S_strong(c)} max(Δp_τ(t), 0) +
  0.5 * Σ_{t in S_weak(c)} max(Δp_τ(t), 0)

Wrong(τ,c) =
  Σ_{t in S_strong(c)} max(-Δp_τ(t), 0) +
  0.5 * Σ_{t in S_weak(c)} max(-Δp_τ(t), 0)
```

如果 `c.polarity = suppress`，把正负号交换。

2. 单条 trace 的极性分数： # 所以是一个比较宽的指标，相当于是判断了方向准确性，而不是logits变化数值的准确性

```text
PA(τ,c) = Correct(τ,c) / (Correct(τ,c) + Wrong(τ,c) + ε)
```

3. 对 claim 聚合：

```text
Polarity_Accuracy(c) = mean_{τ in T_pos(c)} PA(τ,c)
```

含义：

- 高分：target 的变化方向和 claim 一致
- 低分：target 虽然变了，但方向经常相反

### C. `Context_Selectivity`

**原材料**

1. claim `c`
2. prompt buckets：
   - `T_pos(c)`
   - `T_neg(c)`
   - `T_decoy(c)`
3. 每条 trace 上已有的 `TLC(τ,c)` 与 `PA(τ,c)`

**评价对象**

评价的是：  
claim 预测的效应是否主要出现在“应出现”的上下文，而不是到处都成立。

**计算步骤**

1. 先定义单条 trace 的 effect score：

```text
Effect(τ,c) = TLC(τ,c) * PA(τ,c)
```

2. 计算三类 bucket 的均值：

```text
μ_pos(c)   = mean_{τ in T_pos(c)}   Effect(τ,c)
μ_neg(c)   = mean_{τ in T_neg(c)}   Effect(τ,c)
μ_decoy(c) = mean_{τ in T_decoy(c)} Effect(τ,c)
```

3. 计算 claim 的上下文选择性：

```text
Context_Selectivity(c) =
  clip( μ_pos(c) - 0.5 * [μ_neg(c) + μ_decoy(c)], 0, 1 )
```

含义：

- 高分：效应主要只在正例上下文出现
- 低分：负例、decoy 里也经常出现同样效应，说明假说过宽

### D. `Generation_Agreement`

**原材料**

1. claim `c`
2. `T_pos(c)` 中每条 trace 的：
   - `generation_before`
   - `generation_after`
3. 一个固定的 judge prompt 模板，模板中只允许围绕 claim-level 问题回答

**评价对象**

评价的是：  
在最终生成文本层面，输出变化是否支持该 claim。

**计算步骤**

对每条正例 trace `τ in T_pos(c)`：

1. 把以下对象喂给 judge：
   - claim `c`
   - `generation_before`
   - `generation_after`

2. 只允许 judge 回答 4 个问题：

- `Q1`: 目标方向是否出现？
- `Q2`: 极性是否匹配？
- `Q3`: 变化是否是主要变化，而不是边角变化？
- `Q4`: 是否出现在该 claim 声称应出现的上下文？

3. 每题四挡回答：

```text
support = 1
weak_support = 0.5
no_support = 0
contradict = -1
```

4. 题目权重：

```text
w = [0.35, 0.25, 0.20, 0.20]
```

5. 单条 trace 的 generation 分数：

```text
GA_raw(τ,c) = Σ_i w_i * answer_i(τ,c)
GA(τ,c)     = clip((GA_raw(τ,c) + 1) / 2, 0, 1)
```

6. 每条 comparison 默认重复 3 次 judge 调用，取均值：

```text
Generation_Agreement(c) = mean_{τ in T_pos(c)} mean_{r in 3 repeats} GA_r(τ,c)
```

含义：

- 高分：生成文本层面也支持该 claim
- 低分：即使 logits 有变化，最终输出没有体现或体现方向错误

### E. `Control_Robustness`

**原材料**

1. claim `c`
2. 三类 control traces：
   - `T_wrong_feature(c)`
   - `T_mismatch(c)`
   - `T_expected_negative(c)`
3. 在这些 controls 上重复计算的基础分

**评价对象**

评价的是：  
评分器在明显不该支持该 claim 的证据上，是否真的会低分。

**计算步骤**

1. 先定义不含 control 项的基础分：

```text
Base(τ,c) =
  0.45 * TLC(τ,c) +
  0.25 * PA(τ,c) +
  0.30 * GA(τ,c)
```

2. 在三类 control 集上求均值：

```text
μ_wrong(c)    = mean_{τ in T_wrong_feature(c)}    Base(τ,c)
μ_mismatch(c) = mean_{τ in T_mismatch(c)}         Base(τ,c)
μ_negctx(c)   = mean_{τ in T_expected_negative(c)} Base(τ,c)
```

3. claim 的 control robustness：

```text
Control_Robustness(c) =
  1 - (μ_wrong(c) + μ_mismatch(c) + μ_negctx(c)) / 3
```

再 clip 到 `[0,1]`。

含义：

- 高分：在 controls 上确实低分
- 低分：错配证据也能打出高分，说明评分器在奖励宽泛解释

### F. `Vagueness_Penalty`

**原材料**

1. `C(H)` 的 claim 数量
2. `Compile(c)` 是否成功
3. claim 文本本身

**评价对象**

评价的是：  
假说是否写得过宽、过滑、过难被证伪。

**计算步骤**

对整个假说 `H` 做确定性检查：

```text
VP(H) =
  0.05 * I(|C(H)| > 3)
  + 0.03 * I(exists broad target wording)
  + 0.03 * I(exists compile failure)
  + 0.02 * I(exists diffuse scope)
  + 0.02 * I(exists vague context condition)
```

最后：

```text
Vagueness_Penalty(H) = clip(VP(H), 0, 0.15)
```

其中：

- `broad target wording`: 如 “related to many themes”, “general content about ...”
- `vague context condition`: 如 “sometimes”, “in some cases”, 但没有具体条件

含义：

- 这是一个结构惩罚，不依赖 traces
- 目的是压制“写得越滑越容易得高分”的假说

## 3. Pairwise Preference

**原材料**

1. 同一 feature、同一 evidence set
2. 两个候选假说 `H_a`, `H_b`
3. 两者的：
   - `Total(H)`
   - 核心子分项
   - `Control_Robustness(H)`

**评价对象**

评价的是：  
在完全相同证据下，哪个假说更能解释该 feature 的 output-side effect。

**计算步骤**

`H_a` 胜出，当且仅当：

1. `Total(H_a) - Total(H_b) >= margin`
2. `Control_Robustness(H_a)` 不显著差于 `H_b`
3. 在核心三项里至少赢两项：
   - `Targeted_Logit_Capture`
   - `Polarity_Accuracy`
   - `Context_Selectivity`

否则：

- 若分差很小，记 `tie`
- 若控制项明显失败，直接判负

其中默认：

```text
margin = 0.05
```

并且：

- 如果 `Control_Robustness(H_a) < 0.4`，则 `H_a` 不允许胜出
- 如果两者 `Total` 很近，但一方在核心三项中赢两项以上，则仍可判其胜

## 4. Judge 使用规范

1. 不允许回到 input-side 解释
2. 不允许给自由文本总评
3. 只能围绕 claims 和 generation pair 回答
4. 同一 comparison 至少重复 2-3 次，检查稳定性

## 5. 最小输出格式

对单个 `H`：

```text
{
  total_score,
  sub_scores,
  vagueness_penalty,
  claim_level_decisions
}
```

对 `(H_a, H_b)`：

```text
{
  winner: H_a | H_b | tie,
  margin,
  reason: [top 2 decisive sub-scores]
}
```

## 6. 为什么这样设计

- **logits 主导**：让评分真正贴近干预后的因果变化
- **generation 补充**：保留高层行为变化，但不让 judge 接管全局
- **controls 约束**：强迫评分器拒绝宽泛假说
- **pairwise 输出**：直接服务于 hypothesis refinement
