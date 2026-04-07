# Rubric

**目标**  # note: 考虑这个问题的时候，不要考虑workflow的设计（顺便，workflow的设计涉及baseline的选取）
给定候选假说 `H`、其 claim 分解、以及一组 intervention evidence，输出：

1. `absolute score`
2. `pairwise preference`

## 1. 评分结构

```text
CoreScore(H) =
  0.40 * Targeted_Logit_Capture
  0.25 * Polarity_Accuracy
  0.20 * Context_Selectivity
  0.15 * Generation_Agreement

Total(H) = CoreScore(H) - Vagueness_Penalty(H)

ValidityGate(H) = I(Control_Robustness(H) >= τ_ctrl)
```

所有主分项归一到 `[0, 1]`。  
`Vagueness_Penalty` 取值 `[0, 0.15]`。

记号：

- `C(H)`: 假说 `H` 分解后的 claim 集合
- `T_pos(c), T_neg(c), T_decoy(c)`: claim `c` 对应的正例 / 负例 / decoy traces
- `Compile(c) -> (S_strong(c), S_weak(c))`: 把 claim 编译为 token 集
- `p_before(t), p_after(t)`: 由 `logits_before`, `logits_after` 经过 softmax 得到的 token 概率
- `Δp(t) = p_after(t) - p_before(t)`

Response on `C(H)`:

- 评分器不应直接吃自由文本假说。假说生成阶段就应同时输出：
  - `summary`: 1 句自然语言摘要，供人读
  - `claims[]`: 不超过 3 条 atomic claims，供评分器使用
- 评分阶段只对 `claims[]` 打分，`summary` 不参与计算。
- 只有在 claim schema 固定、且“一条 claim 只表达一个输出效应”时，`|C(H)|` 才能被解释为“结构性宽泛”。

Response on `T_pos(c), T_neg(c), T_decoy(c)`:

- 这三类不是 feature 的天然标签，而是相对某个 claim `c` 构造的 prompt buckets。
- `T_pos(c)`: 按 `c` 的内容，干预后应出现该效应的 traces。
- `T_neg(c)`: 按 `c` 的内容，干预后不应出现该效应的 traces。
- `T_decoy(c)`: 语义相邻、容易混淆、但不应被 `c` 解释的 traces。
- `decoy` 不是随机负例，而是“近邻难负例”，专门测 specificity。
- 例：如果 `c` 说“feature 会促进 war-related tokens”，则：
  - `T_pos(c)`: 地缘冲突/战争叙述 prompt
  - `T_neg(c)`: 菜谱、天气、数学题 prompt
  - `T_decoy(c)`: bidding / competition / rivalry prompt

Response on `Compile(c)`:

- `Compile(c)` 不能只靠一次 LLM 直出，必须是三步流程：
  1. `LLM proposal`: 根据 claim 产出少量 `S_strong` seed tokens 和更多 `S_weak` 候选
  2. `expansion`: 用词形变化、别名、词典和 embedding 邻近把弱候选扩充完整
  3. `validation`: 去掉停用词、高频泛词、和 target 不一致的词
- 推荐输出格式：

```text
Compile(c) = {
  S_strong: canonical indicators,
  S_weak: paraphrases / inflections / near indicators,
  compile_status: success | weak_success | fail
}
```

- 如果 `target` 是纯风格/行为而不易编译为 token 集，则：
  - 可以只做 `weak_success`
  - 或把该 claim 的 `manifestation` 设为 `generation`
  - 这类 claim 在 logits 模块上天然权重较低

具体建议写法：

`Compile(c)` 采用 “LLM seed proposal -> lexical / embedding expansion -> validator filtering” 三步流程；`S_strong` 表示 claim 的 canonical token indicators，`S_weak` 表示与 target 语义一致但不是核心标记的辅助 tokens。

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

  # note: 这里有一个问题，就是不能logits一上升，就认为促进。1.上升得有个阈值吧，top我理解更像是看，你的目标是否涵盖了一个更大的，可能涉及激活语义的token logits的变化？这样来说，topk不是在定义哪些token代表了这个语义，而是代表这个语义的tokens的超集 2.诶，这不就可以把多步logits拿出来，看看在多步生成中，哪些token的logits是统一上升的？这些是我认为真正代表这个特征的tokens

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

Response:

- 这里故意使用 `max(Δp, 0)`，因为 `Targeted_Logit_Capture` 只测“有没有打中预测 target”。
- 与 claim 极性相反的变化不在这一项里惩罚，而交给 `Polarity_Accuracy` 统一处理。
- 这样做的原因是避免双重计罚：如果在 `TLC` 和 `PA` 中都对反向变化惩罚，两个指标会强耦合，不利于解释各分项的含义。
- 所以：
  - `TLC` = 命中多少“对的目标”
  - `PA` = 方向是否对

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

2. 单条 trace 的极性分数：

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
- `Q3`: claim 相关变化是否属于 before/after 差异中前两项最显著变化之一？
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
w = [0.40, 0.30, 0.15, 0.15]
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

Response on judge reliability:

- 单独把 LLM judge 当主评分器，不可信；这也是为什么这里它只能作为辅助项。
- 这一项成立的前提有四个：
  1. judge 只回答固定四问，不给自由总评
  2. 同一 item 重复 2 到 3 次取均值
  3. 最终不让 GA 主导总分
  4. 在 20 到 30 个 feature 上做 small human pairwise calibration
- `Q3` 之所以改写成“是否属于前两项最显著变化之一”，就是为了减少“主要变化”这种空泛表述。

Response on generation weights:

- `Q1` 权重最高，因为如果 target 方向根本没出现，claim 就不成立。
- `Q2` 第二，因为方向错了，claim 仍然不成立。
- `Q3` 和 `Q4` 次之，因为它们是“支持强度”和“适用上下文”的补充判断，更容易受 judge 主观性影响。
- 这组权重仍然只是默认值，最终应在开发集上以 human pairwise agreement 最大化为目标做一次性校准并冻结。

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

Response on control construction:

- `wrong-feature control`:
  - 从同层、同 SAE、干预协议相同的 feature 中选取
  - 要求与 `c` 的 target 明显不匹配
  - 为避免过于容易，优先选激活频率和稀疏度相近的 feature
- `mismatched-trace control`:
  - 使用同一 feature、同一 intervention 配置
  - 但把 `H` 与别的 prompt 的 trace 乱配
  - 目的是打破“正确 prompt-feature-trace 配对”
- `expected-negative-context control`:
  - 在构造 `T_neg(c)` 时同步得到
  - 即 claim 明确预测“不应出现该效应”的 prompt

Response on the role of `Control_Robustness`:

- 同意，这个指标更大的作用确实是在验证 explanation-scoring pipeline 的有效性。
- 但对单个假说来说，它也测该假说是否过宽，因此仍然有 per-hypothesis 价值。
- 为避免误解，本版把它从 `CoreScore` 中拿掉，改成 `ValidityGate`：
  - 若 `Control_Robustness(H) < τ_ctrl`，则该假说分数视为无效，或 pairwise 中不允许胜出。
- 默认：

```text
τ_ctrl = 0.6
```

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

Response:

- 这个 comment 是对的：`Vagueness_Penalty` 的有效前提，是 claim schema 固定、claim 切分唯一。
- 因此这项不能在“自由文本 -> 任意切分”的前提下使用。
- 只有当前面已经固定：
  - 一个假说输出 `summary + claims[]`
  - 每条 claim 只表达一个输出效应
  - `|claims| <= 3`
  时，`claim 多 = 更宽泛` 才成立。
- 否则，这一项只能作为弱启发式，不能作为强惩罚。

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
2. `ValidityGate(H_a) = 1`
3. `ValidityGate(H_b) = 1`
4. `Control_Robustness(H_a)` 不显著差于 `H_b`
5. 在核心三项里至少赢两项：
   - `Targeted_Logit_Capture`
   - `Polarity_Accuracy`
   - `Generation assessment`

否则：

- 若分差很小，记 `tie`
- 若任一方未通过 `ValidityGate`，该 pair 记为 `invalid`
- 若控制项明显失败，直接判负

其中默认：

```text
margin = max(0.05, 1.5 * MAD_repeat)
```

并且：

- `ValidityGate(H) = 1` 的默认条件是 `Control_Robustness(H) >= τ_ctrl = 0.6`
- 如果两者 `Total` 很近，但一方在核心三项中赢两项以上，则仍可判其胜

Response on margin:

- `margin` 不应硬编码为理论常数，而应由“重复评测噪声”决定。
- 这里定义：

```text
MAD_repeat = 开发集上，同一 pairwise item 重复评测时，score difference 的中位绝对偏差
```

- 使用 `margin = max(0.05, 1.5 * MAD_repeat)` 的含义是：
  - 如果系统本身噪声很小，保留一个最小安全边界 `0.05`
  - 如果系统噪声较大，margin 自动变宽，避免把噪声当成真实优劣

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
