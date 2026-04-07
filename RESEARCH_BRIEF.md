# Research Brief

## Problem Statement

我目前研究的是：**单个 SAE feature 的 output-side explanation（输出端解释）的评估问题**。

对于一个 SAE 特征，我将“解释”区分为两类，这个区分在整个工作流中都必须严格保留：

1. **Input-side explanation（输入端解释）**  
   指的是：什么样的 token、概念、语义模式、句法结构、格式模式，会激活这个 feature。  
   它回答的是“这个特征在什么输入条件下会亮起”。

2. **Output-side explanation（输出端解释）**  
   指的是：当我对这个 feature 进行干预时，它会如何因果性地影响模型输出。  
   它回答的是“这个特征被增强、抑制或改写后，模型输出会朝什么方向变化”。

我当前真正想研究的，不是 input-side explanation 的评估，也不是一般意义上的 SAE benchmark，而是：

**如何评估一个 output-side explanation 是否正确、是否具体、是否真的抓住了该 feature 对输出的因果作用。**

目前，output-side explanation 的证据主要来自两类输出变化：

- **Generation-based evidence（基于生成文本的证据）**：观察干预前后生成文本的变化。
- **Logits-based evidence（基于 logits 的证据）**：观察干预前后 next-token logits 或 token distribution 的变化。

因此，我关心的问题包括：

- 一个给定的 output-side hypothesis，是否能正确预测干预后的输出变化。
- 这种预测应当如何评估：看生成文本变化、看 logits 变化，这些变化如何被量化为具体的解释评价指标。
- 还有没有其他的指标可以评估。
- 对于 SAE 特征对输出的干预影响，是否有别的输出变化证据可以设计。

我现在让你研究 output-side explanation 的评估，这个问题是我目前研究问题的一个子问题。我目前研究问题是：
基于干预证据，给出一个特征的输出端解释，并通过一个workflow迭代优化假说的质量。
你研究的这个问题，在这个大的研究问题中的作用是，对于一个workflow在不同轮次生成的假说，我需要一些定量的指标，衡量假说的质量，作为指导这个workflow refine假说的信号。当然，指导更新的前提，是这些指标需要能够做到，很好地衡量一个假说的质量。
你在研究这个问题的时候，需要考虑到这个背景。

在这个背景下，我希望整个 workflow 始终围绕这个狭义目标展开：  
**evaluation of output-side explanations for single SAE features**。
不要把 input-side explanation、output-side explanation、general SAE benchmark 混为一类；如果某篇工作只属于邻近背景，请明确标注为“背景”或“排除项”，不要混入主答案。

## Background

- **Field**: Mechanistic interpretability / Sparse Autoencoders / LLM interpretability.
- **Sub-area**: Evaluation of intervention-based output-side explanations for single SAE features.
- **Key papers I've read**:
  - 与 automated interpretability、feature explanation、output-centric explanation、SAE evaluation 相关的一批论文。
  - 其中有些论文可能是重要背景，但并不一定直接解决 output-side explanation evaluation。
  - 后续需要在调研中区分“核心相关论文”和“邻近但不直接回答问题的论文”。
- **What I already tried**:
  - 我初步设计的指标是（后面“我当前已有的baseline”也进行了描述）：给一些对这个特征进行真实干预的例子（包含干预前的输出和干预后的输出），然后给一个llm judge，把例子一个一个给他，每个例子都配有一个SAE特征的输出端解释，让他判断，这个干预前后的变化，是否来自于对这个解释对应的特征的干预。但这里面有很多的问题：
    - llm as a judge本身有很多坑
    - 一个例子只产出一个对与错的信号，数据效率太低
    - 只有这一个指标，太单薄了，至少需要把token logits change也纳入指标。

## Domain Knowledge

### 1. 概念区分

#### A. Input-side explanation
解释“什么会激活这个 feature”。

典型对象包括：
- 某类词汇
- 某类语义主题
- 某类语法结构
- 某类格式化模式
- 某种局部上下文

它研究的是激活条件，而不是输出后果。  
**这不是本项目的主目标。**

#### B. Output-side explanation
解释“这个 feature 被干预后，会怎样影响模型输出”。

典型形式包括：
- 该特征会把输出推向某种情绪方向。
- 该特征会增加某类概念或词汇的生成概率。
- 该特征会改变回答风格、语篇功能、代码行为或推理倾向。
- 该特征会在 logits 层面提高或抑制某类 token / token family / semantic direction 的概率。

**这才是本项目的主目标。**

### 2. 我关心的 output evidence

#### (1) Generation-based evidence
观察干预前后生成文本的差异，例如：
- 语义方向是否变化。
- 情绪、话题、风格、功能是否变化。
- 变化是否与 hypothesis 所描述的方向一致。
- 变化是否稳定出现在多个上下文中。

#### (2) Logits-based evidence
观察干预前后 logits 或 token distribution 的差异，例如：
- 哪些 token 的概率上升或下降。
- 上升/下降是否集中在 hypothesis 所预测的语义方向。
- 这种变化是局部的还是弥散的。
- 在不同 prompt / context 下是否一致。

### 3. 我当前已有的 baseline

我目前有一个粗糙 baseline：

给定：
- 一个候选 output-side hypothesis `H`
- 干预前输出 `y_before`
- 干预后输出 `y_after`

把 `(H, y_before, y_after)` 交给一个外部 LLM judge，判断：

“`y_after` 相对于 `y_before` 的变化，是否与 `H` 一致。”

这个 baseline 可以作为起点，但我已经知道它有明显局限：

- 它较依赖 judge model 的主观判断。
- 它更容易看见生成文本层面的变化，而不容易精确刻画 logits 层面的变化。
- 它可能奖励宽泛、模糊、套话式 hypotheses。
- 它未必能区分 causation 和 correlation。
- 它可能不能很好地比较多个候选 hypotheses 的优劣。

### 4. 我认为一个好的 output-side evaluation 应该测哪些能力

一个好的 output-side explanation evaluation，理想上应覆盖以下若干能力：

- **Faithfulness（忠实性）**  
  解释是否真的对应干预后发生的输出变化。

- **Specificity / Precision（具体性 / 精确性）**  
  解释是否过宽、是否把很多无关变化也说成“符合解释”。

- **Coverage / Recall（覆盖性 / 召回性）**  
  解释是否能覆盖该 feature 在多个上下文中的主要输出效应。

- **Logits sensitivity（logits 层面的敏感性）**  
  指标是否能评估分布变化，而不只是最终生成文本。

- 其他我没有考虑到的、很有必要的能力


你需要避免：
- 把“生成自然语言解释”本身当作主问题，而不关注 intervention evidence。
- 把大量人工标注作为主评价手段；人工评价最多只能作为小规模验证层。
