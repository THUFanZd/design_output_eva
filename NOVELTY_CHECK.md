# Novelty Check Report

**Topic**: evaluation of output-side explanations for single SAE features  
**Date**: 2026-04-05

## Proposed Method

提出一个专门用于单个 SAE feature 的 `output-side hypothesis` 评价器：

- 把候选假说 `H` 分解为可检验 claims
- 以 `intervention-induced logits shift` 作为主证据
- 以 `before/after generation pair + rubric-constrained judge` 作为辅证据
- 加入 `wrong-feature / mismatched-trace / expected-negative-context` controls
- 输出既可做绝对分，也可做 `H_new vs H_old` 的 pairwise ranking，服务于 hypothesis-refine workflow

## Core Claims

1. 现有工作没有把“单个 feature 的 output-side hypothesis 质量评估”单独做成核心对象。
2. 现有 output-side 相关工作更多关注 description generation 或 generic intervention scoring，而不是为 hypothesis refinement 提供稳定评分信号。
3. 真正可守住的新意不在于 “claim decomposition” 或 “hard negatives” 本身，而在于它们被用于 **output-side + intervention-grounded + logits-first** 的 hypothesis evaluation。

## Closest Prior Work

| Work | Overlap | Key Difference |
|------|---------|----------------|
| [Automatically Interpreting Millions of Features in Large Language Models](https://arxiv.org/abs/2410.13928) | 有 intervention scoring；承认 downstream effects 重要 | 仍以自动解释 pipeline 为中心，不是专门评估候选 output-side hypothesis `H` |
| [Enhancing Automated Interpretability with Output-Centric Feature Descriptions](https://arxiv.org/abs/2501.08319) | 明确强调 output-centric / causal effect on outputs | 重点是生成更好的 feature descriptions，不是构造专门的 hypothesis evaluator |
| [Evaluating SAE interpretability without explanations](https://arxiv.org/abs/2507.08473) | 质疑 explanation-centered evaluation；强调更直接评测 | 目标是 latent/SAE interpretability，而不是单个 feature 的 output-side hypothesis 质量 |
| [Revising and Falsifying Sparse Autoencoder Feature Explanations](https://openreview.net/forum?id=OJAW2mHVND) | 结构化 explanations、close negatives、迭代修正 | 与我们的 claim decomposition / hard negatives 部分高度邻近，因此这些不能作为主贡献；必须把新意锚定到 output-side intervention evaluation |
| [Measuring Sparse Autoencoder Feature Sensitivity](https://openreview.net/forum?id=119qowYLUX) | 泛化、敏感性、benchmark methodology | 更偏 general feature sensitivity，不是 hypothesis-conditioned evaluation |
| [Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders](https://openreview.net/forum?id=Q4ooLNOFeR) | pairwise analysis 思想邻近 | 分析对象是 SAE interpretability 与 utility 关系，不是 `H_new` vs `H_old` 这种 hypothesis refinement 比较 |
| [Multi-shot AutoInterp: Agents Can Explain Complex Features By Refining Explanations](https://openreview.net/forum?id=oVn7S2Dkbe) | explanation refinement 过程邻近 | 更偏 explanation generation/refinement，不是 evaluation of output-side hypotheses |

## Overall Novelty Assessment

- **Score**: 6.5/10
- **Recommendation**: PROCEED WITH CAUTION

## Main Novelty Risk

最大的 novelty risk 是：

- 如果论文把重点写成 `claim decomposition`、`hard negatives`、`agent refinement`，会被近邻工作吞掉；
- 如果重点写成一般 SAE benchmark，也会和已有 benchmark 线重合。

## Positioning That Still Looks Defensible

建议把题目和贡献锚定为：

**A hypothesis-conditioned, intervention-grounded, logits-first evaluator for output-side explanations of single SAE features**

其中最该强调的差异是：

1. 评价对象是候选假说 `H`，不是 feature explanation pipeline 本身
2. 主证据是 logits-level causal effect，而不是 activation-only 或 free-form judge
3. 评分器服务于 iterative refinement，需要支持 `H_new` vs `H_old`

## What To Avoid Claiming

- 不要声称“首次使用 hard negatives / structured explanations”
- 不要声称“首次做 pairwise analysis”
- 不要把 human evaluation 当主卖点
- 不要把 contribution 写成通用 SAE benchmark
