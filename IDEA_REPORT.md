# Research Idea Report

**Direction**: evaluation of output-side explanations for single SAE features
**Generated**: 2026-04-05
**Stage**: Phase 2 checkpoint
**Ideas evaluated**: 10 generated -> 5 survived filtering -> 0 piloted -> waiting for user selection

## Scope Lock

This report is strictly about **how to evaluate a candidate output-side hypothesis `H` for a single SAE feature**, using intervention evidence as the center of gravity.

Out of scope as primary targets:

- input-side explanation quality
- general SAE benchmark design
- SAE training quality / reconstruction quality
- downstream steering performance as the sole metric

Additional constraint from the user:

- generation-based evidence is allowed and useful, but should be treated as **auxiliary evidence**
- LLM judge / agent judge is allowed, but only under **explicit rubrics**, with preference for **pairwise comparison** and **cross-checking against logits evidence**

## Landscape Summary

The current literature suggests that output-side evidence matters, but does not yet solve the user's exact problem. `Automatically Interpreting Millions of Features in Large Language Models` introduces intervention scoring and shows that some features are better understood by their downstream effects than by their activating contexts. `Enhancing Automated Interpretability with Output-Centric Feature Descriptions` further shows that output-centric descriptions better capture causal effects on model outputs than purely input-centric descriptions.

However, these works still mostly treat the central object as a **feature description pipeline**. They do not fully isolate the problem of **evaluating a candidate hypothesis `H` itself** as a reusable scoring function for iterative refinement. On the other side, works such as `Evaluating SAE interpretability without explanations`, `SAEBench`, `CE-Bench`, and `Measuring Sparse Autoencoder Feature Sensitivity` provide important benchmark methodology, but they target latent/SAE interpretability more broadly rather than the narrower question of output-side hypothesis quality.

The main gap is therefore:

1. hypothesis-conditioned evaluation rather than generic feature evaluation
2. intervention-grounded scoring rather than activation-only scoring
3. logits-aware metrics rather than generation-only judgment
4. pairwise ranking between competing hypotheses, so the score can guide workflow refinement
5. robustness against vague, broad, or correlation-only hypotheses

## Recommended Ideas (ranked)

### Idea 1: Pairwise Hypothesis Tournament
- **Hypothesis**: Pairwise comparison between candidate hypotheses is more stable and more useful for refinement than assigning an absolute score to a single hypothesis in isolation.
- **Core object**: `(H_i, H_j, intervention trace set)` with the same feature and prompts.
- **Minimum experiment**: For 50-100 features, collect intervention traces and create 3-4 candidate hypotheses per feature: one stronger hypothesis, one too-broad hypothesis, one wrong-polarity hypothesis, and one semantically adjacent hypothesis. Ask a rubric-constrained judge which hypothesis better explains the evidence, while also computing logits-based discriminative features. Fit a Bradley-Terry or Elo-style ranking model.
- **Expected outcome**: Pairwise rankings correlate better with small-scale human preference judgments than single absolute scores.
- **Novelty quick-check**: No close overlap found with existing SAE output-side evaluation papers. Closest prior art uses intervention/output evidence for scoring or description generation, but not hypothesis-vs-hypothesis ranking as the primary object.
- **Feasibility**: Moderate. Mostly CPU/API work plus intervention traces from an open model.
- **Risk**: MEDIUM
- **Contribution type**: diagnostic / evaluation framework
- **Pilot result**: SKIPPED in Phase 2
- **Reviewer's likely objection**: The ranking may depend heavily on how negative hypotheses are generated.
- **Why we should do this**: It matches the downstream workflow directly. A refine loop does not need a philosophically perfect absolute score; it needs a reliable signal for deciding whether `H_new` is better than `H_old`.

### Idea 2: Structured Hypothesis-to-Logits Signature Benchmark
- **Hypothesis**: Turning a free-form hypothesis into a structured prediction target makes output-side evaluation more faithful, less judge-dependent, and more sensitive to logits-level effects.
- **Core object**: `H -> structured effect signature -> empirical logits shift`
- **Minimum experiment**: Define a schema for `H`, such as predicted positive token families, suppressed token families, semantic direction, polarity, locality, and context dependence. Compile each hypothesis into this schema using a constrained parser. Score it against held-out intervention-induced logits shifts using rank correlation, mass shift overlap, and false-positive penalties.
- **Expected outcome**: Structured signatures outperform free-form judge-only scoring in identifying precise, non-broad hypotheses.
- **Novelty quick-check**: Closest work is output-centric description generation from vocabulary projection or token change, but those papers do not evaluate arbitrary candidate hypotheses `H` against held-out structured logits predictions.
- **Feasibility**: Moderate. Requires schema design and token-family construction, but avoids heavy manual annotation.
- **Risk**: MEDIUM
- **Contribution type**: evaluation method
- **Pilot result**: SKIPPED in Phase 2
- **Reviewer's likely objection**: The schema may oversimplify feature effects that are not easily reducible to token families or axes.
- **Why we should do this**: It attacks your most important missing piece directly: how to score `H` using logits evidence without collapsing back to a vague LLM-judge setup.

### Idea 3: Rubric-Calibrated Hybrid Causal Score
- **Hypothesis**: The best practical evaluator is a multi-view score that combines logits evidence, rubric-constrained generation evidence, and specificity penalties, instead of choosing between judge-based and logits-only views.
- **Core object**: `H + intervention traces + logits shifts + generation pairs`
- **Minimum experiment**: Build three sub-scores: `(a)` logits alignment, `(b)` rubric-constrained judge score on generation differences, `(c)` null/specificity penalty from irrelevant prompts or wrong-feature controls. Combine them with simple learned weights or fixed weights validated on a small human pairwise set.
- **Expected outcome**: The hybrid score is more stable than judge-only scoring and more semantically meaningful than logits-only scoring.
- **Novelty quick-check**: Closest prior art is intervention scoring plus newer benchmark work, but not a user-constrained hybrid scorer explicitly designed for evaluating hypotheses rather than SAE latents.
- **Feasibility**: High. This is the most buildable near-term idea.
- **Risk**: LOW-MEDIUM
- **Contribution type**: evaluation method / practical benchmark
- **Pilot result**: SKIPPED in Phase 2
- **Reviewer's likely objection**: The score may look engineered unless the rubric and weight design are well motivated.
- **Why we should do this**: It respects your correction exactly. Generation evidence remains in the system, but only inside a constrained, auditable scoring stack.

### Idea 4: Contrastive Coverage-and-Specificity Sweep
- **Hypothesis**: A good output-side explanation should not only predict the right effect when it appears, but also stay quiet on contexts where the effect should not appear.
- **Core object**: `H + matched prompt families + intervention effect presence/absence`
- **Minimum experiment**: For each feature and candidate hypothesis, generate or retrieve matched prompt sets: expected-positive contexts, expected-negative contexts, and semantically adjacent decoys. Measure whether intervention effects occur where `H` predicts them and stay absent where `H` should not apply. Use logits shifts as primary evidence and rubric-constrained generation judgments as secondary evidence.
- **Expected outcome**: Broad hypotheses collapse under false-positive pressure, while sharper hypotheses gain specificity advantage.
- **Novelty quick-check**: Closest background is sensitivity / contrastive benchmark work, but those papers evaluate feature interpretability in general rather than output-side hypotheses under intervention.
- **Feasibility**: Moderate
- **Risk**: MEDIUM
- **Contribution type**: diagnostic benchmark component
- **Pilot result**: SKIPPED in Phase 2
- **Reviewer's likely objection**: Building fair contrastive prompt sets may itself encode assumptions about the hypothesis.
- **Why we should do this**: It gives you a concrete path to measuring `Specificity / Precision` and `Coverage / Recall`, which your current baseline does not separate well.

### Idea 5: Adversarial Negative Hypothesis Benchmark
- **Hypothesis**: The hardest failure mode for judge-based evaluation is rewarding vague or semantically adjacent hypotheses; therefore, evaluation should explicitly include hard negative hypotheses designed to fool the scorer.
- **Core object**: `(H_true, H_broad, H_adjacent, H_wrong-polarity, evidence)`
- **Minimum experiment**: Starting from a strong hypothesis, use an LLM/agent to synthesize adversarial negatives in several categories: broader-than-necessary, right topic wrong causal direction, correlated-but-not-causal, and partially correct but incomplete. Test whether the scorer reliably ranks the intended hypothesis above these negatives.
- **Expected outcome**: A strong evaluator should sharply separate precise hypotheses from attractive but incorrect near-misses.
- **Novelty quick-check**: Existing work discusses scoring pitfalls, but we did not find a direct SAE output-side benchmark centered on adversarial negative hypotheses for evaluator stress-testing.
- **Feasibility**: High
- **Risk**: LOW-MEDIUM
- **Contribution type**: evaluator stress test / benchmark design
- **Pilot result**: SKIPPED in Phase 2
- **Reviewer's likely objection**: The benchmark quality depends on whether the adversarial negatives are truly hard and not strawmen.
- **Why we should do this**: This idea directly attacks your stated concern that judge-based metrics may reward broad or slippery hypotheses.

## Backup Ideas

### Backup 1: Claim-Decomposed Agent Judge
- Break `H` into atomic claims such as polarity, semantic target, scope, and context dependence, then verify them separately against logits/generation evidence.
- Strength: more interpretable score breakdown.
- Weakness: more engineering and prompt design complexity.

### Backup 2: Complexity-Penalized Explanation Score
- Score = fit to intervention evidence minus a penalty for breadth, ambiguity, or unused claims.
- Strength: directly discourages vague hypotheses.
- Weakness: complexity penalties are easy to criticize as arbitrary.

## Eliminated Ideas (for reference)

| Idea | Reason eliminated |
|------|-------------------|
| Pure free-form LLM judge on `(H, y_before, y_after)` | Too close to current baseline; weak control over specificity and judge drift |
| Pure logits-only absolute score | Over-corrects and may miss high-level behavioral effects that matter to output-side interpretation |
| General SAE-level benchmark | Misses the actual object of interest, which is a candidate hypothesis for a single feature |
| Downstream steering task as sole metric | Too indirect; success on a task does not imply that the hypothesis correctly captures the feature's causal role |
| Heavy human annotation benchmark | Too expensive and mismatched with the user's requirement that human eval should remain small-scale validation only |

## Suggested Execution Order

1. Start with **Idea 1: Pairwise Hypothesis Tournament**
2. In parallel or immediately after, instantiate **Idea 3: Rubric-Calibrated Hybrid Causal Score**
3. Use **Idea 5: Adversarial Negative Hypothesis Benchmark** as the evaluator stress test
4. If the first prototype looks promising, extend toward **Idea 2** for a more logits-grounded benchmark
5. Add **Idea 4** once you need explicit coverage/specificity decomposition

## Recommended Next Step

If continuing beyond this checkpoint, the best next move is:

1. freeze a concrete evaluator target around **Idea 1 + Idea 3 + Idea 5**
2. write a precise scoring rubric and evidence format
3. run a novelty check on the top 2-3 ideas as a bundle
4. then enter Phase 3 / Phase 4 style validation with external review
