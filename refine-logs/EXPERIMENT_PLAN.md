# Experiment Plan

**Project**: evaluation of output-side explanations for single SAE features
**Date**: 2026-04-05

## Main Claim

A logits-first, rubric-calibrated evaluator measures output-side hypothesis quality better than free-form judge baselines and better supports iterative hypothesis refinement.

## Minimal Experiment Package

### Block 1: Build a controlled benchmark slice

Use a manageable subset:

- 50-100 SAE features from one open model + one SAE family
- for each feature, collect intervention traces on a fixed prompt set
- prompts split into:
  - expected-positive contexts
  - expected-negative contexts
  - decoy contexts

For each feature, prepare:

- one stronger hypothesis
- one broad hypothesis
- one wrong-polarity hypothesis
- one adjacent or correlational hypothesis

This benchmark slice is enough to test whether the evaluator separates good hypotheses from hard negatives.

### Block 2: Implement baseline scorers

At minimum compare against:

1. **Free-form judge baseline**
   - input: `(H, y_before, y_after)`
   - output: scalar score from an LLM judge
2. **Simple logits-only baseline**
   - score only by overlap with promoted/suppressed token sets
3. **Proposed hybrid evaluator**
   - logits-first
   - rubric-constrained generation auxiliary
   - controls included

### Block 3: Core validation experiments

#### E1. Hard-negative separation

Question:

- Does the evaluator rank the intended hypothesis above broad / partial / wrong-polarity / adjacent hypotheses?

Metric:

- pairwise accuracy
- AUC over intended-vs-negative comparisons

#### E2. Human alignment

Question:

- Does the evaluator agree with small-scale human pairwise judgments?

Setup:

- small validation set only
- 20-30 features
- humans choose which of two hypotheses better explains the evidence

Metric:

- agreement rate or rank correlation

#### E3. Control robustness

Question:

- Does the evaluator stay low on wrong-feature, mismatched-trace, and expected-negative context controls?

Metric:

- false positive rate on controls

#### E4. Refinement utility

Question:

- When a workflow proposes `H_old -> H_new`, does the evaluator prefer the improved hypothesis more reliably than baselines?

Metric:

- pairwise improvement detection accuracy

### Block 4: Ablations

Required ablations:

1. remove logits module
2. remove generation module
3. remove controls
4. remove pairwise comparison, keep only absolute score

These ablations show which component is actually carrying the result.

## Expected Success Pattern

The proposal is supported if:

- the proposed evaluator beats the free-form judge baseline on hard-negative separation
- it has lower false positive rate on controls
- it aligns better with human pairwise judgments
- pairwise comparison helps detect genuine hypothesis refinement

## Failure Pattern

The proposal weakens if:

- a simple free-form judge baseline performs similarly
- logits-first scoring adds little over generation-only judgment
- controls do not change results
- pairwise preference is unstable across prompt sets

## Priority Order

1. E1 hard-negative separation
2. E3 control robustness
3. E2 human alignment
4. E4 refinement utility
5. ablations

## Compute Notes

This is mostly a CPU / inference / evaluation project.

The expensive part is not training, but:

- collecting intervention traces
- running judged comparisons
- constructing controlled prompt and hypothesis sets

So the first implementation should optimize for:

- one model
- one SAE family
- small but carefully curated feature subset
