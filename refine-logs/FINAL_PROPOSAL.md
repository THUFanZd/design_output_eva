# Final Proposal

**Problem**: evaluation of output-side explanations for single SAE features
**Date**: 2026-04-05
**Status**: READY FOR VALIDATION

## Problem Anchor

We do **not** want to solve general SAE interpretability evaluation. We want to evaluate whether a candidate **output-side hypothesis** `H` correctly captures how intervention on a single SAE feature changes model outputs.

The evaluator must be useful for a workflow that iteratively refines hypotheses. Therefore, it should not only assign a score to one hypothesis, but also support deciding whether `H_new` is better than `H_old`.

## Method Thesis

Build a **rubric-calibrated evaluator** for output-side hypotheses, with:

- **logits-level intervention evidence as the primary signal**
- **generation-level evidence as an auxiliary signal**
- **explicit controls** to punish broad, correlation-only, or wrong-feature explanations
- **pairwise comparison** to support hypothesis refinement

## Core Representation

For each feature `f`, candidate hypothesis `H` is converted into a small set of atomic claims:

1. **Target**: what output content or behavior is affected
2. **Polarity**: whether the feature promotes or suppresses that target
3. **Scope**: whether the effect is local or diffuse
4. **Context condition**: where the effect should or should not appear

This decomposition is not the paper's main novelty. Its role is to make evaluation auditable and to prevent the scorer from rewarding vague natural-language explanations.

## Evidence Format

For each prompt and feature intervention:

- `logits_before`, `logits_after`
- `generation_before`, `generation_after`
- prompt metadata: positive / negative / decoy context under the hypothesis

## Scoring Idea

### 1. Primary score: logits alignment

The main idea is simple:

- if `H` says the feature should promote a target, then intervention should shift probability mass toward that target
- if `H` says the feature should suppress a target, then intervention should shift mass away from it
- if `H` says the effect is context-dependent, that shift should mainly appear in the right contexts

Concrete sub-scores:

- **Promotion Capture**: how much increased probability mass lands in the predicted promoted set
- **Suppression Capture**: how much decreased mass lands in the predicted suppressed set
- **Polarity Accuracy**: whether the sign of change matches the claim
- **Diffuseness Penalty**: penalize hypotheses that predict a narrow effect but evidence shows broad unrelated change
- **Context Consistency**: reward hypotheses whose predicted effect appears in expected-positive contexts more than expected-negative or decoy contexts

### 2. Auxiliary score: rubric-constrained generation judgment

Generation evidence is retained, but the judge must score only pre-defined questions:

- Did the semantic direction change as predicted?
- Did the polarity match the hypothesis?
- Was the change concentrated in the claimed scope?
- Did the effect appear in contexts where the hypothesis says it should?

The judge does not give a free-form overall opinion. It answers claim-level rubric items.

### 3. Control penalties

The evaluator should also fail the wrong explanations.

Controls:

- **Wrong-feature control**: apply `H` to another feature's intervention traces
- **Mismatched-trace control**: shuffle traces so evidence no longer belongs to the right prompt-feature pair
- **Expected-negative context control**: evaluate contexts where the claimed effect should not appear

If `H` still gets a high score on these controls, the evaluator is rewarding vagueness rather than faithfulness.

## Output of the Evaluator

Two outputs are needed:

1. **Absolute score**
   - useful for analysis and thresholding
2. **Pairwise preference**
   - given `H_a` and `H_b`, decide which better explains the same evidence
   - this is the signal most useful for iterative refinement workflows

## Dominant Contribution

The dominant contribution is:

> a dedicated, intervention-grounded, logits-first evaluator for **output-side hypothesis quality** of single SAE features

## Supporting Contributions

- a claim-level rubric that constrains LLM/agent judges
- a control suite for detecting broad and correlation-only explanations
- a pairwise comparison protocol for hypothesis refinement

## Frozen Design Specs

- Claim schema: `refine-logs/CLAIM_SCHEMA.md`
- Evidence format: `refine-logs/EVIDENCE_FORMAT.md`
- Rubric: `refine-logs/RUBRIC.md`

## What Is Explicitly Rejected

- treating general SAE benchmark design as the main story
- making free-form LLM judgment the main scoring mechanism
- selling structured explanations or hard negatives as the primary novelty
- treating downstream steering success as sufficient evidence of explanation quality

## Main Risks

1. Some nearby papers already use structured explanations, close negatives, or pairwise analysis.
2. If the paper is framed too broadly, it will look like another general SAE explanation paper.
3. If the logits side is weakly implemented, the contribution collapses back to a dressed-up judge baseline.

## Positioning

This work should be positioned as:

- not a general SAE interpretability benchmark
- not a better explanation generator
- not a generic falsification pipeline
- but an evaluator for **single-feature output-side hypotheses** under intervention evidence
