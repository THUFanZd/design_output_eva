# Pipeline Summary

**Problem**: Evaluate candidate output-side hypotheses for single SAE features
**Final Method Thesis**: Use a logits-first, rubric-calibrated evaluator with controls and pairwise comparison to score whether a hypothesis correctly captures intervention-induced output changes.
**Final Verdict**: READY
**Date**: 2026-04-05

## Final Deliverables

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Claim schema: `refine-logs/CLAIM_SCHEMA.md`
- Evidence format: `refine-logs/EVIDENCE_FORMAT.md`
- Rubric: `refine-logs/RUBRIC.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Novelty check: `NOVELTY_CHECK.md`

## Contribution Snapshot

- Dominant contribution:
  - dedicated evaluation of output-side hypothesis quality for single SAE features
- Optional supporting contribution:
  - claim-level rubric and hard-negative controls
- Explicitly rejected complexity:
  - general SAE benchmarking
  - free-form judge-only scoring
  - broad explanation generation as the main story

## Must-Prove Claims

- The proposed evaluator separates good hypotheses from hard negatives better than judge-only baselines.
- Logits-first evidence improves faithfulness and specificity.
- Pairwise scoring is useful for iterative hypothesis refinement.

## First Runs to Launch

1. Build a 50-100 feature benchmark slice with intervention traces
2. Implement judge-only baseline and simple logits-only baseline
3. Implement proposed hybrid evaluator and test hard-negative separation

## Main Risks

- Nearby work may cover structured explanations, close negatives, or pairwise analysis
- If logits metrics are weak, the story will collapse into an LLM-judge paper
- Control prompt construction may itself become a source of bias

## Next Action

- Run a stricter external review and then move to implementation of the benchmark slice
