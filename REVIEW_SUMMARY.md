# Review Summary

**Date**: 2026-04-05

## Current Assessment

The research direction still looks promising, but only under a narrow framing.

### What should remain central

- output-side hypothesis evaluation
- intervention-grounded evidence
- logits-first scoring
- pairwise usefulness for hypothesis refinement

### What should not be sold as the main novelty

- structured explanations
- hard negatives
- pairwise comparison alone
- iterative refinement alone

These components are useful, but adjacent work already occupies much of that space.

## Main Novelty Risk

The largest novelty risk comes from nearby papers on structured explanation revision and falsification. Therefore the paper must remain explicitly centered on:

- single-feature output-side hypotheses
- intervention traces
- logits-level evidence
- evaluator quality rather than explanation generation

## Main Reviewer Risk

The strongest likely criticism is:

> this looks like a combination of existing ingredients unless you show that output-side, logits-grounded evaluation is genuinely different from general explanation scoring

## Immediate Recommendation

Do not broaden the project. Keep the proposal compact and evaluator-centered.
