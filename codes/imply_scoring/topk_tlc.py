from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

from .contracts import CompiledClaimEntry, Step0TraceEvidence, TokenDelta
from .io import normalize_token
from .polarity import pick_bucket_by_effective_polarity, resolve_effective_polarity


@dataclass
class TopKTLCCalcConfig:
    weak_weight: float = 0.5
    epsilon: float = 1e-12


def compute_topk_targeted_logit_capture(
    *,
    claims: Sequence[CompiledClaimEntry],
    traces: Sequence[Step0TraceEvidence],
    config: TopKTLCCalcConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or TopKTLCCalcConfig()
    claim_results: List[Dict[str, Any]] = []

    for claim in claims:
        claim_results.append(
            _compute_single_claim_score(
                claim=claim,
                traces=traces,
                weak_weight=cfg.weak_weight,
                epsilon=cfg.epsilon,
            )
        )

    valid_scores = [item["score"] for item in claim_results if item["score"] is not None]
    hypothesis_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
    return {
        "metric": "TopK_Targeted_Logit_Capture",
        "hypothesis_score": hypothesis_score,
        "claim_count": len(claim_results),
        "valid_claim_count": len(valid_scores),
        "claim_scores": claim_results,
    }


def _compute_single_claim_score(
    *,
    claim: CompiledClaimEntry,
    traces: Sequence[Step0TraceEvidence],
    weak_weight: float,
    epsilon: float,
) -> Dict[str, Any]:
    polarity = claim.polarity.lower()
    if polarity not in {"promote", "suppress"}:
        return {
            "claim_index": claim.claim_index,
            "target": claim.target,
            "polarity": claim.polarity,
            "score": None,
            "trace_count": 0,
            "warning": "invalid_claim_polarity",
        }

    strong_set = _normalize_token_set(claim.S_strong)
    weak_set = _normalize_token_set(claim.S_weak) - strong_set
    per_trace: List[Dict[str, Any]] = []

    for trace in traces:
        effective_polarity = resolve_effective_polarity(claim.polarity, trace.scale)
        _, use_positive = pick_bucket_by_effective_polarity(effective_polarity)
        token_rows = trace.top_positive_delta_tokens if use_positive else trace.top_negative_delta_tokens
        trace_result = _compute_trace_coverage(
            token_rows=token_rows,
            strong_set=strong_set,
            weak_set=weak_set,
            polarity=effective_polarity,
            weak_weight=weak_weight,
            epsilon=epsilon,
        )
        per_trace.append(
            {
                "sample_rank": trace.sample_rank,
                "intervention_index": trace.intervention_index,
                "scale": trace.scale,
                "effective_polarity": effective_polarity,
                **trace_result,
            }
        )

    coverage_values = [row["coverage"] for row in per_trace if row["coverage"] is not None]
    score = sum(coverage_values) / len(coverage_values) if coverage_values else None

    return {
        "claim_index": claim.claim_index,
        "target": claim.target,
        "polarity": claim.polarity,
        "S_strong_size": len(strong_set),
        "S_weak_size": len(weak_set),
        "trace_count": len(per_trace),
        "valid_trace_count": len(coverage_values),
        "score": score,
        "per_trace": per_trace,
    }


def _compute_trace_coverage(
    *,
    token_rows: Sequence[TokenDelta],
    strong_set: Set[str],
    weak_set: Set[str],
    polarity: str,
    weak_weight: float,
    epsilon: float,
) -> Dict[str, Any]:
    top_mass = 0.0
    gain_strong = 0.0
    gain_weak = 0.0

    for row in token_rows:
        contribution = _get_contribution(delta_logit=row.delta_logit, polarity=polarity)
        if contribution <= 0:
            continue
        top_mass += contribution
        token = row.token_norm
        if token in strong_set:
            gain_strong += contribution
        elif token in weak_set:
            gain_weak += contribution

    if top_mass <= 0:
        return {
            "coverage": 0.0,
            "top_mass": 0.0,
            "gain_strong": 0.0,
            "gain_weak": 0.0,
        }

    coverage = (gain_strong + weak_weight * gain_weak) / (top_mass + epsilon)
    coverage = max(0.0, min(1.0, coverage))
    return {
        "coverage": coverage,
        "top_mass": top_mass,
        "gain_strong": gain_strong,
        "gain_weak": gain_weak,
    }


def _normalize_token_set(tokens: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for token in tokens:
        normalized = normalize_token(token)
        if normalized:
            out.add(normalized)
    return out


def _get_contribution(*, delta_logit: float, polarity: str) -> float:
    if polarity == "promote":
        return max(delta_logit, 0.0)
    return max(-delta_logit, 0.0)
