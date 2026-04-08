from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

from .contracts import CompiledClaimEntry, Step0TraceEvidence, TokenDelta
from .io import normalize_token
from .polarity import resolve_effective_polarity


@dataclass
class TopKPolarityAccuracyConfig:
    weak_weight: float = 0.5
    epsilon: float = 1e-12


def compute_topk_polarity_accuracy(
    *,
    claims: Sequence[CompiledClaimEntry],
    traces: Sequence[Step0TraceEvidence],
    config: TopKPolarityAccuracyConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or TopKPolarityAccuracyConfig()
    claim_results: List[Dict[str, Any]] = []

    for claim in claims:
        claim_results.append(
            _compute_single_claim_pa(
                claim=claim,
                traces=traces,
                weak_weight=cfg.weak_weight,
                epsilon=cfg.epsilon,
            )
        )

    valid_scores = [item["score"] for item in claim_results if item["score"] is not None]
    hypothesis_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
    return {
        "metric": "TopK_Polarity_Accuracy",
        "hypothesis_score": hypothesis_score,
        "claim_count": len(claim_results),
        "valid_claim_count": len(valid_scores),
        "claim_scores": claim_results,
    }


def _compute_single_claim_pa(
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
        trace_result = _compute_trace_pa(
            trace=trace,
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

    values = [row["pa_obs"] for row in per_trace if row["pa_obs"] is not None]
    score = sum(values) / len(values) if values else None
    return {
        "claim_index": claim.claim_index,
        "target": claim.target,
        "polarity": claim.polarity,
        "S_strong_size": len(strong_set),
        "S_weak_size": len(weak_set),
        "trace_count": len(per_trace),
        "valid_trace_count": len(values),
        "score": score,
        "per_trace": per_trace,
    }


def _compute_trace_pa(
    *,
    trace: Step0TraceEvidence,
    strong_set: Set[str],
    weak_set: Set[str],
    polarity: str,
    weak_weight: float,
    epsilon: float,
) -> Dict[str, Any]:
    correct = 0.0
    wrong = 0.0
    matched_strong = 0
    matched_weak = 0

    for row in trace.top_positive_delta_tokens:
        token = row.token_norm
        if token in strong_set:
            matched_strong += 1
            correct += _correct_contribution(delta_logit=row.delta_logit, polarity=polarity)
            wrong += _wrong_contribution(delta_logit=row.delta_logit, polarity=polarity)
        elif token in weak_set:
            matched_weak += 1
            correct += weak_weight * _correct_contribution(delta_logit=row.delta_logit, polarity=polarity)
            wrong += weak_weight * _wrong_contribution(delta_logit=row.delta_logit, polarity=polarity)

    for row in trace.top_negative_delta_tokens:
        token = row.token_norm
        if token in strong_set:
            matched_strong += 1
            correct += _correct_contribution(delta_logit=row.delta_logit, polarity=polarity)
            wrong += _wrong_contribution(delta_logit=row.delta_logit, polarity=polarity)
        elif token in weak_set:
            matched_weak += 1
            correct += weak_weight * _correct_contribution(delta_logit=row.delta_logit, polarity=polarity)
            wrong += weak_weight * _wrong_contribution(delta_logit=row.delta_logit, polarity=polarity)

    denom = correct + wrong
    if denom <= 0:
        pa_obs = None
    else:
        pa_obs = correct / (denom + epsilon)
        pa_obs = max(0.0, min(1.0, pa_obs))
    return {
        "pa_obs": pa_obs,
        "correct_mass": correct,
        "wrong_mass": wrong,
        "matched_strong_token_rows": matched_strong,
        "matched_weak_token_rows": matched_weak,
    }


def _normalize_token_set(tokens: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for token in tokens:
        normalized = normalize_token(token)
        if normalized:
            out.add(normalized)
    return out


def _correct_contribution(*, delta_logit: float, polarity: str) -> float:
    if polarity == "promote":
        return max(delta_logit, 0.0)
    return max(-delta_logit, 0.0)


def _wrong_contribution(*, delta_logit: float, polarity: str) -> float:
    if polarity == "promote":
        return max(-delta_logit, 0.0)
    return max(delta_logit, 0.0)
