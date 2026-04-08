from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from openai import OpenAI

from function import TokenUsageAccumulator, call_llm, extract_json_object, read_api_key

from .contracts import CompiledClaimEntry, Step0TraceEvidence, TokenDelta
from .polarity import pick_bucket_by_effective_polarity, resolve_effective_polarity

LABEL_TO_SCORE = {
    "support": 1.0,
    "weak_support": 0.5,
    "no_support": 0.0,
    "contradict": -1.0,
}
VALID_LABELS = set(LABEL_TO_SCORE.keys())


@dataclass
class TokenClaimAlignmentConfig:
    llm_config_path: Path = Path("support_info") / "llm_api_info.py"
    llm_model_override: str | None = None
    top_k_tokens_per_trace: int = 10
    judge_repeats: int = 2
    confidence_threshold: float = 0.55
    epsilon: float = 1e-12
    max_tokens: int = 700
    temperature: float = 0.0
    progress_every: int = 20
    progress_callback: Callable[[Dict[str, Any]], None] | None = None


def compute_token_claim_alignment(
    *,
    claims: Sequence[CompiledClaimEntry],
    traces: Sequence[Step0TraceEvidence],
    config: TokenClaimAlignmentConfig,
) -> Dict[str, Any]:
    llm_cfg = _load_llm_config(config.llm_config_path)
    model_name = config.llm_model_override or llm_cfg["model_name"]
    api_key = read_api_key(llm_cfg["api_key_file"])
    client = OpenAI(base_url=llm_cfg["base_url"], api_key=api_key)
    usage_acc = TokenUsageAccumulator()
    progress_every = max(1, int(config.progress_every))
    total_judgements = _estimate_total_judgements(claims=claims, traces=traces, config=config)
    progress = _ProgressState(total=total_judgements, done=0, every=progress_every, callback=config.progress_callback)
    progress.emit(
        {
            "type": "start",
            "total_judgements": total_judgements,
            "claim_count": len(claims),
            "trace_count": len(traces),
            "model": model_name,
        }
    )

    claim_results: List[Dict[str, Any]] = []
    for claim_idx, claim in enumerate(claims, start=1):
        claim_results.append(
            _compute_single_claim_tca(
                claim=claim,
                traces=traces,
                client=client,
                model=model_name,
                config=config,
                usage_acc=usage_acc,
                progress=progress,
            )
        )
        progress.emit(
            {
                "type": "claim_end",
                "claim_index": claim.claim_index,
                "claim_order": claim_idx,
                "total_claims": len(claims),
                "done_judgements": progress.done,
                "total_judgements": progress.total,
            }
        )

    valid_scores = [item["score"] for item in claim_results if item["score"] is not None]
    hypothesis_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    result = {
        "metric": "Token_Claim_Alignment",
        "hypothesis_score": hypothesis_score,
        "claim_count": len(claim_results),
        "valid_claim_count": len(valid_scores),
        "token_usage": usage_acc.as_dict(),
        "claim_scores": claim_results,
    }
    progress.emit(
        {
            "type": "done",
            "done_judgements": progress.done,
            "total_judgements": progress.total,
            "hypothesis_score": hypothesis_score,
            "token_usage": usage_acc.as_dict(),
        }
    )
    return result


def _compute_single_claim_tca(
    *,
    claim: CompiledClaimEntry,
    traces: Sequence[Step0TraceEvidence],
    client: OpenAI,
    model: str,
    config: TokenClaimAlignmentConfig,
    usage_acc: TokenUsageAccumulator,
    progress: "_ProgressState",
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

    per_trace: List[Dict[str, Any]] = []
    for trace in traces:
        effective_polarity = resolve_effective_polarity(claim.polarity, trace.scale)
        bucket_name, use_positive = pick_bucket_by_effective_polarity(effective_polarity)
        rows = trace.top_positive_delta_tokens if use_positive else trace.top_negative_delta_tokens
        if config.top_k_tokens_per_trace > 0:
            rows = rows[: config.top_k_tokens_per_trace]
        trace_result = _compute_trace_tca(
            claim=claim,
            trace=trace,
            token_rows=rows,
            bucket_name=bucket_name,
            effective_polarity=effective_polarity,
            client=client,
            model=model,
            config=config,
            usage_acc=usage_acc,
            progress=progress,
        )
        per_trace.append(trace_result)

    scores = [row["tca_obs"] for row in per_trace if row["tca_obs"] is not None]
    claim_score = sum(scores) / len(scores) if scores else None
    return {
        "claim_index": claim.claim_index,
        "target": claim.target,
        "polarity": claim.polarity,
        "trace_count": len(per_trace),
        "valid_trace_count": len(scores),
        "score": claim_score,
        "per_trace": per_trace,
    }


def _compute_trace_tca(
    *,
    claim: CompiledClaimEntry,
    trace: Step0TraceEvidence,
    token_rows: Sequence[TokenDelta],
    bucket_name: str,
    effective_polarity: str,
    client: OpenAI,
    model: str,
    config: TokenClaimAlignmentConfig,
    usage_acc: TokenUsageAccumulator,
    progress: "_ProgressState",
) -> Dict[str, Any]:
    token_judgements: List[Dict[str, Any]] = []
    total_weight = 0.0
    weighted_score = 0.0

    for row in token_rows:
        weight = abs(float(row.delta_logit))
        if weight <= 0:
            continue
        judged = _judge_single_token_with_repeats(
            claim=claim,
            token_row=row,
            bucket_name=bucket_name,
            client=client,
            model=model,
            repeats=config.judge_repeats,
            confidence_threshold=config.confidence_threshold,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            usage_acc=usage_acc,
            progress=progress,
        )
        token_judgements.append(judged)
        weighted_score += weight * judged["effective_score"]
        total_weight += weight

    tca_obs = None
    if total_weight > 0:
        tca_obs = weighted_score / (total_weight + config.epsilon)
        tca_obs = max(-1.0, min(1.0, tca_obs))

    return {
        "sample_rank": trace.sample_rank,
        "intervention_index": trace.intervention_index,
        "scale": trace.scale,
        "bucket_name": bucket_name,
        "effective_polarity": effective_polarity,
        "token_count": len(token_judgements),
        "tca_obs": tca_obs,
        "token_judgements": token_judgements,
    }


def _judge_single_token_with_repeats(
    *,
    claim: CompiledClaimEntry,
    token_row: TokenDelta,
    bucket_name: str,
    client: OpenAI,
    model: str,
    repeats: int,
    confidence_threshold: float,
    max_tokens: int,
    temperature: float,
    usage_acc: TokenUsageAccumulator,
    progress: "_ProgressState",
) -> Dict[str, Any]:
    outputs: List[Dict[str, Any]] = []
    repeat_count = max(1, int(repeats))
    for _ in range(repeat_count):
        judged = _judge_single_token_once(
            claim=claim,
            token_row=token_row,
            bucket_name=bucket_name,
            client=client,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            usage_acc=usage_acc,
        )
        progress.inc(
            extra={
                "type": "tick",
                "claim_index": claim.claim_index,
                "token": token_row.token,
                "bucket_name": bucket_name,
            }
        )
        outputs.append(judged)

    avg_confidence = sum(item["confidence"] for item in outputs) / len(outputs)
    avg_raw_score = sum(item["raw_score"] for item in outputs) / len(outputs)
    if avg_confidence < confidence_threshold:
        effective_label = "no_support"
        effective_score = 0.0
        confidence_rule = "downgraded_by_low_confidence"
    else:
        effective_label = _majority_label([item["raw_label"] for item in outputs])
        effective_score = LABEL_TO_SCORE.get(effective_label, 0.0)
        confidence_rule = "kept"

    return {
        "token": token_row.token,
        "token_norm": token_row.token_norm,
        "delta_logit": token_row.delta_logit,
        "effective_label": effective_label,
        "effective_score": effective_score,
        "avg_confidence": avg_confidence,
        "confidence_rule": confidence_rule,
        "repeat_outputs": outputs,
    }


def _judge_single_token_once(
    *,
    claim: CompiledClaimEntry,
    token_row: TokenDelta,
    bucket_name: str,
    client: OpenAI,
    model: str,
    max_tokens: int,
    temperature: float,
    usage_acc: TokenUsageAccumulator,
) -> Dict[str, Any]:
    messages = _build_judge_messages(
        claim=claim,
        token_row=token_row,
        bucket_name=bucket_name,
    )
    payload, raw_text = _call_llm_json_with_repair(
        client=client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        usage_acc=usage_acc,
    )
    label = str(payload.get("label", "no_support")).strip().lower()
    if label not in VALID_LABELS:
        label = "no_support"
    confidence = _to_float(payload.get("confidence"), 0.0)
    confidence = max(0.0, min(1.0, confidence))
    return {
        "raw_label": label,
        "raw_score": LABEL_TO_SCORE[label],
        "confidence": confidence,
        "normalized_token": str(payload.get("normalized_token", token_row.token_norm)),
        "detected_language": str(payload.get("detected_language", "")),
        "translated_or_glossed_meaning": str(payload.get("translated_or_glossed_meaning", "")),
        "rationale": str(payload.get("rationale", "")),
        "raw_text": raw_text,
    }


def _build_judge_messages(
    *,
    claim: CompiledClaimEntry,
    token_row: TokenDelta,
    bucket_name: str,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a strict evaluator for claim-token semantic alignment in SAE intervention analysis.\n"
        "Task: evaluate whether ONE token aligns with ONE claim target.\n\n"
        "You must follow this multilingual procedure:\n"
        "1) Normalize token: remove tokenizer markers (e.g., leading '▁'), lowercase, trim punctuation.\n"
        "2) Detect language/script of the normalized token.\n"
        "3) If token is non-English, translate or paraphrase its meaning before judging.\n"
        "4) If token is proper noun/code fragment/garbled and meaning is unclear, use no_support.\n\n"
        "Label rubric:\n"
        "- support: direct canonical indicator of claim target.\n"
        "- weak_support: near indicator (inflection, synonym, close semantic neighbor).\n"
        "- no_support: unrelated, too generic, or uncertain.\n"
        "- contradict: semantically opposite to claim target.\n\n"
        "Hard constraints:\n"
        "- Judge semantic alignment only; do not judge intervention quality.\n"
        "- If uncertain, choose no_support.\n"
        "- Keep rationale <= 20 Chinese characters.\n"
        "- Output exactly one JSON object and nothing else.\n"
        "Output schema:\n"
        "{\n"
        '  "label": "support|weak_support|no_support|contradict",\n'
        '  "confidence": 0.0,\n'
        '  "normalized_token": "",\n'
        '  "detected_language": "",\n'
        '  "translated_or_glossed_meaning": "",\n'
        '  "rationale": ""\n'
        "}"
    )
    user_payload = {
        "claim": claim.raw_claim,
        "trace_token": {
            "token": token_row.token,
            "token_norm": token_row.token_norm,
            "delta_logit": token_row.delta_logit,
            "selected_bucket": bucket_name,
            "bucket_semantics": (
                "positive shift evidence for promote" if "positive" in bucket_name else "negative shift evidence for suppress"
            ),
        },
    }
    user_prompt = f"Evaluate alignment.\n\n{user_payload}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _call_llm_json_with_repair(
    *,
    client: OpenAI,
    model: str,
    messages: Sequence[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    usage_acc: TokenUsageAccumulator,
) -> Tuple[Dict[str, Any], str]:
    text, usage_obj = call_llm(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        response_format_text=True,
    )
    usage_acc.add(usage_obj)
    parsed = extract_json_object(text)
    if isinstance(parsed, dict):
        return parsed, text

    repair_messages = [
        {
            "role": "system",
            "content": "Convert the given content into exactly one valid JSON object. No markdown, no explanation.",
        },
        {"role": "user", "content": text},
    ]
    repaired_text, repaired_usage = call_llm(
        client=client,
        model=model,
        messages=repair_messages,
        temperature=0.0,
        max_tokens=max_tokens,
        stream=False,
        response_format_text=True,
    )
    usage_acc.add(repaired_usage)
    repaired = extract_json_object(repaired_text)
    if not isinstance(repaired, dict):
        return {"label": "no_support", "confidence": 0.0, "rationale": "judge_parse_fail"}, repaired_text
    return repaired, repaired_text


def _majority_label(labels: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    for label in labels:
        key = label if label in VALID_LABELS else "no_support"
        counts[key] = counts.get(key, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    if not sorted_items:
        return "no_support"
    top_count = sorted_items[0][1]
    tied = [label for label, count in sorted_items if count == top_count]
    if "no_support" in tied:
        return "no_support"
    return tied[0]


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_llm_config(config_path: Path) -> Dict[str, str]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"LLM config not found: {path}")
    namespace: Dict[str, Any] = {}
    exec(path.read_text(encoding="utf-8"), namespace)
    api_key_file = str(namespace.get("api_key_file", "")).strip()
    base_url = str(namespace.get("base_url", "")).strip()
    model_name = str(namespace.get("model_name", "")).strip()
    if not api_key_file or not base_url or not model_name:
        raise ValueError("Invalid llm config fields. Required: api_key_file, base_url, model_name.")
    return {
        "api_key_file": api_key_file,
        "base_url": base_url,
        "model_name": model_name,
    }


def _estimate_total_judgements(
    *,
    claims: Sequence[CompiledClaimEntry],
    traces: Sequence[Step0TraceEvidence],
    config: TokenClaimAlignmentConfig,
) -> int:
    repeats = max(1, int(config.judge_repeats))
    top_k = max(0, int(config.top_k_tokens_per_trace))
    total = 0
    for claim in claims:
        if str(claim.polarity).lower() not in {"promote", "suppress"}:
            continue
        for trace in traces:
            effective_polarity = resolve_effective_polarity(claim.polarity, trace.scale)
            _, use_positive = pick_bucket_by_effective_polarity(effective_polarity)
            rows = trace.top_positive_delta_tokens if use_positive else trace.top_negative_delta_tokens
            token_count = len(rows) if top_k <= 0 else min(len(rows), top_k)
            total += token_count * repeats
    return total


@dataclass
class _ProgressState:
    total: int
    done: int
    every: int
    callback: Callable[[Dict[str, Any]], None] | None

    def emit(self, payload: Dict[str, Any]) -> None:
        if self.callback is not None:
            self.callback(payload)

    def inc(self, *, extra: Dict[str, Any] | None = None) -> None:
        self.done += 1
        if self.callback is None:
            return
        if self.done == 1 or self.done % self.every == 0 or (self.total > 0 and self.done >= self.total):
            payload: Dict[str, Any] = {
                "type": "tick",
                "done_judgements": self.done,
                "total_judgements": self.total,
            }
            if extra:
                payload.update(extra)
            self.callback(payload)
