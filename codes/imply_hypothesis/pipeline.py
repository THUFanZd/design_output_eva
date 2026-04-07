from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

from function import (
    TokenUsageAccumulator,
    build_round_dir,
    call_llm,
    extract_json_object,
    normalize_round_id,
    read_api_key,
)
from neuronpedia_feature_api import extract_explanations, fetch_feature_json

from .hypothesis_contract import (
    AtomicClaim,
    CompiledClaimTokens,
    add_simple_lexical_expansion,
    infer_compile_status,
    normalize_hypothesis_payload,
    normalize_token_candidates,
)
from .prompts import (
    build_compile_expand_messages,
    build_compile_seed_messages,
    build_hypothesis_messages,
)


@dataclass
class LLMConfig:
    api_key_file: str
    base_url: str
    model_name: str


@dataclass
class PipelineRunConfig:
    model_id: str
    layer_id: str
    feature_id: str
    width: str = "16k"
    round_id: str = "round_0"
    timestamp: Optional[str] = None
    top_k_pairs: int = 24
    max_claims: int = 3
    neuronpedia_api_key: Optional[str] = None
    timeout: int = 30
    llm_config_path: Path = Path("support_info") / "llm_api_info.py"
    llm_model_override: Optional[str] = None
    hypothesis_temperature: float = 0.2
    compile_temperature: float = 0.1
    hypothesis_max_tokens: int = 1200
    compile_max_tokens: int = 900
    compile_max_weak_tokens: int = 48


def load_llm_config(config_path: Path) -> LLMConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"LLM config not found: {config_path}")
    namespace: Dict[str, Any] = {}
    source = config_path.read_text(encoding="utf-8")
    exec(source, namespace)

    api_key_file = str(namespace.get("api_key_file", "")).strip()
    base_url = str(namespace.get("base_url", "")).strip()
    model_name = str(namespace.get("model_name", "")).strip()
    if not api_key_file or not base_url or not model_name:
        raise ValueError(
            "Invalid llm config file. Required fields: api_key_file, base_url, model_name."
        )
    return LLMConfig(api_key_file=api_key_file, base_url=base_url, model_name=model_name)


def run_pipeline(config: PipelineRunConfig) -> Dict[str, Any]:
    llm_cfg = load_llm_config(config.llm_config_path)
    model_name = config.llm_model_override or llm_cfg.model_name
    api_key = read_api_key(llm_cfg.api_key_file)
    client = OpenAI(base_url=llm_cfg.base_url, api_key=api_key)
    usage_acc = TokenUsageAccumulator()

    timestamp = config.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_round_id = normalize_round_id(config.round_id, round_index=0)
    round_dir = build_round_dir(
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        timestamp=timestamp,
        round_id=resolved_round_id,
        round_index=0,
    )
    round_dir.mkdir(parents=True, exist_ok=True)

    source = _build_source(config.layer_id, config.width)
    feature_payload = fetch_feature_json(
        model_id=config.model_id,
        source=source,
        feature_id=config.feature_id,
        api_key=config.neuronpedia_api_key,
        timeout=config.timeout,
    )
    raw_payload_path = round_dir / f"layer{config.layer_id}-feature{config.feature_id}-neuronpedia-raw.json"
    _write_json(raw_payload_path, feature_payload)

    evidence = _build_initial_evidence(
        feature_payload=feature_payload,
        model_id=config.model_id,
        layer_id=config.layer_id,
        feature_id=config.feature_id,
        width=config.width,
        top_k_pairs=config.top_k_pairs,
    )
    evidence_path = round_dir / f"layer{config.layer_id}-feature{config.feature_id}-output-evidence.json"
    _write_json(evidence_path, evidence)

    hypothesis_obj, hypothesis_raw_text = _generate_hypothesis_ch(
        client=client,
        model=model_name,
        evidence=evidence,
        max_claims=config.max_claims,
        temperature=config.hypothesis_temperature,
        max_tokens=config.hypothesis_max_tokens,
        usage_acc=usage_acc,
    )
    hypothesis_path = round_dir / f"layer{config.layer_id}-feature{config.feature_id}-hypothesis-ch.json"
    _write_json(hypothesis_path, hypothesis_obj.to_dict())

    compile_results, compile_debug = _compile_claims(
        client=client,
        model=model_name,
        claims=hypothesis_obj.claims,
        evidence=evidence,
        temperature=config.compile_temperature,
        max_tokens=config.compile_max_tokens,
        max_weak_tokens=config.compile_max_weak_tokens,
        usage_acc=usage_acc,
    )
    compile_path = round_dir / f"layer{config.layer_id}-feature{config.feature_id}-compile-tokens.json"
    _write_json(compile_path, {"compiled_claims": [item.to_dict() for item in compile_results]})

    result = {
        "metadata": {
            "model_id": config.model_id,
            "layer_id": config.layer_id,
            "feature_id": config.feature_id,
            "width": config.width,
            "timestamp": timestamp,
            "round_id": resolved_round_id,
            "llm_model": model_name,
            "source": source,
        },
        "evidence": evidence,
        "C_H": hypothesis_obj.to_dict(),
        "compile_results": [item.to_dict() for item in compile_results],
        "debug": {
            "hypothesis_raw_text": hypothesis_raw_text,
            "compile_llm_debug": compile_debug,
        },
        "token_usage": usage_acc.as_dict(),
        "paths": {
            "round_dir": str(round_dir),
            "raw_payload": str(raw_payload_path),
            "evidence": str(evidence_path),
            "hypothesis_ch": str(hypothesis_path),
            "compile_tokens": str(compile_path),
        },
    }
    result_path = round_dir / f"layer{config.layer_id}-feature{config.feature_id}-imply-result.json"
    _write_json(result_path, result)
    result["paths"]["result"] = str(result_path)
    return result


def _build_source(layer_id: str, width: str) -> str:
    return f"{layer_id}-gemmascope-res-{width}"


def _pair_str_values(strings: Any, values: Any, *, top_k_pairs: int) -> List[Dict[str, Any]]:
    if not isinstance(strings, list) or not isinstance(values, list):
        return []
    out: List[Dict[str, Any]] = []
    pair_count = min(len(strings), len(values), max(0, int(top_k_pairs)))
    for idx in range(pair_count):
        out.append(
            {
                "rank": idx + 1,
                "text": str(strings[idx]),
                "value": values[idx],
            }
        )
    return out


def _build_initial_evidence(
    *,
    feature_payload: Dict[str, Any],
    model_id: str,
    layer_id: str,
    feature_id: str,
    width: str,
    top_k_pairs: int,
) -> Dict[str, Any]:
    return {
        "feature_ref": {
            "model_id": model_id,
            "layer_id": layer_id,
            "feature_id": feature_id,
            "source": _build_source(layer_id, width),
        },
        "output_side_observation": {
            "pos_pairs": _pair_str_values(
                feature_payload.get("pos_str"),
                feature_payload.get("pos_values"),
                top_k_pairs=top_k_pairs,
            ),
            "neg_pairs": _pair_str_values(
                feature_payload.get("neg_str"),
                feature_payload.get("neg_values"),
                top_k_pairs=top_k_pairs,
            ),
        },
        "annotation_candidates": extract_explanations(feature_payload, limit=3),
    }


def _generate_hypothesis_ch(
    *,
    client: OpenAI,
    model: str,
    evidence: Dict[str, Any],
    max_claims: int,
    temperature: float,
    max_tokens: int,
    usage_acc: TokenUsageAccumulator,
) -> Tuple[Any, str]:
    messages = build_hypothesis_messages(evidence=evidence, max_claims=max_claims)
    payload, raw_text = _call_llm_json_with_repair(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        usage_acc=usage_acc,
    )
    hypothesis = normalize_hypothesis_payload(payload, max_claims=max_claims)
    return hypothesis, raw_text


def _compile_claims(
    *,
    client: OpenAI,
    model: str,
    claims: Sequence[AtomicClaim],
    evidence: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    max_weak_tokens: int,
    usage_acc: TokenUsageAccumulator,
) -> Tuple[List[CompiledClaimTokens], List[Dict[str, Any]]]:
    compiled: List[CompiledClaimTokens] = []
    debug_items: List[Dict[str, Any]] = []

    for idx, claim in enumerate(claims):
        seed_messages = build_compile_seed_messages(claim=claim, evidence=evidence)
        seed_obj, seed_raw = _call_llm_json_with_repair(
            client=client,
            model=model,
            messages=seed_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            usage_acc=usage_acc,
        )
        seed_strong = _extract_candidate_list(seed_obj, primary="S_strong")
        seed_weak = _extract_candidate_list(seed_obj, primary="S_weak")

        expand_messages = build_compile_expand_messages(
            claim=claim,
            seed_strong=seed_strong,
            seed_weak=seed_weak,
        )
        expand_obj, expand_raw = _call_llm_json_with_repair(
            client=client,
            model=model,
            messages=expand_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            usage_acc=usage_acc,
        )
        extra_strong = _extract_candidate_list(expand_obj, primary="extra_strong")
        extra_weak = _extract_candidate_list(expand_obj, primary="extra_weak")

        strong_candidates = seed_strong + extra_strong + add_simple_lexical_expansion(seed_strong)
        weak_candidates = seed_weak + extra_weak + add_simple_lexical_expansion(seed_weak)
        S_strong = normalize_token_candidates(strong_candidates, max_items=24)
        weak_all = normalize_token_candidates(weak_candidates, max_items=max_weak_tokens + 24)
        strong_set = set(S_strong)
        S_weak = [token for token in weak_all if token not in strong_set][:max_weak_tokens]

        raw_status = str(seed_obj.get("compile_status", "")).strip().lower() if isinstance(seed_obj, dict) else ""
        compile_status = infer_compile_status(
            claim=claim,
            strong_tokens=S_strong,
            weak_tokens=S_weak,
            raw_status=raw_status,
        )
        warnings = _collect_compile_warnings(claim=claim, strong_tokens=S_strong, weak_tokens=S_weak)

        compiled_item = CompiledClaimTokens(
            claim_index=idx,
            claim=claim,
            S_strong=S_strong,
            S_weak=S_weak,
            compile_status=compile_status,
            warnings=warnings,
        )
        compiled.append(compiled_item)
        debug_items.append(
            {
                "claim_index": idx,
                "seed_raw_text": seed_raw,
                "seed_json": seed_obj,
                "expand_raw_text": expand_raw,
                "expand_json": expand_obj,
            }
        )

    return compiled, debug_items


def _collect_compile_warnings(
    *,
    claim: AtomicClaim,
    strong_tokens: Sequence[str],
    weak_tokens: Sequence[str],
) -> List[str]:
    warnings: List[str] = []
    if not strong_tokens and claim.manifestation in {"logits", "both"}:
        warnings.append("no_strong_tokens_for_logit_claim")
    if not weak_tokens:
        warnings.append("no_weak_tokens")
    if len(strong_tokens) > 20:
        warnings.append("too_many_strong_tokens")
    return warnings


def _extract_candidate_list(data: Any, *, primary: str) -> List[str]:
    if not isinstance(data, dict):
        return []
    candidates = data.get(primary)
    if isinstance(candidates, list):
        return [str(item) for item in candidates]
    if isinstance(candidates, str):
        return [candidates]
    return []


def _call_llm_json_with_repair(
    *,
    client: OpenAI,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: int,
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
            "content": (
                "Convert the given content into exactly one valid JSON object. "
                "Do not add markdown or explanations."
            ),
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
        raise ValueError("LLM output is not a valid JSON object after repair.")
    return repaired, repaired_text


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
