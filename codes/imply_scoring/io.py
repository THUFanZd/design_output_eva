from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from .contracts import CompiledClaimEntry, Step0TraceEvidence, TokenDelta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEER_ROOT = PROJECT_ROOT / "steer_output"
DEFAULT_LOGS_ROOT = PROJECT_ROOT / "codes" / "logs"


def normalize_token(token: str) -> str:
    text = token.strip().lower().replace("▁", "")
    text = re.sub(r"\s+", " ", text)
    return text


def build_steer_json_path(
    *,
    layer_id: str,
    feature_id: str,
    intervention_scope: str = "natural_support_mask",
    steer_steps: int = 1,
    steer_root: Path = DEFAULT_STEER_ROOT,
) -> Path:
    return (
        Path(steer_root)
        / f"layer-{layer_id}"
        / f"feature-{feature_id}"
        / intervention_scope
        / f"steer_steps_{steer_steps}"
        / "steer_from_neuronpedia.json"
    )


def find_latest_compile_tokens_path(
    *,
    layer_id: str,
    feature_id: str,
    logs_root: Path = DEFAULT_LOGS_ROOT,
) -> Path:
    feature_dir = Path(logs_root) / f"layer-{layer_id}" / f"feature-{feature_id}"
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature logs directory not found: {feature_dir}")
    matches = sorted(
        feature_dir.glob(f"**/layer{layer_id}-feature{feature_id}-compile-tokens.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No compile-tokens json found under: {feature_dir}"
        )
    return matches[0]


def load_compiled_claim_entries(path: Path) -> List[CompiledClaimEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_claims = payload.get("compiled_claims", [])
    if not isinstance(raw_claims, list):
        return []

    out: List[CompiledClaimEntry] = []
    for idx, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            continue
        raw_claim = item.get("claim", {})
        if not isinstance(raw_claim, dict):
            raw_claim = {}
        claim_index = _to_int(item.get("claim_index"), idx)
        target = str(raw_claim.get("target", "")).strip()
        polarity = str(raw_claim.get("polarity", "")).strip().lower()
        strong = _to_str_list(item.get("S_strong"))
        weak = _to_str_list(item.get("S_weak"))
        out.append(
            CompiledClaimEntry(
                claim_index=claim_index,
                target=target,
                polarity=polarity,
                S_strong=strong,
                S_weak=weak,
                raw_claim=raw_claim,
            )
        )
    return out


def load_step0_traces_from_steer_json(
    path: Path,
    *,
    top_k_limit: Optional[int] = None,
) -> List[Step0TraceEvidence]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        return []

    traces: List[Step0TraceEvidence] = []
    for sample_idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        sample_rank = _to_int(sample.get("rank"), sample_idx + 1)
        interventions = sample.get("interventions", [])
        if not isinstance(interventions, list):
            continue

        for intervention_idx, intervention in enumerate(interventions):
            if not isinstance(intervention, dict):
                continue
            scale = _to_float(intervention.get("scale"), 0.0)
            if scale == 0.0:
                continue

            analysis = intervention.get("logit_analysis")
            if not isinstance(analysis, dict):
                analysis = intervention.get("logits_analysis")
            if not isinstance(analysis, dict):
                continue

            step0 = _pick_step0(analysis.get("steps"))
            if not isinstance(step0, dict):
                continue

            pos_rows = _extract_token_delta_rows(
                step0.get("top_positive_delta_tokens"),
                top_k_limit=top_k_limit,
            )
            neg_rows = _extract_token_delta_rows(
                step0.get("top_negative_delta_tokens"),
                top_k_limit=top_k_limit,
            )
            traces.append(
                Step0TraceEvidence(
                    sample_rank=sample_rank,
                    intervention_index=intervention_idx,
                    scale=scale,
                    top_positive_delta_tokens=pos_rows,
                    top_negative_delta_tokens=neg_rows,
                )
            )
    return traces


def _pick_step0(steps: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(steps, list) or not steps:
        return None
    for item in steps:
        if isinstance(item, dict) and _to_int(item.get("step"), -1) == 0:
            return item
    first = steps[0]
    if isinstance(first, dict):
        return first
    return None


def _extract_token_delta_rows(rows: Any, *, top_k_limit: Optional[int]) -> List[TokenDelta]:
    if not isinstance(rows, list):
        return []
    out: List[TokenDelta] = []
    for idx, row in enumerate(rows):
        if top_k_limit is not None and idx >= top_k_limit:
            break
        if not isinstance(row, dict):
            continue
        token = str(row.get("token", "")).strip()
        if not token:
            continue
        delta = _to_float(row.get("delta_logit"), 0.0)
        out.append(
            TokenDelta(
                token=token,
                token_norm=normalize_token(token),
                delta_logit=delta,
            )
        )
    return out


def _to_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
    return out


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

