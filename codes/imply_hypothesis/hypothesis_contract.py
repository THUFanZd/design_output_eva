from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

ALLOWED_TARGET_TYPES: Set[str] = {
    "token_family",
    "semantic_direction",
    "style_or_function",
    "task_behavior",
}
ALLOWED_POLARITY: Set[str] = {"promote", "suppress"}
ALLOWED_SCOPE: Set[str] = {"focused", "moderate", "diffuse"}
ALLOWED_CONTEXT_CONDITION: Set[str] = {"always", "positive_only", "conditional"}
ALLOWED_MANIFESTATION: Set[str] = {"logits", "generation", "both"}

FORBIDDEN_TARGET_HINTS: Tuple[str, ...] = (
    "input-side",
    "input side",
    "activation",
    "max_token",
    "max token",
    "trace index",
)

GENERIC_TOKEN_BLACKLIST: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "general",
    "in",
    "is",
    "it",
    "many",
    "of",
    "on",
    "or",
    "related",
    "some",
    "text",
    "that",
    "the",
    "their",
    "theme",
    "themes",
    "this",
    "to",
    "token",
    "tokens",
    "topic",
    "with",
}


@dataclass
class AtomicClaim:
    target_type: str
    target: str
    polarity: str
    scope: str
    context_condition: str
    condition_note: str
    manifestation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_type": self.target_type,
            "target": self.target,
            "polarity": self.polarity,
            "scope": self.scope,
            "context_condition": self.context_condition,
            "condition_note": self.condition_note,
            "manifestation": self.manifestation,
        }


@dataclass
class HypothesisCH:
    summary: str
    claims: List[AtomicClaim]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "claims": [claim.to_dict() for claim in self.claims],
            "warnings": list(self.warnings),
        }


@dataclass
class CompiledClaimTokens:
    claim_index: int
    claim: AtomicClaim
    S_strong: List[str]
    S_weak: List[str]
    compile_status: str
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_index": self.claim_index,
            "claim": self.claim.to_dict(),
            "S_strong": list(self.S_strong),
            "S_weak": list(self.S_weak),
            "compile_status": self.compile_status,
            "warnings": list(self.warnings),
        }


def _to_clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_hypothesis_payload(payload: Dict[str, Any], *, max_claims: int = 3) -> HypothesisCH:
    warnings: List[str] = []
    summary = _to_clean_str(payload.get("summary"))
    if not summary:
        summary = "No reliable summary generated."
        warnings.append("missing_summary")

    raw_claims = payload.get("claims", [])
    if not isinstance(raw_claims, list):
        raw_claims = []
        warnings.append("claims_not_list")

    claims: List[AtomicClaim] = []
    for idx, raw in enumerate(raw_claims):
        if len(claims) >= max_claims:
            warnings.append(f"claims_truncated_to_{max_claims}")
            break
        if not isinstance(raw, dict):
            warnings.append(f"claim_{idx}_not_object")
            continue
        parsed, claim_warnings = _parse_claim(raw)
        warnings.extend([f"claim_{idx}_{w}" for w in claim_warnings])
        if parsed is not None:
            claims.append(parsed)

    if not claims:
        warnings.append("no_valid_claims")

    return HypothesisCH(summary=summary, claims=claims, warnings=warnings)


def _parse_claim(raw: Dict[str, Any]) -> Tuple[AtomicClaim | None, List[str]]:
    warnings: List[str] = []

    target_type = _to_clean_str(raw.get("target_type")).lower()
    target = _to_clean_str(raw.get("target"))
    polarity = _to_clean_str(raw.get("polarity")).lower()
    scope = _to_clean_str(raw.get("scope")).lower()
    context_condition = _to_clean_str(raw.get("context_condition")).lower()
    condition_note = _to_clean_str(raw.get("condition_note"))
    manifestation = _to_clean_str(raw.get("manifestation")).lower()

    if target_type not in ALLOWED_TARGET_TYPES:
        warnings.append("invalid_target_type")
    if polarity not in ALLOWED_POLARITY:
        warnings.append("invalid_polarity")
    if scope not in ALLOWED_SCOPE:
        warnings.append("invalid_scope")
    if context_condition not in ALLOWED_CONTEXT_CONDITION:
        warnings.append("invalid_context_condition")
    if manifestation not in ALLOWED_MANIFESTATION:
        warnings.append("invalid_manifestation")

    if not target:
        warnings.append("missing_target")
    lower_target = target.lower()
    for hint in FORBIDDEN_TARGET_HINTS:
        if hint in lower_target:
            warnings.append("target_contains_input_side_hint")
            break
    if len(target.split()) < 2:
        warnings.append("target_too_short")

    if context_condition == "conditional" and not condition_note:
        warnings.append("conditional_without_condition_note")
    if context_condition != "conditional":
        condition_note = ""

    hard_errors = {
        "invalid_target_type",
        "invalid_polarity",
        "invalid_scope",
        "invalid_context_condition",
        "invalid_manifestation",
        "missing_target",
    }
    if any(w in hard_errors for w in warnings):
        return None, warnings

    claim = AtomicClaim(
        target_type=target_type,
        target=target,
        polarity=polarity,
        scope=scope,
        context_condition=context_condition,
        condition_note=condition_note,
        manifestation=manifestation,
    )
    return claim, warnings


def normalize_token_candidates(
    candidates: Iterable[Any],
    *,
    max_items: int = 48,
) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()

    def _consume_text(text: str) -> None:
        parts = re.split(r"[,\n;/]+", text)
        for part in parts:
            token = part.strip().strip("\"'`").lower()
            token = re.sub(r"\s+", " ", token)
            if not token:
                continue
            if len(token) < 2:
                continue
            if token in GENERIC_TOKEN_BLACKLIST:
                continue
            if re.fullmatch(r"[\W_]+", token):
                continue
            if re.fullmatch(r"\d+", token):
                continue
            if token not in seen:
                seen.add(token)
                normalized.append(token)
            if len(normalized) >= max_items:
                return

    for item in candidates:
        if len(normalized) >= max_items:
            break
        if isinstance(item, str):
            _consume_text(item)
            continue
        if isinstance(item, Sequence):
            for sub in item:
                if isinstance(sub, str):
                    _consume_text(sub)
                if len(normalized) >= max_items:
                    break

    return normalized[:max_items]


def infer_compile_status(
    claim: AtomicClaim,
    strong_tokens: Sequence[str],
    weak_tokens: Sequence[str],
    *,
    raw_status: str = "",
) -> str:
    lowered = raw_status.strip().lower()
    if lowered in {"success", "weak_success", "fail"}:
        if lowered == "success" and (not strong_tokens or not weak_tokens):
            return "weak_success" if weak_tokens else "fail"
        return lowered

    if strong_tokens and weak_tokens:
        return "success"
    if weak_tokens:
        return "weak_success"
    if claim.manifestation == "generation":
        return "weak_success"
    return "fail"


def add_simple_lexical_expansion(tokens: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for token in tokens:
        base = token.strip().lower()
        if not base:
            continue
        expanded.append(base)
        if " " in base:
            expanded.append(base.replace(" ", "_"))
            expanded.append(base.replace(" ", "-"))
        if base.endswith("y") and len(base) > 3:
            expanded.append(base[:-1] + "ies")
        elif base.endswith("s") and len(base) > 3:
            expanded.append(base[:-1])
        else:
            expanded.append(base + "s")
    return normalize_token_candidates(expanded, max_items=96)

