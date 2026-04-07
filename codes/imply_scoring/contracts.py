from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CompiledClaimEntry:
    claim_index: int
    target: str
    polarity: str
    S_strong: List[str]
    S_weak: List[str]
    raw_claim: Dict[str, Any]


@dataclass
class TokenDelta:
    token: str
    token_norm: str
    delta_logit: float


@dataclass
class Step0TraceEvidence:
    sample_rank: int
    intervention_index: int
    scale: float
    top_positive_delta_tokens: List[TokenDelta]
    top_negative_delta_tokens: List[TokenDelta]

