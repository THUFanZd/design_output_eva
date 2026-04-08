from __future__ import annotations

from typing import Tuple


def resolve_effective_polarity(claim_polarity: str, scale: float) -> str:
    """
    Apply sign-aware mapping:
    - claim=promote, scale>0 -> promote
    - claim=promote, scale<0 -> suppress
    - claim=suppress, scale>0 -> suppress
    - claim=suppress, scale<0 -> promote
    """
    polarity = str(claim_polarity).strip().lower()
    if polarity not in {"promote", "suppress"}:
        return polarity
    if scale < 0:
        return "suppress" if polarity == "promote" else "promote"
    return polarity


def pick_bucket_by_effective_polarity(effective_polarity: str) -> Tuple[str, bool]:
    """
    Returns:
      - bucket_name: top_positive_delta_tokens or top_negative_delta_tokens
      - use_positive: True if positive bucket, else False
    """
    if effective_polarity == "promote":
        return "top_positive_delta_tokens", True
    return "top_negative_delta_tokens", False

