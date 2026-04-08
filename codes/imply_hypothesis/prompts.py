from __future__ import annotations

import json
from typing import Any, Dict, List

from .hypothesis_contract import AtomicClaim


def build_hypothesis_messages(
    *,
    evidence: Dict[str, Any],
    max_claims: int,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are an expert for SAE output-side hypothesis construction.\n"
        "Your task is to generate C(H) from output-side evidence only.\n"
        "Return exactly one JSON object. No markdown.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "summary": "one concise sentence",\n'
        '  "claims": [\n'
        "    {\n"
        '      "target_type": "token_family | semantic_direction | style_or_function | task_behavior",\n'
        '      "target": "specific concrete target phrase",\n'
        '      "polarity": "promote | suppress",\n'
        '      "scope": "focused | moderate | diffuse",\n'
        '      "context_condition": "always | positive_only | conditional",\n'
        '      "condition_note": "required only when context_condition=conditional, else empty string",\n'
        '      "manifestation": "logits | generation | both"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Hard constraints:\n"
        f"- claims length: 1 to {max_claims}\n"
        "- include at least one promote claim and at least one suppress claim\n"
        "- each claim expresses one output-side effect only\n"
        "- morphological variants from the same lemma/stem must stay in ONE claim, not split across claims\n"
        "- if a token family includes multiple forms (e.g., conclude/conclusion/concluding), merge them into one target description\n"
        "- do not mention input-side signals such as activations/max_token/input-side traces\n"
        "- target must be specific, avoid broad wording like general themes\n"
        "- if uncertain, prefer conservative scope/context rather than inventing details\n"
        "- if context_condition is not conditional, set condition_note to empty string\n"
        "- bad split example: claim1 target=conclusion, claim2 target=conclude\n"
        "- good merged example: one claim target=conclude/conclusion token family"
    )

    user_prompt = (
        "Construct C(H) from the following initial evidence.\n"
        "Use the evidence and avoid unsupported claims.\n\n"
        f"{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_compile_seed_messages(
    *,
    claim: AtomicClaim,
    evidence: Dict[str, Any],
) -> List[Dict[str, str]]:
    system_prompt = (
        "You compile a single atomic claim into token sets.\n"
        "Return exactly one JSON object. No markdown.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "S_strong": ["canonical indicators, high precision"],\n'
        '  "S_weak": ["paraphrases, inflections, near indicators"],\n'
        '  "compile_status": "success | weak_success | fail"\n'
        "}\n\n"
        "Rules:\n"
        "- provide short lexical items that can be matched in text/token space\n"
        "- avoid stopwords and generic words\n"
        "- S_strong should be strict and precise\n"
        "- S_weak can be broader but still semantically aligned\n"
        "- if claim is mostly style/behavior, weak_success is allowed"
    )
    user_prompt = (
        "Compile this claim into token indicators.\n"
        f"Claim:\n{json.dumps(claim.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        "Reference evidence:\n"
        f"{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_compile_expand_messages(
    *,
    claim: AtomicClaim,
    seed_strong: List[str],
    seed_weak: List[str],
) -> List[Dict[str, str]]:
    system_prompt = (
        "Expand token candidates for one claim.\n"
        "Return exactly one JSON object. No markdown.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "extra_strong": ["optional additional canonical indicators"],\n'
        '  "extra_weak": ["optional synonyms, inflections, aliases, near indicators"]\n'
        "}\n\n"
        "Rules:\n"
        "- keep high semantic alignment with the claim target\n"
        "- do not output generic words\n"
        "- do not include long phrases over 4 words"
    )
    user_prompt = (
        "Expand candidates based on the claim and current seed sets.\n"
        f"Claim:\n{json.dumps(claim.to_dict(), ensure_ascii=False, indent=2)}\n"
        f"Current S_strong seed: {json.dumps(seed_strong, ensure_ascii=False)}\n"
        f"Current S_weak seed: {json.dumps(seed_weak, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
