from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from imply_scoring import (
    DEFAULT_LOGS_ROOT,
    DEFAULT_STEER_ROOT,
    TopKPolarityAccuracyConfig,
    build_steer_json_path,
    compute_topk_polarity_accuracy,
    find_latest_compile_tokens_path,
    load_compiled_claim_entries,
    load_step0_traces_from_steer_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute TopK_Polarity_Accuracy (PA_obs without coverage term) from "
            "compiled claim tokens and step0 top-k delta evidence."
        )
    )
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)

    parser.add_argument("--compile-tokens-path", default=None)
    parser.add_argument("--logs-root", default=str(DEFAULT_LOGS_ROOT))

    parser.add_argument("--steer-root", default=str(DEFAULT_STEER_ROOT))
    parser.add_argument("--intervention-scope", default="natural_support_mask")
    parser.add_argument("--steer-steps", type=int, default=1)
    parser.add_argument("--top-k-limit", type=int, default=None)

    parser.add_argument("--weak-weight", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--output-path", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    compile_path = _resolve_compile_tokens_path(
        layer_id=args.layer_id,
        feature_id=args.feature_id,
        compile_tokens_path=args.compile_tokens_path,
        logs_root=Path(args.logs_root),
    )
    steer_json_path = build_steer_json_path(
        layer_id=args.layer_id,
        feature_id=args.feature_id,
        intervention_scope=args.intervention_scope,
        steer_steps=args.steer_steps,
        steer_root=Path(args.steer_root),
    )
    if not steer_json_path.exists():
        raise FileNotFoundError(f"steer_from_neuronpedia.json not found: {steer_json_path}")

    claims = load_compiled_claim_entries(compile_path)
    traces = load_step0_traces_from_steer_json(
        steer_json_path,
        top_k_limit=args.top_k_limit,
    )
    metric_result = compute_topk_polarity_accuracy(
        claims=claims,
        traces=traces,
        config=TopKPolarityAccuracyConfig(
            weak_weight=args.weak_weight,
            epsilon=args.epsilon,
        ),
    )

    output: Dict[str, Any] = {
        "metadata": {
            "layer_id": args.layer_id,
            "feature_id": args.feature_id,
            "compile_tokens_path": str(compile_path),
            "steer_json_path": str(steer_json_path),
            "intervention_scope": args.intervention_scope,
            "steer_steps": args.steer_steps,
            "top_k_limit": args.top_k_limit,
            "trace_count": len(traces),
            "claim_count": len(claims),
            "formula_variant": "PA_obs_without_coverage",
        },
        "TopK_Polarity_Accuracy": metric_result,
    }

    output_path = _resolve_output_path(
        output_path=args.output_path,
        compile_tokens_path=compile_path,
        layer_id=args.layer_id,
        feature_id=args.feature_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "metric": "TopK_Polarity_Accuracy",
                "hypothesis_score": metric_result.get("hypothesis_score"),
                "valid_claim_count": metric_result.get("valid_claim_count"),
                "trace_count": len(traces),
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _resolve_compile_tokens_path(
    *,
    layer_id: str,
    feature_id: str,
    compile_tokens_path: str | None,
    logs_root: Path,
) -> Path:
    if compile_tokens_path:
        path = Path(compile_tokens_path)
        if not path.exists():
            raise FileNotFoundError(f"compile-tokens file not found: {path}")
        return path
    return find_latest_compile_tokens_path(
        layer_id=layer_id,
        feature_id=feature_id,
        logs_root=logs_root,
    )


def _resolve_output_path(
    *,
    output_path: str | None,
    compile_tokens_path: Path,
    layer_id: str,
    feature_id: str,
) -> Path:
    if output_path:
        return Path(output_path)
    return (
        compile_tokens_path.parent
        / f"layer{layer_id}-feature{feature_id}-topk-polarity-accuracy.json"
    )


if __name__ == "__main__":
    main()

