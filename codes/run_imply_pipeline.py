from __future__ import annotations

import argparse
import json
from pathlib import Path

from imply_hypothesis.pipeline import PipelineRunConfig, run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate C(H) and compile S_strong/S_weak from Neuronpedia output-side evidence."
    )
    parser.add_argument("--model-id", default="gemma-2-2b")
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--width", default="16k")
    parser.add_argument("--round-id", default="round_0")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--top-k-pairs", type=int, default=24)
    parser.add_argument("--max-claims", type=int, default=3)
    parser.add_argument("--neuronpedia-api-key", default=None)
    parser.add_argument("--timeout", type=int, default=30)

    parser.add_argument("--llm-config-path", default=str(Path("support_info") / "llm_api_info.py"))
    parser.add_argument("--llm-model-override", default=None)
    parser.add_argument("--hypothesis-temperature", type=float, default=0.2)
    parser.add_argument("--compile-temperature", type=float, default=0.1)
    parser.add_argument("--hypothesis-max-tokens", type=int, default=1200)
    parser.add_argument("--compile-max-tokens", type=int, default=900)
    parser.add_argument("--compile-max-weak-tokens", type=int, default=48)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = PipelineRunConfig(
        model_id=args.model_id,
        layer_id=args.layer_id,
        feature_id=args.feature_id,
        width=args.width,
        round_id=args.round_id,
        timestamp=args.timestamp,
        top_k_pairs=args.top_k_pairs,
        max_claims=args.max_claims,
        neuronpedia_api_key=args.neuronpedia_api_key,
        timeout=args.timeout,
        llm_config_path=Path(args.llm_config_path),
        llm_model_override=args.llm_model_override,
        hypothesis_temperature=args.hypothesis_temperature,
        compile_temperature=args.compile_temperature,
        hypothesis_max_tokens=args.hypothesis_max_tokens,
        compile_max_tokens=args.compile_max_tokens,
        compile_max_weak_tokens=args.compile_max_weak_tokens,
    )
    result = run_pipeline(config)
    print(
        json.dumps(
            {
                "summary": result["C_H"]["summary"],
                "claim_count": len(result["C_H"]["claims"]),
                "compiled_claim_count": len(result["compile_results"]),
                "token_usage": result["token_usage"],
                "result_path": result["paths"]["result"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

