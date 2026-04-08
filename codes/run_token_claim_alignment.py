from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

from imply_scoring import (
    DEFAULT_LOGS_ROOT,
    DEFAULT_STEER_ROOT,
    TokenClaimAlignmentConfig,
    build_steer_json_path,
    compute_token_claim_alignment,
    find_latest_compile_tokens_path,
    load_compiled_claim_entries,
    load_step0_traces_from_steer_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Token_Claim_Alignment using LLM-as-a-judge on top-delta tokens "
            "from steer_from_neuronpedia step0 evidence."
        )
    )
    parser.add_argument("--layer-id", required=True)
    parser.add_argument("--feature-id", required=True)

    parser.add_argument("--compile-tokens-path", default=None)
    parser.add_argument("--logs-root", default=str(DEFAULT_LOGS_ROOT))

    parser.add_argument("--steer-root", default=str(DEFAULT_STEER_ROOT))
    parser.add_argument("--intervention-scope", default="natural_support_mask")
    parser.add_argument("--steer-steps", type=int, default=1)

    parser.add_argument("--llm-config-path", default=str(Path("support_info") / "llm_api_info.py"))
    parser.add_argument("--llm-model-override", default=None)
    parser.add_argument("--top-k-tokens-per-trace", type=int, default=10)
    parser.add_argument("--judge-repeats", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
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
    traces = load_step0_traces_from_steer_json(steer_json_path)
    metric_result = compute_token_claim_alignment(
        claims=claims,
        traces=traces,
        config=TokenClaimAlignmentConfig(
            llm_config_path=Path(args.llm_config_path),
            llm_model_override=args.llm_model_override,
            top_k_tokens_per_trace=args.top_k_tokens_per_trace,
            judge_repeats=args.judge_repeats,
            confidence_threshold=args.confidence_threshold,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            progress_every=args.progress_every,
            progress_callback=_build_progress_printer(enabled=args.show_progress),
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
            "trace_count": len(traces),
            "claim_count": len(claims),
            "top_k_tokens_per_trace": args.top_k_tokens_per_trace,
            "judge_repeats": args.judge_repeats,
            "confidence_threshold": args.confidence_threshold,
        },
        "Token_Claim_Alignment": metric_result,
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
                "metric": "Token_Claim_Alignment",
                "hypothesis_score": metric_result.get("hypothesis_score"),
                "valid_claim_count": metric_result.get("valid_claim_count"),
                "trace_count": len(traces),
                "output_path": str(output_path),
                "token_usage": metric_result.get("token_usage"),
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
        / f"layer{layer_id}-feature{feature_id}-token-claim-alignment.json"
    )


def _build_progress_printer(*, enabled: bool):
    if not enabled:
        return None

    def _printer(event: Dict[str, Any]) -> None:
        et = event.get("type")
        now = datetime.now().strftime("%H:%M:%S")
        if et == "start":
            print(
                f"[{now}] start | model={event.get('model')} | "
                f"claims={event.get('claim_count')} | traces={event.get('trace_count')} | "
                f"planned_judgements={event.get('total_judgements')}",
                flush=True,
            )
            return
        if et == "tick":
            done = event.get("done_judgements")
            total = event.get("total_judgements")
            pct = 0.0
            if isinstance(done, int) and isinstance(total, int) and total > 0:
                pct = done * 100.0 / total
            print(
                f"[{now}] progress | {done}/{total} ({pct:.1f}%) | "
                f"claim={event.get('claim_index')} | token={event.get('token')}",
                flush=True,
            )
            return
        if et == "claim_end":
            print(
                f"[{now}] claim_done | claim_index={event.get('claim_index')} | "
                f"{event.get('claim_order')}/{event.get('total_claims')}",
                flush=True,
            )
            return
        if et == "done":
            print(
                f"[{now}] done | judgements={event.get('done_judgements')}/{event.get('total_judgements')} | "
                f"hypothesis_score={event.get('hypothesis_score')}",
                flush=True,
            )

    return _printer


if __name__ == "__main__":
    main()
