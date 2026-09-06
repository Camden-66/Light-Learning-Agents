"""Operational checks for local LLM setup and evaluation-artifact commands."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

from .ollama import OllamaChatClient, OllamaError

REQUIRED_FREE_BYTES = 10 * 1024**3
DEFAULT_MODELS = ("qwen3:1.7b", "qwen3:4b")


def preflight(*, base_url: str, models: Sequence[str]) -> tuple[dict[str, object], bool]:
    """Check disk/server/model readiness without pulling or altering anything."""

    free_bytes = shutil.disk_usage(Path.cwd()).free
    report: dict[str, object] = {
        "required_free_bytes": REQUIRED_FREE_BYTES,
        "free_bytes": free_bytes,
        "disk_ready": free_bytes >= REQUIRED_FREE_BYTES,
        "base_url": base_url,
        "required_models": list(models),
    }
    try:
        with OllamaChatClient(base_url) as client:
            installed = client.list_models()
        installed_names = {
            str(item.get("name") or item.get("model")) for item in installed
        }
        missing = [model for model in models if model not in installed_names]
        report["ollama_ready"] = True
        report["installed_models"] = sorted(installed_names)
        report["missing_models"] = missing
    except OllamaError as error:
        report["ollama_ready"] = False
        report["ollama_error"] = str(error)
        report["missing_models"] = list(models)
    ready = bool(
        report["disk_ready"] and report["ollama_ready"] and not report["missing_models"]
    )
    return report, ready


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="light-learning")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser(
        "preflight", help="check local disk, Ollama server, and model availability"
    )
    preflight_parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    preflight_parser.add_argument("--model", dest="models", action="append")
    web_parser = subparsers.add_parser("web", help="interactive RoomEnv explainer")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8768)
    validate_parser = subparsers.add_parser(
        "validate-profiles",
        help="check evaluation profile determinism and theta stratification",
    )
    validate_parser.add_argument("--manifest", default=None)
    report_parser = subparsers.add_parser(
        "report", help="summarize JSONL episode records into a metrics summary"
    )
    report_parser.add_argument("records", nargs="+")
    report_parser.add_argument("--output", required=True)
    compare_parser = subparsers.add_parser(
        "compare",
        help="emergent world-model vs basic RL vs oracle MLE (not official eval)",
    )
    compare_parser.add_argument("--profile", default="pilot")
    compare_parser.add_argument("--seed", type=int, default=0)
    compare_parser.add_argument("--reinforce-episodes", type=int, default=120)
    compare_parser.add_argument("--pool-episodes", type=int, default=12)
    compare_parser.add_argument(
        "--budget",
        dest="budgets",
        type=int,
        action="append",
        help="repeat to choose budgets; default is 4 and 8",
    )
    args = parser.parse_args(argv)

    if args.command == "preflight":
        report, ready = preflight(
            base_url=args.base_url,
            models=tuple(args.models or DEFAULT_MODELS),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if ready else 2
    if args.command == "web":
        from .web_server import serve

        serve(host=args.host, port=args.port)
        return 0
    if args.command == "validate-profiles":
        from .evaluator import PROFILE_MANIFEST, validate_profiles

        result = validate_profiles(args.manifest or PROFILE_MANIFEST)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "report":
        from .evaluator import aggregate_metrics, read_records, write_summary

        write_summary(args.output, aggregate_metrics(read_records(args.records)))
        return 0
    if args.command == "compare":
        from .compare import run_organic_comparison

        budgets = tuple(args.budgets) if args.budgets else (4, 8)
        report = run_organic_comparison(
            profile=args.profile,
            seed=args.seed,
            reinforce_episodes=args.reinforce_episodes,
            pool_episodes=args.pool_episodes,
            budgets=budgets,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
