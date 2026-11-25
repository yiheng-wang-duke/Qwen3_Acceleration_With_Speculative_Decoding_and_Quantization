"""Use python -m acceleratellm --help without installing the package."""
import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acceleratellm")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report environment and visible GPUs")
    for command, help_text in (
        ("validate", "Validate and print merged YAML"),
        ("check-models", "Check pinned target/draft tokenizers without loading weights"),
        ("baseline", "Run the autoregressive target baseline"),
        ("speculative", "Validate and compare AR versus speculative decoding"),
    ):
        item = sub.add_parser(command, help=help_text)
        item.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .experiments.artifacts import environment_report
            print(json.dumps(environment_report(), indent=2))
            return 0
        from .config import load_config
        config = load_config(args.config)
        if args.command == "validate":
            import yaml
            print(yaml.safe_dump(config, sort_keys=False), end="")
        elif args.command == "check-models":
            from .models.compatibility import check_model_pair
            print(json.dumps(check_model_pair(config["models"], config["runtime"]["local_files_only"]), indent=2))
        elif args.command == "baseline":
            from .experiments.baseline import run_baseline
            print(run_baseline(config).resolve())
        elif args.command == "speculative":
            from .experiments.speculative import run_speculative
            print(run_speculative(config).resolve())
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, FloatingPointError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
