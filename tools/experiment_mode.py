"""Read-only CLI for named experimental modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

THIS = Path(__file__).resolve()
BASE_DIR = THIS.parents[1] if THIS.parent.name == "tools" else THIS.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from runtime.experiment_modes import (  # noqa: E402
    apply_experiment_mode,
    describe_experiment_mode,
    get_experiment_mode,
    list_experiment_modes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect read-only experimental mode overrides")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List supported experiment modes")
    group.add_argument("--describe", metavar="MODE", help="Describe a mode and its env overrides")
    group.add_argument("--print-env", metavar="MODE", help="Print mode overrides as KEY=value lines")
    group.add_argument("--json", metavar="MODE", dest="json_mode", help="Print a mode definition as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.list:
            for name in list_experiment_modes():
                print(name)
            return 0

        if args.describe:
            print(describe_experiment_mode(args.describe))
            return 0

        if args.print_env:
            env = apply_experiment_mode(args.print_env, base_env={})
            for key, value in env.items():
                print(f"{key}={value}")
            return 0

        if args.json_mode:
            mode = get_experiment_mode(args.json_mode)
            print(json.dumps(mode.as_dict(), indent=2))
            return 0
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
