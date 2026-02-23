from __future__ import annotations

import argparse
from typing import List, Optional

from ..experiments import runner
from ..experiments import view_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symbolic-discovery",
        description="Symbolic discovery experiment CLI (BACON.3/BACON.7)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run one or more models on one or more datasets")
    run_p.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the experiment runner",
    )

    view_p = sub.add_parser("view", help="View results CSVs")
    view_p.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the results viewer",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        runner.main(args.args)
        return

    if args.command == "view":
        view_results.main(args.args)
        return

    raise SystemExit(2)
