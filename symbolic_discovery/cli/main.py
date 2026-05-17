from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ..experiments import runner, viewer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symbolic-discovery",
        description="Symbolic discovery experiment CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Run one or more models on one or more datasets")
    sub.add_parser("view", help="View results CSVs")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        build_arg_parser().print_help()
        return

    command = argv[0]
    forwarded = argv[1:]

    if command == "run":
        runner.main(forwarded)
        return

    if command == "view":
        viewer.main(forwarded)
        return

    parser = build_arg_parser()
    parser.parse_args(argv)
    raise SystemExit(2)
