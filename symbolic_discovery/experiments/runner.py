"""
Experiment runner.

Drives the benchmark loop: expand dataset selectors, resolve configs,
load data, run solvers, write results CSV.
"""

import csv
import argparse
import os
from typing import List, Optional

from symbolic_discovery.data import (
    expand_datasets,
    get_exclusion_reason,
    load,
    pretty_equation,
    resolve,
)
from symbolic_discovery.solvers import SOLVER_REGISTRY, SolverResult


def run_experiment(args):
    datasets = expand_datasets(args.datasets, args.feynman_root)

    fieldnames = [
        "run_id", "dataset", "method", "noise", "seed",
        "equation", "raw_equation", "r2", "mse", "mae", "time_s", "status",
    ]

    file_exists = os.path.isfile(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        print(f"Running: {args.models} on {datasets}")

        for key in datasets:
            try:
                config = resolve(key, args.feynman_root, args.target)
            except ValueError as e:
                print(f"Skipping {key}: {e}")
                continue

            for noise in args.noise:
                for model_name in args.models:
                    if model_name not in SOLVER_REGISTRY:
                        print(f"Warning: Model '{model_name}' not found.")
                        continue

                    for seed in args.seeds:
                        run_id = f"{model_name}_{config.key}_N{noise}_S{seed}"
                        is_bacon = model_name.startswith("bacon")

                        # 0. Check exclusions
                        # TBD, because we standardise operator set
                        exclusion = get_exclusion_reason(config) if is_bacon else None

                        if exclusion:
                            result = SolverResult(
                                equation="No law found",
                                raw_equation=(
                                    f"Excluded: equation contains operators BACON cannot discover"
                                    if exclusion != "complex"
                                    else f"Excluded: equation is too complex for BACON"
                                ),
                                r2=0.0,
                                mse=float("inf"),
                                mae=float("inf"),
                                time_sec=0.0,
                                status="Failure",
                            )
                            bench_pretty_map = None
                        else:
                            # 1. Prepare data
                            try:
                                train_df, test_df, bench_pretty_map = load(
                                    config,
                                    noise=noise,
                                    seed=seed,
                                    n_samples=args.n_samples,
                                    feynman_root=args.feynman_root,
                                )
                            except Exception as e:
                                print(f"Failed to load {config.key}: {e}")
                                continue

                            # 2. Construct solver
                            solver_kwargs: dict = {
                                "verbose": args.verbose,
                            }

                            solver_class = SOLVER_REGISTRY[model_name]
                            solver = solver_class(**solver_kwargs)

                            # 3. Run
                            result = solver.solve(train_df, test_df, config.target, seed)

                        # 4. Pretty-print
                        equation = result.equation
                        if bench_pretty_map and result.status == "Found":
                            equation = pretty_equation(
                                result.equation, bench_pretty_map
                            )

                        # 5. Write CSV row
                        writer.writerow({
                            "run_id": run_id,
                            "dataset": config.key,
                            "method": model_name,
                            "noise": noise,
                            "seed": seed,
                            "equation": equation,
                            "raw_equation": result.raw_equation,
                            "r2": f"{result.r2:.4f}",
                            "mse": f"{result.mse:.4f}",
                            "mae": f"{result.mae:.4f}",
                            "time_s": f"{result.time_sec:.2f}",
                            "status": result.status,
                        })
                        f.flush()

                        # 6. Console output
                        pretty_eq = pretty_equation(
                            equation or "", bench_pretty_map
                        )
                        eq_preview = (pretty_eq or "").replace("\n", " ").strip()
                        if len(eq_preview) > 160:
                            eq_preview = eq_preview[:157] + "..."

                        print(
                            f"[{model_name}] {config.key} "
                            f"(N={noise}, S={seed}) -> "
                            f"{result.status} (R²={result.r2:.4f})"
                        )
                        if eq_preview and eq_preview not in (
                            "No law found", "Error",
                        ):
                            print(f"    Eq: {eq_preview}")
                        else:
                            print(f"    Details: {result.raw_equation}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run symbolic regression benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
dataset selectors:
    S1..S4          individual synthetic datasets
    T1..T5          individual textbook laws
    F1..F100        individual Feynman equations (by number)
    B1..B20         individual Bonus equations (by number)
    S / T / F / B   all datasets in that family
    [custom].csv    path to custom CSV file (requires --target)

examples:
    %(prog)s --models bacon3f bacon7f --datasets S T --verbose
    %(prog)s --models bacon3f --datasets F1 F8 F12 --seeds 67 73
    %(prog)s --models bacon3f bacon7f --datasets F --noise 0.0 0.01 0.05
    %(prog)s --models bacon3f --datasets mydata.csv --target y""",
    )

    parser.add_argument(
        "--models", nargs="+", required=True,
        choices=SOLVER_REGISTRY.keys(),
        help="Solvers to run",
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="Dataset selectors (see below)",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Target column name (required for custom .csv files)",
    )
    parser.add_argument(
        "--noise", nargs="+", type=float, default=[0.0],
        help="Noise levels to inject (default: 0.0)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Random seeds (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default="experiment_results.csv",
        help="Output CSV path (default: experiment_results.csv)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose solver output",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Rows per dataset",
    )
    parser.add_argument(
        "--feynman-root", type=str, default="feynman",
        help="Root directory for Feynman/Bonus data files",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()