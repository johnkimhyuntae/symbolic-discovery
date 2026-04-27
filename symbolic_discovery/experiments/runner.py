"""
Experiment runner.

Drives the benchmark loop: resolves variants and suite files 
into a flat list of Runs, then for each Run loads data, instantiates 
the solver wrapper with the variant's kwargs, executes solve(), and 
appends the result to the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from .suite import (
    Run,
    Variant,
    expand_to_runs,
    load_suite_file,
    parse_sweep_spec,
    parse_variant_spec,
    variants_from_suite,
)


# Run construction

def _build_runs(args) -> List[Run]:
    """
    Resolve CLI flags + suite file into a flat list of Runs.

    CLI flags take precedence over suite-file values for every axis.
    --models is treated as sugar for '--variant X=X' (default kwargs).
    """
    suite = load_suite_file(args.suite) if args.suite else {}

    variants: List[Variant] = []
    for m in (args.models or []):
        variants.append(Variant(name=m, model=m, params={}))
    for spec in (args.variant or []):
        variants.append(parse_variant_spec(spec))
    for spec in (args.sweep or []):
        variants.extend(parse_sweep_spec(spec))
    if not variants:
        variants.extend(variants_from_suite(suite))

    datasets    = args.datasets    or suite.get("datasets", [])
    noise       = args.noise       or suite.get("noise", [0.0])
    noise_types = args.noise_types or suite.get("noise_types", ["multiplicative"])
    n_samples   = args.n_samples   or suite.get("n_samples", [1000])
    seeds       = args.seeds       or suite.get("seeds", [42])

    if not variants:
        raise ValueError(
            "No variants resolved. Supply --models, --variant, --sweep, "
            "or a --suite file with a 'variants' section."
        )
    if not datasets:
        raise ValueError(
            "No datasets resolved. Supply --datasets or a --suite file "
            "with a 'datasets' section."
        )

    return expand_to_runs(
        variants=variants, datasets=datasets, noise=noise,
        noise_types=noise_types, n_samples=n_samples, seeds=seeds,
    )


# Main loop

def run_experiment(args) -> None:
    runs = _build_runs(args)

    selectors = sorted({r.dataset for r in runs})

    # Map original selector to expanded list, so 'F' fans out per Run.
    fanout: dict[str, list[str]] = {}
    for sel in selectors:
        fanout[sel] = expand_datasets([sel], args.feynman_root)

    fieldnames = [
        "run_id", "dataset", "method", "variant", "params_json",
        "noise", "noise_type", "n_samples", "seed",
        "equation", "raw_equation", "r2", "mse", "mae", "time_s", "status",
    ]

    output_path = os.path.join(args.output_root, args.output)

    file_exists = os.path.isfile(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Total run count after dataset selector fanout, for progress.
        total = sum(len(fanout[r.dataset]) for r in runs)
        print(f"Total runs (after dataset fanout): {total}")

        for run in runs:
            for ds_key in fanout[run.dataset]:
                _execute_one(run, ds_key, args, writer)


def _execute_one(run: Run, ds_key: str, args, writer) -> None:
    """Execute a single Run on a single dataset key."""
    v = run.variant

    try:
        config = resolve(ds_key, args.feynman_root, args.target)
    except ValueError as e:
        print(f"Skipping {ds_key}: {e}")
        return

    run_id = (
        f"{v.name}_{config.key}"
        f"_N{run.noise}{run.noise_type[:1]}"
        f"_n{run.n_samples}_S{run.seed}"
    )
    is_bacon = v.model.startswith("bacon")
    exclusion = get_exclusion_reason(config) if is_bacon else None

    # Excluded equations short-circuit before data loading for BACONs
    if exclusion:
        result = SolverResult(
            equation="No law found",
            raw_equation=(
                "Excluded: equation contains operators BACON cannot discover"
                if exclusion != "complex"
                else "Excluded: equation is too complex for BACON"
            ),
            r2=0.0,
            mse=float("inf"),
            mae=float("inf"),
            time_sec=0.0,
            status="Failure",
        )
        bench_pretty_map = None
    else:
        train_df, test_df, bench_pretty_map = load(
            config,
            noise=run.noise,
            noise_type=run.noise_type,
            n_samples=run.n_samples,
            seed=run.seed,
            feynman_root=args.feynman_root,
        )

        SolverClass = SOLVER_REGISTRY[v.model]
        solver = SolverClass(verbose=args.verbose, **v.params)
        result = solver.solve(train_df, test_df, config.target, run.seed)

    _write_row(writer, run_id, run, config, v, result)
    _print_progress(run_id, run, v, config, result, bench_pretty_map)


def _write_row(writer, run_id, run, config, v, result) -> None:
    writer.writerow({
        "run_id": run_id,
        "dataset": config.key,
        "method": v.model,
        "variant": v.name,
        "params_json": json.dumps(v.params, sort_keys=True),
        "noise": run.noise,
        "noise_type": run.noise_type,
        "n_samples": run.n_samples,
        "seed": run.seed,
        "equation": result.equation,
        "raw_equation": result.raw_equation,
        "r2": result.r2,
        "mse": result.mse,
        "mae": result.mae,
        "time_s": result.time_sec,
        "status": result.status,
    })


def _print_progress(run_id, run, v, config, result, bench_pretty_map) -> None:
    pretty_eq = pretty_equation(result.equation or "", bench_pretty_map)
    eq_preview = (pretty_eq or "").replace("\n", " ").strip()
    if len(eq_preview) > 160:
        eq_preview = eq_preview[:157] + "..."

    print(
        f"[{v.name}] {config.key} "
        f"(N={run.noise}/{run.noise_type}, n={run.n_samples}, S={run.seed}) -> "
        f"{result.status} (R²={result.r2:.4f})"
    )
    if eq_preview and eq_preview not in ("No law found", "Error"):
        print(f"    Eq: {eq_preview}")
    else:
        print(f"    Details: {result.raw_equation}")


# CLI

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
    # Defaults — same as before
    %(prog)s --models bacon3f bacon7f --datasets S T

    # One-knob hyperparameter sweep
    %(prog)s --datasets S T --sweep bacon7f.n_folds=1,3,5,7 --seeds 42 43 44

    # Named ablation variants
    %(prog)s --datasets S T --noise 0.0 0.05 \\
        --variant full=bacon7f \\
        --variant no_voting=bacon7f:n_folds=1 \\
        --variant no_relax=bacon7f:scale_factor=1.0 \\
        --variant baseline=bacon3f

    # Full preregistered suite from a suite file
    %(prog)s --suite experiments/noise_robustness.yaml --output noise.csv

    # Sample-efficiency curve
    %(prog)s --models bacon7f pysr --datasets T1 T2 \\
        --n-samples 50 100 250 500 1000 --seeds 42 43 44""",
    )

    # Solver / variant specification
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=SOLVER_REGISTRY.keys(),
        help="Solvers to run with default kwargs. Sugar for '--variant X=X'.",
    )
    parser.add_argument(
        "--variant", action="append", default=[], metavar="SPEC",
        help="Named solver variant. Format: 'name=model[:k=v,k=v,...]'. "
             "Repeatable. Example: --variant no_voting=bacon7f:n_folds=1",
    )
    parser.add_argument(
        "--sweep", action="append", default=[], metavar="SPEC",
        help="One-knob hyperparameter sweep. Format: 'model.param=v1,v2,...'. "
             "Repeatable. Example: --sweep bacon7f.n_folds=1,3,5,7",
    )
    parser.add_argument(
        "--suite", type=str, default=None, metavar="PATH",
        help="YAML suite file specifying variants, datasets, noise, "
            "noise_types, n_samples, and seeds. CLI flags override suite fields.",
    )

    # Data axes
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Dataset selectors (see below). Required unless given by --suite.",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Target column name (required for custom .csv files)",
    )
    parser.add_argument(
        "--noise", nargs="+", type=float, default=None,
        help="Noise levels to inject. Default: 0.0",
    )
    parser.add_argument(
        "--noise-types", nargs="+", default=None,
        choices=["gaussian", "multiplicative", "heavy_tail", "outliers"],
        help="Noise distributions to inject. Default: multiplicative",
    )
    parser.add_argument(
        "--n-samples", nargs="+", type=int, default=None,
        help="Rows per dataset. Accepts multiple values to sweep. Default: 1000",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Random seeds. Default: 42",
    )

    # Output and misc
    parser.add_argument(
        "--output_root", type=str, default="results",
        help="Root directory for output CSV files (default: results)",
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
        "--feynman-root", type=str, default="feynman",
        help="Root directory for Feynman/Bonus data files",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    """Cross-flag validation for runner arguments."""
    if not (args.models or args.variant or args.sweep or args.suite):
        parser.error(
            "must supply at least one of --models, --variant, --sweep, --suite"
        )
    if not args.datasets and not args.suite:
        parser.error("--datasets is required unless --suite provides them")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    run_experiment(args)


if __name__ == "__main__":
    main()