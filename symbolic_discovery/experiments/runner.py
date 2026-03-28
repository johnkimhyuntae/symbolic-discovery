import csv
import time
import argparse
import os
import pandas as pd
import numpy as np
import re
from typing import List, Optional
from symbolic_discovery.data.benchmarks import (
    get_benchmark_config,
    list_benchmark_equations,
    list_feynman_dimensionless_equations,
    load_benchmark_df_and_pretty_map,
)
from symbolic_discovery.data.catalogue import CATALOGUE, DatasetConfig
from symbolic_discovery.data.synthetic import DatasetGenerator, split_train_test
from symbolic_discovery.solvers import SOLVER_REGISTRY

def get_data_config(ds_arg: str, target_arg: Optional[str] = None) -> DatasetConfig:
    """
    Resolves whether the argument is a Catalogue ID or a Custom CSV.
    """
    # Synthetic
    if ds_arg in CATALOGUE:
        return CATALOGUE[ds_arg]
    
    # CSV
    if os.path.exists(ds_arg) and ds_arg.endswith('.csv'):
        if not target_arg:
            raise ValueError(f"You must provide --target when using a custom file: {ds_arg}")
        
        return DatasetConfig(
            id=os.path.basename(ds_arg),
            variables=[], # Will be inferred later
            target=target_arg,
            formula="Unknown",
            domain={}
        )
    
    raise ValueError(f"Dataset argument '{ds_arg}' is neither a valid Catalogue ID nor a CSV file.")


def _maybe_parse_benchmark_arg(ds_arg: str) -> Optional[dict[str, object]]:
    """Parse benchmark dataset arguments.

    Supported forms:
      - feynman:<eq_id>                (dimensionless / without_units)
      - feynman:dim:<eq_id>            (dimensional / with_units)
      - feynman:dimless:<eq_id>        (explicit dimensionless)
      - bonus:<eq_id>                  (dimensionless / without_units)
      - bonus:dim:<eq_id>              (dimensional / with_units)
      - feynman:all / feynman:*        (dimensionless feynman sweep)
      - feynman:dim:all                (dimensional feynman sweep)
      - bonus:all                      (dimensionless bonus sweep)
      - bonus:dim:all                  (dimensional bonus sweep)

    Returns dict with keys: family, dimensionless, eq_id, all.
    """
    if not isinstance(ds_arg, str):
        return None

    for family in ("feynman", "bonus"):
        prefix = f"{family}:"
        if not ds_arg.startswith(prefix):
            continue

        rest = ds_arg[len(prefix):].strip()
        if not rest:
            return None

        parts = rest.split(":")
        dimensionless = True

        if parts and parts[0] in ("dim", "dimensional", "with_units", "with-units"):
            dimensionless = False
            parts = parts[1:]
        elif parts and parts[0] in ("dimless", "dimensionless", "without_units", "without-units"):
            dimensionless = True
            parts = parts[1:]

        if not parts or not parts[0].strip():
            return None

        token = parts[0].strip()
        is_all = token in ("all", "*")
        eq_id = None if is_all else token

        return {
            "family": family,
            "dimensionless": dimensionless,
            "eq_id": eq_id,
            "all": is_all,
        }

    return None


def _inject_target_noise(df: pd.DataFrame, target_col: str, noise_level: float, seed: int) -> pd.DataFrame:
    if noise_level <= 0.0:
        return df
    if target_col not in df.columns:
        return df

    y = df[target_col].to_numpy()
    y_range = float(np.max(y) - np.min(y))
    scale = y_range if y_range > 1e-9 else 1.0
    rng = np.random.default_rng(seed)
    noisy = y + rng.normal(loc=0.0, scale=noise_level * scale, size=len(y))
    out = df.copy()
    out[target_col] = noisy
    return out


def _pretty_equation(eq: str, pretty_map: Optional[dict[str, str]]) -> str:
    if not eq or not pretty_map:
        return eq

    # Replace longer tokens first (x1³, x1², then x1) to avoid partial overlaps.
    keys = sorted(pretty_map.keys(), key=len, reverse=True)
    out = eq
    for k in keys:
        v = pretty_map[k]
        if not v or v == k:
            continue
        # Word-boundary replacement for safe symbols (x1, x2, ...).
        out = re.sub(rf"\b{re.escape(k)}\b", v, out)
    return out

def run_experiment(args):
    # Expand benchmark selectors into explicit equation IDs.
    expanded_datasets: list[str] = []
    for ds in args.datasets:
        if ds in ("feynman:all_full", "feynman:full", "feynman:everything"):
            for family, dimensionless in (
                ("feynman", True),
                ("feynman", False),
                ("bonus", True),
                ("bonus", False),
            ):
                if family == "feynman" and dimensionless:
                    eqs = list_feynman_dimensionless_equations(
                        root_dir=args.feynman_root,
                        require_data_file=True,
                    )
                else:
                    eqs = list_benchmark_equations(
                        root_dir=args.feynman_root,
                        family=family,
                        dimensionless=dimensionless,
                        require_data_file=True,
                    )
                variant = "dimless" if dimensionless else "dim"
                expanded_datasets.extend([f"{family}:{variant}:{eq}" for eq in eqs])
            continue

        parsed = _maybe_parse_benchmark_arg(ds)
        if parsed and bool(parsed.get("all")):
            family = str(parsed["family"])
            dimensionless = bool(parsed["dimensionless"])

            # Back-compat: feynman:all remains the historical dimensionless subset.
            if family == "feynman" and dimensionless and ds in ("feynman:all", "feynman:*"):
                eqs = list_feynman_dimensionless_equations(root_dir=args.feynman_root, require_data_file=True)
            else:
                eqs = list_benchmark_equations(
                    root_dir=args.feynman_root,
                    family=family,
                    dimensionless=dimensionless,
                    require_data_file=True,
                )

            variant = "dimless" if dimensionless else "dim"
            expanded_datasets.extend([f"{family}:{variant}:{eq}" for eq in eqs])
        else:
            expanded_datasets.append(ds)
    args.datasets = expanded_datasets

    # Setup Output
    fieldnames = ["run_id", "dataset", "method", "noise", "seed", "found_eq", "found_eq_raw", "r2", "time_s", "status"]

    def _fmt_noise(n: float) -> str:
        s = f"{n:.4f}".rstrip('0').rstrip('.')
        return s if s else "0"

    output_path = args.output
    output_dir = args.output_dir
    per_run_dir = None
    shard_dir = None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "combined.csv")
        if args.per_run_csv:
            per_run_dir = os.path.join(output_dir, "runs")
            os.makedirs(per_run_dir, exist_ok=True)
        if args.shard_by != "none":
            shard_dir = os.path.join(output_dir, "shards")
            os.makedirs(shard_dir, exist_ok=True)

    file_exists = os.path.isfile(output_path)

    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        print(f"Running: {args.models} on {args.datasets}")
        
        for ds_arg in args.datasets:
            parsed_bench = _maybe_parse_benchmark_arg(ds_arg)
            # Expanded form uses family:dimless:<id> or family:dim:<id>
            if parsed_bench and parsed_bench.get("eq_id") is None and not bool(parsed_bench.get("all")):
                parsed_bench = None

            if isinstance(ds_arg, str):
                # Accept expanded three-part form: family:dimless:<eq_id>
                m = re.match(r"^(feynman|bonus):(dimless|dim):(.+)$", ds_arg)
                if m:
                    parsed_bench = {
                        "family": m.group(1),
                        "dimensionless": (m.group(2) == "dimless"),
                        "eq_id": m.group(3),
                        "all": False,
                    }

            bench_pretty_map: Optional[dict[str, str]] = None
            bench_dataset_id: Optional[str] = None

            # Load Data Configuration
            try:
                if parsed_bench and parsed_bench.get("eq_id"):
                    family = str(parsed_bench["family"])
                    dimensionless = bool(parsed_bench["dimensionless"])
                    eq_id = str(parsed_bench["eq_id"])
                    config = get_benchmark_config(
                        eq_id,
                        root_dir=args.feynman_root,
                        family=family,
                        dimensionless=dimensionless,
                        target=args.feynman_target,
                    )
                    variant = "dimless" if dimensionless else "dim"
                    bench_dataset_id = f"{family}:{variant}:{eq_id}"
                else:
                    config = get_data_config(ds_arg, args.target)
            except ValueError as e:
                print(f"Skipping {ds_arg}: {e}")
                continue

            for noise in args.noise:
                for model_name in args.models:
                    if model_name not in SOLVER_REGISTRY:
                        print(f"Warning: Model '{model_name}' not found.")
                        continue
                    
                    # Initialise Solver Strategy
                    # We pass noise so the wrapper can adjust thresholds (e.g. BACON strictness)
                    solver_class = SOLVER_REGISTRY[model_name]
                    solver = solver_class(noise_level=noise, verbose=args.verbose)

                    for seed in args.seeds:
                        dataset_id_for_logs = bench_dataset_id or config.id
                        run_id = f"{model_name}_{dataset_id_for_logs}_N{noise}_S{seed}"
                        
                        # 1. Prepare Data
                        if bench_dataset_id and parsed_bench:
                            try:
                                family = str(parsed_bench["family"])
                                dimensionless = bool(parsed_bench["dimensionless"])
                                eq_id = str(parsed_bench["eq_id"])
                                full_df, bench_pretty_map = load_benchmark_df_and_pretty_map(
                                    eq_id,
                                    root_dir=args.feynman_root,
                                    family=family,
                                    dimensionless=dimensionless,
                                    n_samples=args.feynman_n_samples,
                                    seed=seed,
                                    target=args.feynman_target,
                                )
                                full_df = _inject_target_noise(full_df, config.target, noise, seed)
                                train_df, _ = split_train_test(full_df, seed=seed)
                            except Exception as e:
                                print(f"Failed to load {bench_dataset_id}: {e}")
                                continue

                        elif ds_arg in CATALOGUE:
                            # Use Generator
                            gen = DatasetGenerator(seed=seed)
                            train_df, _, _ = gen.generate(config.id, noise)
                        else:
                            # Load Custom CSV
                            try:
                                full_df = pd.read_csv(ds_arg)
                                # Simple random split for custom data
                                train_df = full_df.sample(frac=0.75, random_state=seed)
                            except Exception as e:
                                print(f"Failed to load custom CSV {ds_arg}: {e}")
                                continue

                        # 2. Run Solve
                        result = solver.solve(train_df, config.target, seed)

                        found_eq_to_save = result.equation
                        if bench_pretty_map and result.status == "Success":
                            found_eq_to_save = _pretty_equation(found_eq_to_save, bench_pretty_map)
                        
                        # 3. Log
                        row = {
                            "run_id": run_id,
                            "dataset": dataset_id_for_logs,
                            "method": model_name,
                            "noise": noise,
                            "seed": seed,
                            "found_eq": found_eq_to_save,
                            "found_eq_raw": result.raw_equation,
                            "r2": f"{result.train_r2:.4f}",
                            "time_s": f"{result.time_sec:.2f}",
                            "status": result.status
                        }
                        writer.writerow(row)
                        f.flush()

                        # Optional: write separate per-run CSVs (one row each)
                        if per_run_dir:
                            per_run_path = os.path.join(per_run_dir, f"{run_id}.csv")
                            if (not os.path.exists(per_run_path)) or args.overwrite_runs:
                                with open(per_run_path, 'w', newline='') as rf:
                                    r_writer = csv.DictWriter(rf, fieldnames=fieldnames)
                                    r_writer.writeheader()
                                    r_writer.writerow(row)

                        # Optional: shard outputs (append rows into shard CSVs)
                        if shard_dir:
                            shard_name = None
                            if args.shard_by == "noise":
                                shard_name = f"N{_fmt_noise(noise)}.csv"
                            elif args.shard_by == "seed":
                                shard_name = f"S{seed}.csv"
                            elif args.shard_by == "noise_seed":
                                shard_name = f"N{_fmt_noise(noise)}_S{seed}.csv"

                            if shard_name:
                                shard_path = os.path.join(shard_dir, shard_name)
                                shard_exists = os.path.isfile(shard_path)
                                with open(shard_path, 'a', newline='') as sf:
                                    s_writer = csv.DictWriter(sf, fieldnames=fieldnames)
                                    if not shard_exists:
                                        s_writer.writeheader()
                                    s_writer.writerow(row)
                        
                        # Console Feedback
                        pretty_eq = _pretty_equation(found_eq_to_save or "", bench_pretty_map)
                        eq_preview = (pretty_eq or "").replace("\n", " ").strip()
                        if len(eq_preview) > 160:
                            eq_preview = eq_preview[:157] + "..."
                        print(f"[{model_name}] {dataset_id_for_logs} (N={noise}, S={seed}) -> {result.status} (R2: {result.train_r2:.4f})")
                        if eq_preview and eq_preview not in ("No law found", "Error"):
                            print(f"    Eq: {eq_preview}")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Symbolic Regression Models (BACON/PySR)")

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=SOLVER_REGISTRY.keys(),
        help="List of models to run (e.g. bacon3 bacon7)",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=(
            "List of Catalogue IDs (S-1, T-1) OR paths to .csv files OR "
            "feynman:<equation_id> (e.g. feynman:I.18.12)"
        ),
    )

    parser.add_argument(
        "--noise",
        nargs="+",
        type=float,
        default=[0.0],
        help="Noise levels to inject (e.g. 0.0 0.01 0.05)",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for reproducibility",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name (REQUIRED if using custom .csv files)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="experiment_results.csv",
        help="Output CSV file path",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "If set, write a combined CSV to <dir>/combined.csv and also write one CSV per run_id "
            "under <dir>/runs/"
        ),
    )

    parser.add_argument(
        "--shard-by",
        type=str,
        default="none",
        choices=["none", "noise", "seed", "noise_seed"],
        help=(
            "When using --output-dir, also write sharded CSVs under <dir>/shards/ "
            "(e.g. one per noise+seed)"
        ),
    )

    parser.add_argument(
        "--per-run-csv",
        action="store_true",
        help="When using --output-dir, also write one CSV per run_id under <dir>/runs/",
    )

    parser.add_argument(
        "--overwrite-runs",
        action="store_true",
        help="When using --per-run-csv, overwrite existing per-run CSVs if present",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from BACON algorithms",
    )

    # Feynman dataset options
    parser.add_argument(
        "--feynman-root",
        type=str,
        default="feynman",
        help="Root directory containing FeynmanEquationsDimensionless.csv and Feynman_without_units/",
    )
    parser.add_argument(
        "--feynman-n-samples",
        type=int,
        default=400,
        help="Number of rows to load from each Feynman equation file",
    )
    parser.add_argument(
        "--feynman-target",
        type=str,
        default="y",
        help="Target column name to use when loading Feynman datasets",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()