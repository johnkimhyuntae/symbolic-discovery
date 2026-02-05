'''
TEMPORARY RUNNER.PY
'''


import csv
import time
import argparse
import os
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

# Import your modules
from symbolic_discovery.bacon3 import BACON3
from symbolic_discovery.bacon7 import BACON7
from symbolic_discovery.datasets import DatasetGenerator, CATALOGUE, DatasetConfig

# --- 1. Standardised Result Interface ---
@dataclass
class SolverResult:
    equation: str
    train_r2: float
    mse: float
    time_sec: float
    status: str

# --- 2. Solver Strategies ---
class BaseSolver(ABC):
    @abstractmethod
    def solve(self, train_df: pd.DataFrame, target_col: str, seed: int) -> SolverResult:
        pass

class Bacon3Wrapper(BaseSolver):
    def __init__(self, noise_level: float = 0.0, **kwargs):
        # Dynamic strictness based on noise
        self.r2_threshold = 0.98 if noise_level == 0.0 else 0.90
        self.max_depth = kwargs.get('max_depth', 3)
        self.verbose = kwargs.get('verbose', False)

    def solve(self, train_df: pd.DataFrame, target_col: str, seed: int) -> SolverResult:
        start_time = time.time()
        # Instantiate solver for this specific run
        model = BACON3(r2_threshold=self.r2_threshold, max_depth=self.max_depth, verbose=self.verbose)
        
        try:
            # BACON3 returns (equation, diagnostics)
            eq, diagnostics = model.discover(train_df, target_col, seed=seed)
            duration = time.time() - start_time
            
            # Check for failure
            if not eq or "No law found" in eq or "Failed" in eq:
                return SolverResult(eq or "No law found", 0.0, float('inf'), duration, "Failed")
                
            return SolverResult(
                equation=eq,
                train_r2=diagnostics.get("R-squared", 0.0),
                mse=diagnostics.get("MSE", 0.0),
                time_sec=duration,
                status="Success"
            )
        except Exception as e:
            return SolverResult(f"Error: {e}", 0.0, float('inf'), time.time() - start_time, "Error")

# Placeholder for future implementation
class Bacon7Wrapper(BaseSolver):
    def __init__(self, noise_level: float = 0.0, **kwargs):
        # BACON7 has its own parameter relaxation system
        self.initial_epsilon = 0.05 if noise_level == 0.0 else 0.10
        self.initial_delta = 0.05 if noise_level == 0.0 else 0.10
        self.r2_threshold = 0.98 if noise_level == 0.0 else 0.90
        self.max_depth = kwargs.get('max_depth', 4)
        self.verbose = kwargs.get('verbose', False)

    def solve(self, train_df: pd.DataFrame, target_col: str, seed: int) -> SolverResult:
        start_time = time.time()
        # Instantiate solver for this specific run
        model = BACON7(
            max_depth=self.max_depth,
            initial_epsilon=self.initial_epsilon,
            initial_delta=self.initial_delta,
            r2_threshold=self.r2_threshold,
            verbose=self.verbose
        )
        
        try:
            # BACON7 returns (equation, diagnostics)
            eq, diagnostics = model.discover(train_df, target_col, seed=seed)
            duration = time.time() - start_time
            
            # Check for failure
            if not eq or "No law found" in eq:
                return SolverResult(eq or "No law found", 0.0, float('inf'), duration, "Failed")
                
            return SolverResult(
                equation=eq,
                train_r2=diagnostics.get("R-squared", 0.0),
                mse=diagnostics.get("MSE", 0.0),
                time_sec=duration,
                status="Success"
            )
        except Exception as e:
            return SolverResult(f"Error: {e}", 0.0, float('inf'), time.time() - start_time, "Error")

# --- 3. Registry ---
SOLVER_REGISTRY = {
    "bacon3": Bacon3Wrapper,
    "bacon7": Bacon7Wrapper, 
    # "pysr": PySRWrapper,    # Uncomment when installed
    # "minisr": MiniSRWrapper # Uncomment when built
}

# --- 4. Experiment Runner Logic ---
def get_data_config(ds_arg: str, target_arg: Optional[str] = None) -> DatasetConfig:
    """
    Resolves whether the argument is a Catalogue ID or a Custom CSV.
    """
    # Case A: It's a generated dataset (S-1, T-1, etc.)
    if ds_arg in CATALOGUE:
        return CATALOGUE[ds_arg]
    
    # Case B: It's a custom CSV file
    if os.path.exists(ds_arg) and ds_arg.endswith('.csv'):
        if not target_arg:
            raise ValueError(f"You must provide --target when using a custom file: {ds_arg}")
        
        # Create a dummy config for custom data
        return DatasetConfig(
            id=os.path.basename(ds_arg),
            variables=[], # Will be inferred later
            target=target_arg,
            formula="Unknown", # No ground truth for custom data
            domain={}
        )
    
    raise ValueError(f"Dataset argument '{ds_arg}' is neither a valid Catalogue ID nor a CSV file.")

def run_experiment(args):
    # Setup Output
    fieldnames = ["run_id", "dataset", "method", "noise", "seed", "found_eq", "r2", "time_s", "status"]

    def _fmt_noise(n: float) -> str:
        # Stable, human-friendly filename component for common noise values.
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
            # Load Data Configuration
            try:
                config = get_data_config(ds_arg, args.target)
            except ValueError as e:
                print(f"Skipping {ds_arg}: {e}")
                continue

            for noise in args.noise:
                for model_name in args.models:
                    if model_name not in SOLVER_REGISTRY:
                        print(f"Warning: Model '{model_name}' not found.")
                        continue
                    
                    # Initialize Solver Strategy
                    # We pass noise so the wrapper can adjust thresholds (e.g. BACON strictness)
                    solver_class = SOLVER_REGISTRY[model_name]
                    solver = solver_class(noise_level=noise, verbose=args.verbose)

                    for seed in args.seeds:
                        run_id = f"{model_name}_{config.id}_N{noise}_S{seed}"
                        
                        # 1. Prepare Data
                        if ds_arg in CATALOGUE:
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
                        
                        # 3. Log
                        row = {
                            "run_id": run_id,
                            "dataset": config.id,
                            "method": model_name,
                            "noise": noise,
                            "seed": seed,
                            "found_eq": result.equation,
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
                        eq_preview = (result.equation or "").replace("\n", " ").strip()
                        if len(eq_preview) > 160:
                            eq_preview = eq_preview[:157] + "..."
                        print(f"[{model_name}] {config.id} (N={noise}, S={seed}) -> {result.status} (R2: {result.train_r2:.4f})")
                        if eq_preview:
                            print(f"    Eq: {eq_preview}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Symbolic Regression Models (BACON/PySR)")
    
    parser.add_argument("--models", nargs="+", required=True, 
                        choices=SOLVER_REGISTRY.keys(),
                        help="List of models to run (e.g. bacon3 bacon7)")
    
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="List of Catalogue IDs (S-1, T-1) OR paths to .csv files")
    
    parser.add_argument("--noise", nargs="+", type=float, default=[0.0],
                        help="Noise levels to inject (e.g. 0.0 0.01 0.05)")
    
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds for reproducibility")
    
    parser.add_argument("--target", type=str, default=None,
                        help="Target column name (REQUIRED if using custom .csv files)")
    
    parser.add_argument("--output", type=str, default="experiment_results.csv",
                        help="Output CSV file path")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="If set, write a combined CSV to <dir>/combined.csv and also write one CSV per run_id under <dir>/runs/")

    parser.add_argument("--shard-by", type=str, default="none",
                        choices=["none", "noise", "seed", "noise_seed"],
                        help="When using --output-dir, also write sharded CSVs under <dir>/shards/ (e.g. one per noise+seed)")

    parser.add_argument("--per-run-csv", action="store_true",
                        help="When using --output-dir, also write one CSV per run_id under <dir>/runs/")

    parser.add_argument("--overwrite-runs", action="store_true",
                        help="When using --per-run-csv, overwrite existing per-run CSVs if present")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output from BACON algorithms")

    args = parser.parse_args()
    run_experiment(args)