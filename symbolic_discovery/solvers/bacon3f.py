from __future__ import annotations
import time
from typing import Any
import pandas as pd
import sympy
import numpy as np
from symbolic_discovery.algorithms import BACON3F
from .base import BaseSolver, SolverResult
from symbolic_discovery.utils import calculate_mse, calculate_r2, calculate_mae


class BACON3FSolver(BaseSolver):
    """
    Wrapper that adapts the BACON.3F core algorithm to the BaseSolver interface.
    """
    def __init__(self, **kwargs: Any):
        self.max_depth = kwargs.get("max_depth", 6)
        self.constancy_threshold = kwargs.get("initial_delta", 0.1)
        self.r2_threshold = kwargs.get("r2_threshold", 0.9)
        self.verbose = kwargs.get("verbose", False)


    def solve(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, seed: int) -> SolverResult:
        """
        Run BACON.3F and return the best discovered law as a SolverResult.
        """
        start_time = time.time()
        model = BACON3F(max_depth=self.max_depth, constancy_threshold=self.constancy_threshold, 
                        r2_threshold=self.r2_threshold, verbose=self.verbose)
        
        if target_col not in train_df.columns:
            return SolverResult(
                equation="Error",
                raw_equation=f"Target column '{target_col}' not found in training dataframe",
                r2=0.0,
                mse=float("inf"),
                mae=float("inf"),
                time_sec=time.time() - start_time,
                status="Error",
            )

        try:
            eq, _ = model.discover(train_df, target_col, seed=seed)
            duration = time.time() - start_time

            if eq == "No law found":
                return SolverResult(
                    equation="No law found",
                    raw_equation="No law found",
                    r2=0.0,
                    mse=float("inf"),
                    mae=float("inf"),
                    time_sec=duration,
                    status="Failure",
                )
            
            eq_clean = eq  # TBD: For now, just return the raw equation

            # Evaluate on test set if possible
            r2, mse, mae = 0.0, float("inf"), float("inf")
            try:
                rhs = eq_clean.split("=", 1)[-1].strip()
                local_syms = {col: sympy.Symbol(col) for col in test_df.columns if col != target_col}
                expr = sympy.sympify(rhs, locals=local_syms, evaluate=False)
                sym_map = {s: test_df[str(s)].to_numpy() for s in expr.free_symbols}
                y_pred = sympy.lambdify(list(sym_map.keys()), expr, modules=["numpy"])(*sym_map.values())
                y_test = test_df[target_col].to_numpy()
                r2 = calculate_r2(y_test, y_pred)
                mse = calculate_mse(y_test, y_pred)
                mae = calculate_mae(y_test, y_pred)
            except Exception:
                # TBD?
                pass

            return SolverResult(
                equation=eq_clean,
                raw_equation=eq,
                r2=r2,
                mse=mse,
                mae=mae,
                time_sec=duration,
                status="Found",
            )
        
        except Exception as e:
            return SolverResult(
                equation="Error",
                raw_equation=f"Error: {e}",
                r2=0.0,
                mse=float("inf"),
                mae=float("inf"),
                time_sec=time.time() - start_time,
                status="Error",
            )
