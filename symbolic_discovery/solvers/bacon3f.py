from __future__ import annotations
import time
from typing import Any
import pandas as pd
from symbolic_discovery.algorithms import BACON3F
from .base import BaseSolver, SolverResult
from symbolic_discovery.utils import equation_to_metrics


class BACON3FSolver(BaseSolver):
    """
    Wrapper that adapts the BACON.3F core algorithm to the BaseSolver interface.
    """
    def __init__(self, **kwargs: Any):
        self._kwargs = kwargs


    def solve(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
              target_col: str, seed: int) -> SolverResult:
        """
        Run BACON.3F and return the best discovered law as a SolverResult.
        """
        start_time = time.time()
        model = BACON3F(**self._kwargs)
        
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
            
            eq_clean = eq  # TODO: For now, just return the raw equation

            # Evaluate on test set if possible
            r2, mse, mae = 0.0, float("inf"), float("inf")
            try:
                r2, mse, mae = equation_to_metrics(eq_clean, test_df, target_col)
            except Exception:
                # TODO?
                pass

            if r2 < 0.0:
                return SolverResult(
                    equation=eq_clean,
                    raw_equation=eq,
                    r2=r2,
                    mse=mse,
                    mae=mae,
                    time_sec=duration,
                    status="Failure",
                )
            
            else:
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
