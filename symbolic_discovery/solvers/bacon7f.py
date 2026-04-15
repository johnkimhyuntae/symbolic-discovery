from __future__ import annotations
import time
from typing import Any
import pandas as pd
from symbolic_discovery.algorithms import BACON7F
from .base import BaseSolver, SolverResult
from symbolic_discovery.utils import equation_to_metrics


class BACON7FSolver(BaseSolver):
    """
    Wrapper that adapts the BACON.7F core algorithm to the BaseSolver interface.
    """
    def __init__(self, **kwargs: Any):
        # TBD: tune params
        self.max_depth: int = kwargs.get("max_depth", 6)
        self.initial_epsilon: float = kwargs.get("initial_epsilon", 0.01)
        self.initial_delta: float = kwargs.get("initial_delta", 0.1)    
        self.c_val: float = kwargs.get("c_val", 0.05)
        self.scale_factor: float = kwargs.get("scale_factor", 1.2)
        self.big_delta: float = kwargs.get("big_delta", 0.1)
        self.n_folds: int = kwargs.get("n_folds", 5)
        self.r2_threshold: float = kwargs.get("r2_threshold", 0.9)
        self.verbose: bool = kwargs.get("verbose", False)


    def solve(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
              target_col: str, seed: int) -> SolverResult:
        """
        Run BACON.7F and return the best discovered law as a SolverResult.
        """
        start_time = time.time()
        model = BACON7F(max_depth=self.max_depth, initial_epsilon=self.initial_epsilon, 
                        initial_delta=self.initial_delta, c_val=self.c_val, 
                        scale_factor=self.scale_factor, big_delta=self.big_delta, 
                        n_folds=self.n_folds, r2_threshold=self.r2_threshold, 
                        verbose=self.verbose)

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
                r2, mse, mae = equation_to_metrics(eq_clean, test_df, target_col)
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
