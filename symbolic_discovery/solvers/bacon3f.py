from __future__ import annotations
import time
from typing import Any
import pandas as pd
from symbolic_discovery.algorithms import BACON3F
from .base import BaseSolver, SolverResult


class Bacon3FSolver(BaseSolver):
    """
    Wrapper that adapts the BACON.3F core algorithm to the BaseSolver interface.
    """
    def __init__(self, noise_level: float = 0.0, **kwargs: Any):
        # TBD: dynamic hyperparameteres based on noise
        self.r2_threshold = 0.990 if noise_level == 0.0 else 0.900
        self.max_depth = kwargs.get("max_depth", 3)
        self.verbose = kwargs.get("verbose", False)


    def solve(self, train_df: pd.DataFrame, target_col: str, seed: int) -> SolverResult:
        """
        Run BACON.3F and return the best discovered law as a SolverResult.
        """
        start_time = time.time()
        model = BACON3F(max_depth=self.max_depth, r2_threshold=self.r2_threshold, verbose=self.verbose)

        try:
            # Returns (equation_str, diagnostics) for the best discovered law, 
            # or ("No law found", {...}) on failure.
            results = model.discover(train_df, target_col, seed=seed)
            duration = time.time() - start_time

            if results[0] == "No law found":
                return SolverResult(
                    equation="No law found",
                    raw_equation="No law found",
                    r2=0.0,
                    mse=float("inf"),
                    mae=float("inf"),
                    time_sec=duration,
                    status="Failed",
                )
            
            eq, diagnostics = results
            eq_clean = eq  # TBD: For now, just return the raw equation

            return SolverResult(
                equation=eq_clean,
                raw_equation=eq,
                r2=diagnostics["R-squared"],
                mse=diagnostics["MSE"],
                mae=diagnostics["MAE"],
                time_sec=duration,
                status="Success",
            )
        
        except Exception as e:
            return SolverResult(
                equation=f"Error: {e}",
                raw_equation=f"Error: {e}",
                r2=0.0,
                mse=float("inf"),
                mae=float("inf"),
                time_sec=time.time() - start_time,
                status="Error",
            )
