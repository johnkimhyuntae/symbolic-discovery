from __future__ import annotations

import time
from typing import Any

import pandas as pd

from symbolic_discovery._core.bacon3 import BACON3

from .base import BaseSolver, SolverResult


class Bacon3Wrapper(BaseSolver):
    def __init__(self, noise_level: float = 0.0, **kwargs: Any):
        # Dynamic strictness based on noise
        self.r2_threshold = 0.98 if noise_level == 0.0 else 0.90
        self.max_depth = kwargs.get("max_depth", 3)
        self.verbose = kwargs.get("verbose", False)

    def solve(self, train_df: pd.DataFrame, target_col: str, seed: int) -> SolverResult:
        start_time = time.time()
        model = BACON3(r2_threshold=self.r2_threshold, max_depth=self.max_depth, verbose=self.verbose)

        try:
            eq, diagnostics = model.discover(train_df, target_col, seed=seed)
            duration = time.time() - start_time

            raw_eq = eq or ""
            if (not raw_eq) or ("No law found" in raw_eq) or ("Failed" in raw_eq):
                return SolverResult(
                    equation="No law found",
                    raw_equation=raw_eq or "No law found",
                    train_r2=0.0,
                    mse=float("inf"),
                    time_sec=duration,
                    status="Failed",
                )

            return SolverResult(
                equation=raw_eq,
                raw_equation=raw_eq,
                train_r2=float(diagnostics.get("R-squared", 0.0)),
                mse=float(diagnostics.get("MSE", 0.0)),
                time_sec=duration,
                status="Success",
            )
        except Exception as e:
            return SolverResult(
                equation="Error",
                raw_equation=f"Error: {e}",
                train_r2=0.0,
                mse=float("inf"),
                time_sec=time.time() - start_time,
                status="Error",
            )
