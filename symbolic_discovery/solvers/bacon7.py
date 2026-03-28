from __future__ import annotations

import time
from typing import Any

import pandas as pd

from symbolic_discovery._core.bacon7 import BACON7

from .base import BaseSolver, SolverResult


class Bacon7Wrapper(BaseSolver):
    def __init__(self, noise_level: float = 0.0, **kwargs: Any):
        # BACON7 has its own parameter relaxation system
        self.initial_epsilon = 0.05 if noise_level == 0.0 else 0.10
        self.initial_delta = 0.05 if noise_level == 0.0 else 0.10
        self.r2_threshold = 0.98 if noise_level == 0.0 else 0.90
        self.max_depth = kwargs.get("max_depth", 4)
        self.verbose = kwargs.get("verbose", False)

    def solve(self, train_df: pd.DataFrame, target_col: str, seed: int) -> SolverResult:
        start_time = time.time()
        model = BACON7(
            max_depth=self.max_depth,
            initial_epsilon=self.initial_epsilon,
            initial_delta=self.initial_delta,
            r2_threshold=self.r2_threshold,
            verbose=self.verbose,
        )

        try:
            eq, diagnostics = model.discover(train_df, target_col, seed=seed)
            duration = time.time() - start_time

            raw_eq = eq or ""
            if (not raw_eq) or ("No law found" in raw_eq):
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
