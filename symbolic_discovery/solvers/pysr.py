from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import pandas as pd

from symbolic_discovery.utils import calculate_mse, calculate_r2, calculate_mae
from .base import BaseSolver, SolverResult


def _extract_equation(model: Any) -> str:
    """Pull the best symbolic equation string from a fitted PySRRegressor."""
    if hasattr(model, "get_best"):
        best = model.get_best()
        if isinstance(best, pd.Series):
            for key in ("sympy_format", "equation"):
                if key in best:
                    return str(best[key])
            return str(best.to_dict())
        if isinstance(best, dict):
            return str(best.get("sympy_format") or best.get("equation") or best)
        return str(best)

    if hasattr(model, "sympy"):
        return str(model.sympy())

    if hasattr(model, "equations_"):
        eqs = model.equations_
        if hasattr(eqs, "iloc") and len(eqs) > 0:
            for col in ("equation", "sympy_format"):
                if col in getattr(eqs, "columns", []):
                    return str(eqs.iloc[0][col])
            return str(eqs.iloc[0])

    return ""


class PySRSolver(BaseSolver):
    """Wrapper for PySR, standardised to the BaseSolver interface."""

    def __init__(self, **kwargs: Any):
        # PySR uses single 'verbosity' flag so we map here.
        if "log_level" in kwargs:
            self.verbose = True if kwargs["log_level"] == "verbose" else False
            kwargs.pop("log_level")
        else:
            self.verbose = False

        self.kwargs = kwargs
        

    def solve(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
        target_col: str, seed: int) -> SolverResult:
        start = time.time()

        try:
            from pysr import PySRRegressor  # type: ignore
        except Exception as e:
            return SolverResult(
                equation="Error",
                raw_equation=f"PySR import failed: {e}",
                r2=float("nan"), 
                mse=float("nan"), 
                mae=float("nan"),
                time_sec=time.time() - start,
                status="Error",
                logs=[],
            )

        X = train_df.drop(columns=[target_col])
        y = train_df[target_col].to_numpy(copy=True)

        set_params: dict[str, Any] = {
            "binary_operators": ["+", "-", "*", "/"], # standardise operator set with BACON
            "verbosity": 1 if self.verbose else 0,
            "parallelism": "serial",
            "deterministic": True,
            "random_state": seed,
            "temp_equation_file": True,
            "delete_tempfiles": True,
        }

        for key in set_params:
            if key not in self.kwargs:
                self.kwargs[key] = set_params[key]

        model = PySRRegressor(**self.kwargs)

        try:
            model.fit(X, y)
        except Exception as e:
            return SolverResult(
                equation="Error",
                raw_equation=f"PySR fit failed: {e}",
                r2=float("nan"), 
                mse=float("nan"), 
                mae=float("nan"),
                time_sec=time.time() - start,
                status="Error",
                logs=[]
            )

        # Metrics calculation
        r2, mse, mae = float("nan"), float("nan"), float("nan")
        try:
            y_test = test_df[target_col].to_numpy(copy=True)
            x_test = test_df.drop(columns=[target_col])
            y_pred = model.predict(x_test)
            r2 = calculate_r2(y_test, y_pred)
            mse = calculate_mse(y_test, y_pred)
            mae = calculate_mae(y_test, y_pred)
        except Exception as e:
            raw_eq = f"PySR fitted but metrics calculation failed: {e}"
            return SolverResult(
                equation="Error",
                raw_equation=raw_eq,
                r2=float("nan"), 
                mse=float("nan"), 
                mae=float("nan"),
                time_sec=time.time() - start,
                status="Error",
                logs=[],
            )

        # Equation extraction
        try:
            raw_eq = _extract_equation(model)
        except Exception as e:
            raw_eq = f"PySR fitted but equation extraction failed: {e}"
            return SolverResult(
                equation="Error",
                raw_equation=raw_eq,
                r2=float("nan"), 
                mse=float("nan"), 
                mae=float("nan"),
                time_sec=time.time() - start,
                status="Error",
                logs=[],
            )

        duration = time.time() - start

        if not raw_eq:
            return SolverResult(
                equation="No law found",
                raw_equation="No law found",
                r2=float("nan"), 
                mse=float("nan"), 
                mae=float("nan"),
                time_sec=duration,
                status="Failure",
                logs=[],
            )

        equation = f"{target_col} = {raw_eq}" if "=" not in raw_eq else raw_eq

        return SolverResult(
            equation=equation,
            raw_equation=raw_eq,
            r2=r2, mse=mse, mae=mae,
            time_sec=duration,
            status="Found",
            logs=[], # For now, empty.
        )