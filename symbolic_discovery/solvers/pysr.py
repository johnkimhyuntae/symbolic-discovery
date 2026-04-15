from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import pandas as pd

from symbolic_discovery.utils import calculate_mse, calculate_r2, calculate_mae
from .base import BaseSolver, SolverResult


# SymPy reserved names that PySR will misinterpret as built-in constants.
_SYMPY_RESERVED = {"I", "E", "pi"}


def _sanitise_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename columns that collide with SymPy built-ins.

    Returns ``(X_renamed, rename_map)`` where *rename_map* is only
    populated for columns that were actually changed.
    """
    rename_map: dict[str, str] = {}
    used: set[str] = set()

    for idx, col in enumerate(X.columns):
        safe = (
            col not in _SYMPY_RESERVED
            and re.match(r"^[A-Za-z_]\w*$", col) is not None
            and col not in used
        )
        if safe:
            used.add(col)
            continue

        candidate = f"x{idx + 1}"
        j = 1
        while candidate in used or candidate in _SYMPY_RESERVED:
            j += 1
            candidate = f"x{idx + 1}_{j}"
        rename_map[col] = candidate
        used.add(candidate)

    if rename_map:
        X = X.rename(columns=rename_map)
    return X, rename_map


def _unsanitise_equation(eq: str, rename_map: dict[str, str]) -> str:
    """Replace sanitised variable names back to originals in *eq*."""
    if not rename_map:
        return eq
    inv = {v: k for k, v in rename_map.items()}
    for k in sorted(inv, key=len, reverse=True):
        eq = re.sub(rf"\b{re.escape(k)}\b", inv[k], eq)
    return eq


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
        self.verbose = kwargs.get("verbose", False)

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
                r2=0.0, mse=float("inf"), mae=float("inf"),
                time_sec=time.time() - start,
                status="Error",
            )

        X = train_df.drop(columns=[target_col])
        y = train_df[target_col].to_numpy(copy=True)
        X, rename_map = _sanitise_columns(X)

        params: dict[str, Any] = {
            # "niterations": 40,
            "binary_operators": ["+", "-", "*", "/"], # standardise operator set with BACON
            "verbosity": 1 if self.verbose else 0,
            "parallelism": "serial",
            "deterministic": True,
            "random_state": seed,
            "temp_equation_file": True,
            "delete_tempfiles": True,
            "maxdepth": 6, # TBD: let me make same to BACONs
        }

        model = PySRRegressor(**params)

        try:
            model.fit(X, y)
        except Exception as e:
            return SolverResult(
                equation="Error",
                raw_equation=f"PySR fit failed: {e}",
                r2=0.0, mse=float("inf"), mae=float("inf"),
                time_sec=time.time() - start,
                status="Error",
            )

        # Metrics TBD TBD
        r2, mse, mae = 0.0, float("inf"), float("inf")
        try:
            y_test = test_df[target_col].to_numpy(copy=True)
            x_test = test_df.drop(columns=[target_col])
            x_test, _ = _sanitise_columns(x_test)
            y_pred = model.predict(x_test)
            r2 = calculate_r2(y_test, y_pred)
            mse = calculate_mse(y_test, y_pred)
            mae = calculate_mae(y_test, y_pred)
        except Exception:
            pass

        # Equation extraction
        try:
            raw_eq = _extract_equation(model)
        except Exception as e:
            raw_eq = f"PySR fitted but equation extraction failed: {e}"
            return SolverResult(
                equation="Error",
                raw_equation=raw_eq,
                r2=r2, mse=mse, mae=mae,
                time_sec=time.time() - start,
                status="Error",
            )

        duration = time.time() - start

        if not raw_eq:
            return SolverResult(
                equation="No law found",
                raw_equation="No law found",
                r2=r2, mse=mse, mae=mae,
                time_sec=duration,
                status="Failure",
            )

        raw_eq = _unsanitise_equation(raw_eq, rename_map)
        equation = f"{target_col} = {raw_eq}" if "=" not in raw_eq else raw_eq

        return SolverResult(
            equation=equation,
            raw_equation=raw_eq,
            r2=r2, mse=mse, mae=mae,
            time_sec=duration,
            status="Found",
        )