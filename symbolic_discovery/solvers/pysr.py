from __future__ import annotations
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import re
from build.lib.symbolic_discovery.utils.metrics import calculate_mae
from symbolic_discovery.utils import calculate_mse, calculate_r2
from .base import BaseSolver, SolverResult


class PySRSolver(BaseSolver):
    """
    Wrapper for the PySR symbolic regression library
    TBD: MORE
    """
    def __init__(self, noise_level: float = 0.0, **kwargs: Any):
        self.noise_level = noise_level
        self.verbose = kwargs.get("verbose", False)
        self.engine_params: dict[str, Any] = dict(kwargs)
        self.engine_params.pop("verbose", None)


    def solve(self, train_df: pd.DataFrame, target_col: str, seed: int) -> SolverResult:
        start_time = time.time()

        try:
            from pysr import PySRRegressor  # type: ignore
        except Exception as e:
            return SolverResult(
                equation="Error",
                raw_equation=(
                    "PySR is not available in this environment. "
                    "Install it (and ensure Julia works), then rerun with --models pysr. "
                    f"Import error: {e}"
                ),
                r2=0.0,
                mse=float("inf"),
                mae=float("inf"),
                time_sec=time.time() - start_time,
                status="Error",
            )

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

        X = train_df.drop(columns=[target_col])
        y = np.asarray(train_df[target_col].to_numpy(copy=True))

        if X.shape[1] == 0:
            return SolverResult(
                equation="Error",
                raw_equation="No feature columns found after removing target",
                r2=0.0,
                mse=float("inf"),
                mae=float("inf"),
                time_sec=time.time() - start_time,
                status="Error",
            )

        # Some column names collide with SymPy built-ins (e.g. I = imaginary unit, E = Euler's number).
        # PySR uses SymPy-compatible variable naming, so we sanitise names before fitting.
        reserved = {"I", "E", "pi"}
        rename_map: dict[str, str] = {}
        used: set[str] = set()

        def _is_safe(name: str) -> bool:
            if name in reserved:
                return False
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                return False
            return True

        for idx, col in enumerate(list(X.columns)):
            if _is_safe(col) and col not in used:
                used.add(col)
                continue

            base = f"x{idx+1}"
            candidate = base
            j = 1
            while candidate in used or candidate in reserved:
                j += 1
                candidate = f"{base}_{j}"
            rename_map[col] = candidate
            used.add(candidate)

        if rename_map:
            X = X.rename(columns=rename_map)

        base_params: dict[str, Any] = {
            "niterations": int(self.engine_params.get("niterations", 40)),
        }
        # Align default operator set with the BACON solvers for fair comparisons.
        if "binary_operators" in self.engine_params:
            base_params["binary_operators"] = self.engine_params["binary_operators"]
        else:
            base_params["binary_operators"] = ["+", "-", "*", "/"]
        if "unary_operators" in self.engine_params:
            base_params["unary_operators"] = self.engine_params["unary_operators"]

        # Keep PySR side-effects (files + progress output) consistent with the BACON solvers by default.
        # - If the user did not request an output directory/run_id, keep outputs temporary and auto-cleaned.
        # - If the user did request outputs, use a readable timestamp run_id.
        if "progress" not in self.engine_params:
            base_params["progress"] = False
        if "verbosity" not in self.engine_params:
            base_params["verbosity"] = 0
        # PySR warns if random_state is set without deterministic+serial.
        # We always pass random_state (seed) below, so make searches deterministic by default.
        if "parallelism" not in self.engine_params:
            base_params["parallelism"] = "serial"
        if "deterministic" not in self.engine_params:
            base_params["deterministic"] = True

        user_output_directory = self.engine_params.get("output_directory")
        user_run_id = self.engine_params.get("run_id")

        def _timestamp_run_id() -> str:
            return datetime.now().strftime("%Y%m%d_%H%M%S")

        def _unique_run_id(parent: Path, base: str) -> str:
            candidate = base
            i = 0
            while (parent / candidate).exists():
                i += 1
                candidate = f"{base}_{i}"
            return candidate

        if user_output_directory is None and user_run_id is None:
            # No explicit request to persist artifacts; write to a temp dir and delete.
            base_params.setdefault("temp_equation_file", True)
            base_params.setdefault("delete_tempfiles", True)
        else:
            # Persist artifacts, but make naming legible.
            base_params.setdefault("temp_equation_file", False)
            if user_output_directory is not None:
                base_params["output_directory"] = user_output_directory

            if user_run_id is not None:
                base_params["run_id"] = user_run_id
            else:
                # Determine where the run_id directory will live.
                parent = Path(str(user_output_directory) if user_output_directory is not None else "outputs")
                base = _timestamp_run_id()
                base_params["run_id"] = _unique_run_id(parent, base)

        candidate_params: list[dict[str, Any]] = [
            {**base_params, "random_state": seed},
            {**base_params, "seed": seed},
            {**base_params},
        ]

        model = None
        last_type_error: Exception | None = None
        for params in candidate_params:
            try:
                model = PySRRegressor(**params)
                break
            except TypeError as e:
                last_type_error = e
                continue

        if model is None:
            return SolverResult(
                equation="Error",
                raw_equation=f"Could not initialise PySRRegressor with provided params: {last_type_error}",
                r2=0.0,
                mse=float("inf"),
                mae=float("inf"),
                time_sec=time.time() - start_time,
                status="Error",
            )

        try:
            model.fit(X, y)
        except Exception as e:
            return SolverResult(
                equation="Error",
                raw_equation=f"PySR fit failed: {e}",
                r2=0.0,
                mse=float("inf"),
                mae=float("inf"),
                time_sec=time.time() - start_time,
                status="Error",
            )

        r2 = 0.0
        mse = float("inf")
        mae = float("inf")
        try:
            y_pred = np.asarray(model.predict(X))
            r2 = calculate_r2(y, y_pred)
            mse = calculate_mse(y, y_pred)
            mae = calculate_mae(y, y_pred)
        except Exception:
            pass

        raw_eq = ""
        try:
            if hasattr(model, "get_best"):
                best = model.get_best()
                # PySR commonly returns a pandas Series for a single row.
                if isinstance(best, pd.Series):
                    if "sympy_format" in best:
                        raw_eq = str(best["sympy_format"])
                    elif "equation" in best:
                        raw_eq = str(best["equation"])
                    else:
                        raw_eq = str(best.to_dict())
                elif isinstance(best, dict):
                    raw_eq = str(best.get("sympy_format") or best.get("equation") or best)
                else:
                    raw_eq = str(best)
            elif hasattr(model, "sympy"):
                raw_eq = str(model.sympy())
            elif hasattr(model, "equations_"):
                eqs = getattr(model, "equations_")
                if hasattr(eqs, "iloc") and len(eqs) > 0:
                    for col in ("equation", "sympy_format", "latex_format"):
                        if hasattr(eqs, "columns") and col in eqs.columns:
                            raw_eq = str(eqs.iloc[0][col])
                            break
                    if not raw_eq:
                        raw_eq = str(eqs.iloc[0])
        except Exception as e:
            raw_eq = f"PySR fitted but could not extract equation: {e}"

        duration = time.time() - start_time
        if not raw_eq:
            return SolverResult(
                equation="No law found",
                raw_equation="No law found",
                r2=r2,
                mse=mse,
                mae=mae,
                time_sec=duration,
                status="Failed",
            )

        # Map sanitised variable names back to the original dataset columns.
        if rename_map:
            inv_map = {v: k for k, v in rename_map.items()}
            keys = sorted(inv_map.keys(), key=len, reverse=True)
            for k in keys:
                raw_eq = re.sub(rf"\b{re.escape(k)}\b", inv_map[k], raw_eq)

        eq_to_save = raw_eq
        if "=" not in eq_to_save:
            eq_to_save = f"{target_col} = {raw_eq}"

        return SolverResult(
            equation=eq_to_save,
            raw_equation=raw_eq,
            r2=r2,
            mse=mse,
            mae=mae,
            time_sec=duration,
            status="Success",
        )
