"""Shared utility functions for BACON algorithms."""
import numpy as np
import pandas as pd
from typing import Tuple


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        R² value between 0 and 1 (can be negative for very poor fits)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_mean = float(np.mean(y_true))
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    if ss_total == 0:
        return 1.0 if ss_residual == 0 else 0.0
    
    r2 = 1 - (ss_residual / ss_total)
    return float(r2)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MSE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def fit_linear_model(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, dict]:
    """
    Fit a linear model y = ax + b and return coefficients and diagnostics.
    
    Args:
        x: Independent variable values
        y: Dependent variable values
    
    Returns:
        Tuple of (a, b, diagnostics_dict) where diagnostics contains R², MSE, MAE
    """
    # This function is called extremely frequently during BACON search.
    # Using np.polyfit can emit noisy warnings (and even LAPACK stderr output)
    # when inputs are degenerate (e.g., constant x, NaN/Inf). A closed-form
    # OLS fit avoids those issues and is equivalent for degree-1 regression.
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()

    if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
        return 0.0, 0.0, {"R-squared": -np.inf, "MSE": np.inf, "MAE": np.inf}

    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]

    if x_arr.size < 2:
        b = float(np.mean(y_arr)) if y_arr.size else 0.0
        y_pred = np.full_like(y_arr, b, dtype=float)
        return 0.0, b, {
            "R-squared": calculate_r2(y_arr, y_pred) if y_arr.size else 0.0,
            "MSE": calculate_mse(y_arr, y_pred) if y_arr.size else np.inf,
            "MAE": calculate_mae(y_arr, y_pred) if y_arr.size else np.inf,
        }

    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    x_centered = x_arr - x_mean
    y_centered = y_arr - y_mean

    denom = float(np.dot(x_centered, x_centered))
    if abs(denom) < 1e-18:
        # x is effectively constant; best linear predictor is a constant.
        a = 0.0
        b = y_mean
    else:
        a = float(np.dot(x_centered, y_centered) / denom)
        b = float(y_mean - a * x_mean)

    y_pred = (a * x_arr) + b
    return a, b, {
        "R-squared": calculate_r2(y_arr, y_pred),
        "MSE": calculate_mse(y_arr, y_pred),
        "MAE": calculate_mae(y_arr, y_pred),
    }


def evaluate_equation_constancy(
    equation_str: str,
    const_val: float,
    eval_df: pd.DataFrame,
    target_name: str
) -> Tuple[float, float]:
    """
    Evaluate how well an equation of form "LHS = constant" holds.
    
    Args:
        equation_str: Equation string like "P/(T*T³) = 5.67e-08"
        const_val: Expected constant value
        eval_df: DataFrame with all variables for evaluation
        target_name: Name of target variable
    
    Returns:
        Tuple of (r2, mse) measuring how constant the LHS actually is
    """
    try:
        if " = " not in equation_str:
            return 1.0, 0.0
        
        lhs_str, rhs_str = equation_str.split(" = ", 1)
        # Use the passed const_val parameter (exact value) not the formatted string
        # which may have precision loss from .4g formatting
        
        # Replace unicode superscripts for pandas eval
        lhs_eval = lhs_str.replace('²', '**2').replace('³', '**3')
        
        # Evaluate the LHS expression
        lhs_values = eval_df.eval(lhs_eval)
        lhs_arr = np.asarray(lhs_values, dtype=float)
        
        # For "LHS = constant" equations, we want to measure how constant LHS actually is
        # Problem: Standard R² doesn't work when const_val = mean(LHS)
        # because ss_res = ss_tot, giving R² = 0
        #
        # Solution: Measure "constancy" as 1 - (coefficient of variation)²
        # CV = std/|mean| measures relative variation
        # CV close to 0 → high R²
        # CV = 0.01 (1% variation) → R² = 0.9999
        # CV = 0.05 (5% variation) → R² = 0.9975
        # CV = 0.10 (10% variation) → R² = 0.99
        
        mean_lhs = float(np.mean(lhs_arr))
        std_lhs = float(np.std(lhs_arr))
        
        if abs(mean_lhs) < 1e-10:
            # LHS is ~0, check if std is also ~0 (perfectly zero)
            r2 = 1.0 if std_lhs < 1e-10 else 0.0
        else:
            cv = std_lhs / abs(mean_lhs)
            # R² = 1 - CV² ensures:
            # - CV << 1 → R² ≈ 1 (highly constant)
            # - CV ~ 1 → R² ≈ 0 (not constant)
            # - CV > 1 → R² negative (very poor), clamp to 0
            r2 = max(0.0, 1 - cv**2)
        
        mse = float(np.mean((lhs_arr - float(const_val)) ** 2))
        
        return float(r2), mse
        
    except Exception:
        # If we can't evaluate the expression, treat as non-constant.
        return 0.0, float('inf')
