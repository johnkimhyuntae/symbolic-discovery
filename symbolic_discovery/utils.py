"""
Shared utility functions for BACON algorithms.
"""

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
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    if ss_total == 0:
        return 1.0 if ss_residual == 0 else 0.0
    
    r2 = 1 - (ss_residual / ss_total)
    return r2


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MSE value
    """
    return np.mean((y_true - y_pred) ** 2)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def fit_linear_model(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, dict]:
    """
    Fit a linear model y = ax + b and return coefficients and diagnostics.
    
    Args:
        x: Independent variable values
        y: Dependent variable values
    
    Returns:
        Tuple of (a, b, diagnostics_dict) where diagnostics contains R², MSE, MAE
    """
    try:
        # Fit line: y = ax + b
        coeffs = np.polyfit(x, y, 1)
        a, b = coeffs
        
        # Predictions
        y_pred = (a * x) + b
        
        # Calculate diagnostics
        r2 = calculate_r2(y, y_pred)
        mse = calculate_mse(y, y_pred)
        mae = calculate_mae(y, y_pred)
        
        diagnostics = {
            "R-squared": r2,
            "MSE": mse,
            "MAE": mae
        }
        
        return a, b, diagnostics
        
    except (np.linalg.LinAlgError, ValueError):
        # Fit failed
        return 0.0, 0.0, {
            "R-squared": -np.inf,
            "MSE": np.inf,
            "MAE": np.inf
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
        
        mean_lhs = np.mean(lhs_values)
        std_lhs = np.std(lhs_values)
        
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
        
        mse = np.mean((lhs_values - const_val) ** 2)
        
        return r2, mse
        
    except Exception:
        return 1.0, 0.0
