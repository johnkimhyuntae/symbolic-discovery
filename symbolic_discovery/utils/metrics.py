"""Shared utility functions for solvers."""
import numpy as np
import pandas as pd
import sympy
from typing import Tuple


def calculate_r(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient r between two arrays.
    
    Args:
        X: Independent variable values
        Y: Dependent variable values
    
    Returns:
        r value between -1 and 1
    """
    if len(X) != len(Y):
        raise ValueError("Input arrays must have the same length for correlation calculation.")
    
    X_mean = float(np.mean(X))
    Y_mean = float(np.mean(Y))
    
    numerator = np.sum((X - X_mean) * (Y - Y_mean))
    denominator = np.sqrt(np.sum((X - X_mean) ** 2) * np.sum((Y - Y_mean) ** 2))
    
    if denominator == 0:
        return 0.0
    
    r = numerator / denominator
    return float(r)


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
    
    if ss_total < 1e-9:
        return 1.0 if ss_residual < 1e-9 else 0.0
    
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


# For SR models that return an equation string, we can calculate metrics by evaluating the equation on the test set.
def equation_to_metrics(eq_str: str, test_df: pd.DataFrame, target_col: str) -> Tuple[float, float, float]:
    """
    Given an equation string and test data, calculate R², MSE, and MAE.
    
     Args:
        eq_str: Equation string in the form "y = expression"
        test_df: DataFrame containing test data
        target_col: Name of the target column in test_df
    """
    if target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in test dataframe.")
    
    if eq_str == "No law found":
        return 0.0, float("inf"), float("inf")

    rhs = eq_str.split("=", 1)[-1].strip()
    local_syms = {col: sympy.Symbol(col) for col in test_df.columns if col != target_col}
    expr = sympy.sympify(rhs, locals=local_syms, evaluate=False)
    sym_map = {s: test_df[str(s)].to_numpy() for s in expr.free_symbols}

    y_pred = sympy.lambdify(list(sym_map.keys()), expr, modules=["numpy"])(*sym_map.values())
    y_test = test_df[target_col].to_numpy()

    r2 = calculate_r2(y_test, y_pred)
    mse = calculate_mse(y_test, y_pred)
    mae = calculate_mae(y_test, y_pred)

    return r2, mse, mae
