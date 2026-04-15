"""Shared utility functions for solvers."""
import numpy as np


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
