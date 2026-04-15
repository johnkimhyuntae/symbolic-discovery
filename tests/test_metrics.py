import pytest
import numpy as np
import pandas as pd
from symbolic_discovery.utils.metrics import (
    calculate_r,
    calculate_r2,
    calculate_mse,
    calculate_mae,
    equation_to_metrics,
)


# calculate_r

def test_calculate_r_perfect_positive():
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([2, 4, 6, 8, 10])
    assert calculate_r(X, Y) == pytest.approx(1.0)


def test_calculate_r_perfect_negative():
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([10, 8, 6, 4, 2])
    assert calculate_r(X, Y) == pytest.approx(-1.0)


def test_calculate_r_zero_variance():
    X = np.array([1, 1, 1, 1])
    Y = np.array([2, 4, 6, 8])
    assert calculate_r(X, Y) == 0.0


# calculate_r2

def test_calculate_r2_perfect():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    assert calculate_r2(y_true, y_pred) == pytest.approx(1.0)


def test_calculate_r2_poor():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1])
    assert calculate_r2(y_true, y_pred) < 0


def test_calculate_r2_constant_target():
    y_true = np.array([5, 5, 5, 5])
    y_pred = np.array([5, 5, 5, 5])
    assert calculate_r2(y_true, y_pred) == 1.0


# calculate_mse

def test_calculate_mse_perfect():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    assert calculate_mse(y_true, y_pred) == 0.0


def test_calculate_mse_known():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 3, 4])
    assert calculate_mse(y_true, y_pred) == pytest.approx(1.0)


# calculate_mae

def test_calculate_mae_perfect():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    assert calculate_mae(y_true, y_pred) == 0.0


def test_calculate_mae_known():
    y_true = np.array([0, 2, 3])
    y_pred = np.array([2, 4, 5])
    assert calculate_mae(y_true, y_pred) == pytest.approx(2.0)


# equation_to_metrics

def test_equation_to_metrics_simple():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    r2, mse, mae = equation_to_metrics("y = 2*x", df, "y")
    assert r2 == pytest.approx(1.0)
    assert mse == pytest.approx(0.0)
    assert mae == pytest.approx(0.0)


def test_equation_to_metrics_no_law():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    r2, mse, mae = equation_to_metrics("No law found", df, "y")
    assert r2 == 0.0
    assert mse == float("inf")
    assert mae == float("inf")


def test_equation_to_metrics_missing_target():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        equation_to_metrics("y = x", df, "y")
