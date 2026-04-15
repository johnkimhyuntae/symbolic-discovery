"""Integration tests for BACON.3F on datasets."""
import pytest
import numpy as np
import pandas as pd
import sympy
from symbolic_discovery.algorithms import BACON3F
from symbolic_discovery.data import CATALOGUE, load
from symbolic_discovery.utils import equation_to_metrics


# Clean data: exact recovery

@pytest.mark.parametrize("dataset_id, expected_expr", [
    ("S1", "x1 + x2"),
    ("S2", "x1*x2"),
    ("S3", "x1/(x2 + 1)"),
    ("T1", "I*R"),
    ("T2", "k*x"),
    ("T3", "0.5*9.81*t**2"),
    ("T4", "P*V/(n*8.314)"),
    ("T5", "5.67e-8*T**4"),
])
def test_baseline_exactness(dataset_id, expected_expr):
    """
    BACON.3F on clean data: expect high R², low MSE/MAE,
    and equation proportional to the true law.
    """
    config = CATALOGUE[dataset_id]
    train_df, test_df, _ = load(config, noise=0.0)

    solver = BACON3F(r2_threshold = 0.999,
            constancy_threshold = 0.01,
            verbose=True)
    equation, _ = solver.discover(train_df, target_col=config.target)

    assert not equation == "No law found"
    r2, mse, mae = equation_to_metrics(equation, test_df, config.target)
    assert r2 > 0.999
    assert mse < 0.001
    assert mae < 0.001

    _, rhs = equation.split("=", 1)
    discovered = sympy.sympify(rhs.strip())
    expected = sympy.sympify(expected_expr.strip())
    ratio = sympy.simplify(discovered / expected)
    assert ratio.is_number is True
    assert abs(1 - ratio) < 0.01, \
        f"Expected {expected}, got {discovered}"


# Known structural failures

@pytest.mark.parametrize("dataset_id", [
    "S4",   # x1² + x2²: sum of two basic squares
])
def test_expected_failures_clean(dataset_id):
    """Known BACON.3F limitations: should return failure or very low R²."""
    config = CATALOGUE[dataset_id]
    train_df, test_df, _ = load(config, noise=0.0)
    solver = BACON3F(r2_threshold = 0.999,
            constancy_threshold = 0.01,
            verbose=True)
    equation, _ = solver.discover(train_df, target_col=config.target)

    is_failure = equation == "No law found"
    r2, _, _ = equation_to_metrics(equation, test_df, config.target)
    is_poor_fit = r2 < 0.5
    assert is_failure or is_poor_fit, \
        f"Expected failure for {dataset_id} but got R²={r2:.4f}: {equation}"


# Determinism

def test_determinism():
    """Same seed, same data should give identical results."""
    config = CATALOGUE["S2"]
    train_df, test_df, _ = load(config, noise=0.0)

    eq_a, _ = BACON3F(verbose=True).discover(train_df, target_col=config.target, seed=42)
    eq_b, _ = BACON3F(verbose=True).discover(train_df, target_col=config.target, seed=42)

    r2_a, mse_a, mae_a = equation_to_metrics(eq_a, test_df, config.target)
    r2_b, mse_b, mae_b = equation_to_metrics(eq_b, test_df, config.target)

    assert eq_a == eq_b
    assert r2_a == pytest.approx(r2_b)
    assert mse_a == pytest.approx(mse_b)
    assert mae_a == pytest.approx(mae_b)


# Edge cases (use hand-crafted DataFrames)

def test_constant_target():
    """If target is already constant, BACON should find it trivially."""
    df = pd.DataFrame({
        "x": np.linspace(1, 10, 20),
        "y": np.full(20, 3.14),
    })
    equation, _ = BACON3F(verbose=True).discover(df, target_col="y")
    _, mse, _ = equation_to_metrics(equation, df, "y")
    assert mse < 1e-10


def test_two_identical_columns():
    """Two identical columns: y = x. Should find y/x = 1 or y - x = 0."""
    vals = np.linspace(1, 10, 20)
    df = pd.DataFrame({"x": vals, "y": vals.copy()})
    equation, _ = BACON3F(verbose=True).discover(df, target_col="y")

    r2, _, _ = equation_to_metrics(equation, df, "y")

    assert not equation == "No law found"
    assert r2 > 0.999


def test_single_column_returns_failure():
    """Only target column, no independents — should return failure, not crash."""
    df = pd.DataFrame({"y": np.linspace(1, 10, 20)})
    equation, _ = BACON3F(verbose=True).discover(df, target_col="y")
    assert equation == "No law found"
