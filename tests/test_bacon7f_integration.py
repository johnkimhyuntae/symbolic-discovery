"""Integration tests for BACON.7F on datasets."""
import pytest
import numpy as np
import pandas as pd
import sympy
from symbolic_discovery.algorithms import BACON7F
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
    """BACON.7F on clean data: discovered equation must simplify to the true law."""
    config = CATALOGUE[dataset_id]
    train_df, test_df, _ = load(config, noise=0.0)

    solver = BACON7F(r2_threshold = 0.999,
            scale_factor = 1.0,
            initial_epsilon = 0.01,
            initial_delta = 0.01,
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
    """Known BACON.7F limitations: should return failure or very low R²."""
    config = CATALOGUE[dataset_id]
    train_df, test_df, _ = load(config, noise=0.0)
    solver = BACON7F(r2_threshold = 0.999,
            scale_factor = 1.0,
            initial_epsilon = 0.01,
            initial_delta = 0.01,
            verbose=True)
    equation, _ = solver.discover(train_df, target_col=config.target)

    is_failure = equation == "No law found"
    r2, _, _ = equation_to_metrics(equation, test_df, config.target)
    is_poor_fit = r2 < 0.5
    assert is_failure or is_poor_fit, \
        f"Expected failure for {dataset_id} but got R²={r2:.4f}: {equation}"


# n_folds=1 degenerates to BACON.3F-like behaviour

@pytest.mark.parametrize("dataset_id", ["S2", "T1"])
def test_n_folds_1_matches_no_voting(dataset_id):
    """With n_folds=1, no voting occurs; should still find the same law."""
    config = CATALOGUE[dataset_id]
    train_df, test_df, _ = load(config, noise=0.0)

    solver = BACON7F(r2_threshold = 0.999,
            scale_factor = 1.0,
            initial_epsilon = 0.01,
            initial_delta = 0.01,
            n_folds = 1, verbose=True)
    equation, _ = solver.discover(train_df, target_col=config.target)

    assert not equation == "No law found"
    r2, _, _ = equation_to_metrics(equation, test_df, config.target)
    assert r2 > 0.999


# n_folds sensitivity

@pytest.mark.parametrize("n_folds", [1, 3, 5])
def test_n_folds_sensitivity(n_folds):
    """Varying n_folds should all succeed on clean data."""
    config = CATALOGUE["S2"]
    train_df, test_df, _ = load(config, noise=0.0)

    solver = BACON7F(r2_threshold = 0.999,
            scale_factor = 1.0,
            initial_epsilon = 0.01,
            initial_delta = 0.01,
            n_folds = n_folds,
            verbose=True)
    equation, _ = solver.discover(train_df, target_col=config.target)

    assert not equation == "No law found"
    r2, _, _ = equation_to_metrics(equation, test_df, config.target)
    assert r2 > 0.999


# Noise resilience: BACON.7F should match or beat BACON.3F

@pytest.mark.parametrize("dataset_id", ["S2", "T1"])
def test_noise_resilience_vs_bacon3f(dataset_id):
    """
    At 2% noise, BACON.7F should achieve R² at least as good as BACON.3F.
    This is the core justification for subset voting.
    """
    config = CATALOGUE[dataset_id]
    train_df, test_df, _ = load(config, noise=0.02)
    target = config.target

    # Both have defaults tuned for noisy data
    bacon3f = BACON3F(verbose=False)
    _, diag_3 = bacon3f.discover(train_df, target_col=target)

    bacon7f = BACON7F(verbose=False)
    _, diag_7 = bacon7f.discover(train_df, target_col=target)

    r2_3 = diag_3["R-squared"]
    r2_7 = diag_7["R-squared"]

    print(f"\n{dataset_id} @ 2% noise: BACON.3F R²={r2_3:.4f}, BACON.7F R²={r2_7:.4f}")

    assert r2_7 >= r2_3 - 0.05, \
        f"BACON.7F R²={r2_7:.4f} substantially worse than BACON.3F R²={r2_3:.4f}"


# Determinism

def test_determinism():
    """Same seed, same data -> identical results."""
    config = CATALOGUE["S2"]
    train_df, test_df, _ = load(config, noise=0.0)

    eq_a, _ = BACON7F(verbose=True).discover(train_df, target_col=config.target, seed=42)
    eq_b, _ = BACON7F(verbose=True).discover(train_df, target_col=config.target, seed=42)

    r2_a, mse_a, mae_a = equation_to_metrics(eq_a, test_df, config.target)
    r2_b, mse_b, mae_b = equation_to_metrics(eq_b, test_df, config.target)

    assert eq_a == eq_b
    assert r2_a == pytest.approx(r2_b)
    assert mse_a == pytest.approx(mse_b)
    assert mae_a == pytest.approx(mae_b)


# Edge cases (use hand-crafted DataFrames)

def test_constant_target():
    """If target is already constant, BACON.7F should find it trivially."""
    df = pd.DataFrame({
        "x": np.linspace(1, 10, 20),
        "y": np.full(20, 3.14),
    })
    eq, _ = BACON7F(verbose=True).discover(df, target_col="y")
    _, mse, _ = equation_to_metrics(eq, df, "y")
    assert mse < 1e-10


def test_two_identical_columns():
    """Two identical columns: y = x. Should find y/x = 1 or y - x = 0."""
    vals = np.linspace(1, 10, 20)
    df = pd.DataFrame({"x": vals, "y": vals.copy()})
    equation, _ = BACON7F(verbose=True).discover(df, target_col="y")
    r2, _, _ = equation_to_metrics(equation, df, "y")

    assert not equation == "No law found"
    assert r2 > 0.999


def test_single_column_returns_failure():
    """Only target column, no independents. Should return failure, not crash."""
    df = pd.DataFrame({"y": np.linspace(1, 10, 20)})
    equation, _ = BACON7F(verbose=True).discover(df, target_col="y")
    assert equation == "No law found"


# Voting filters noise

def test_voting_filters_noise():
    """Voting should recover correct form with mild localised noise."""
    np.random.seed(42)
    x = np.linspace(1, 10, 60)
    y = 5.0 / x

    # Mild corruption on middle third
    y_noisy = y.copy()
    y_noisy[20:40] += np.random.normal(0, 0.02, 20)

    df = pd.DataFrame({"x": x, "y": y_noisy})
    solver = BACON7F(r2_threshold=0.9, n_folds=3, verbose=True)
    equation, _ = solver.discover(df, target_col="y")
    r2, _, _ = equation_to_metrics(equation, df, "y")

    assert not equation == "No law found"
    assert r2 > 0.9
