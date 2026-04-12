import pytest
import numpy as np
import pandas as pd
import sympy
from symbolic_discovery.algorithms import BACON3F
from symbolic_discovery.data.catalogue import CATALOGUE
from symbolic_discovery.data.synthetic import DatasetGenerator


def is_failure_equation(equation: str) -> bool:
    """Helper function to identify failure cases in BACON.3F's output."""
    if not equation:
        return True
    return ("No law found" in equation) or ("Failed" in equation) or (equation.strip() == "Error")


@pytest.fixture
def generator():
    """Provides a fresh dataset generator for each test."""
    return DatasetGenerator(seed=42)


# Clean data: exact recovery at depth=3

@pytest.mark.parametrize("dataset_id, expected_term", [
    ("S-1", "x1 + x2"),
    ("S-2", "x1*x2"),
    ("S-3", "x1/(x2 + 1)"),
    ("T-1", "I*R"),
    ("T-2", "k*x"),
    ("T-3", "t**2"),
    ("T-4", "P*V/n"),
    pytest.param("T-5", "T**4", marks=pytest.mark.xfail(
        reason="T**4 requires max_depth>=5: layer 4 promotes T**4, layer 5 detects constancy"
    )),
])
def test_baseline_exactness(generator, dataset_id, expected_term):
    """
    BACON.3F on clean data: expect high R², low MSE/MAE,
    and equation proportional to the true law.
    """
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.0)

    solver = BACON3F(max_depth=3, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE[dataset_id].target)

    assert not is_failure_equation(equation)
    assert diagnostics["R-squared"] > 0.999
    assert diagnostics["MSE"] < 1e-6
    assert diagnostics["MAE"] < 1e-3

    _, rhs = equation.split("=", 1)
    discovered = sympy.sympify(rhs.strip())
    expected = sympy.sympify(expected_term.strip())
    ratio = sympy.simplify(discovered / expected)
    assert ratio.is_number is True, \
        f"'{rhs.strip()}' not proportional to '{expected_term}'"


# Deeper search

def test_t5_deeper_search(generator):
    """T-5 (Stefan-Boltzmann) needs 5 layers to build constant * T**4."""
    train_df, _, _ = generator.generate("T-5", noise_level=0.0)
    solver = BACON3F(max_depth=5, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE["T-5"].target)

    assert not is_failure_equation(equation)
    assert diagnostics["R-squared"] > 0.999

    _, rhs = equation.split("=", 1)
    discovered = sympy.sympify(rhs.strip())
    expected = sympy.sympify("T**4")
    ratio = sympy.simplify(discovered / expected)
    assert ratio.is_number is True


# Known structural failures

@pytest.mark.parametrize("dataset_id", [
    "S-4",   # x1² + x2²: sum of two basic squares
])
def test_expected_failures_clean(generator, dataset_id):
    """Known BACON.3F limitations: should return failure or very low R²."""
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.0)
    solver = BACON3F(max_depth=3, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE[dataset_id].target)

    is_failure = is_failure_equation(equation)
    is_poor_fit = diagnostics["R-squared"] < 0.5
    assert is_failure or is_poor_fit, \
        f"Expected failure for {dataset_id} but got R²={diagnostics['R-squared']:.4f}: {equation}"


# Determinism

def test_determinism(generator):
    """Same seed, same data should give identical results."""
    train_df, _, _ = generator.generate("S-2", noise_level=0.0)

    eq_a, diag_a = BACON3F(max_depth=3).discover(train_df, target_col="y", seed=42)
    eq_b, diag_b = BACON3F(max_depth=3).discover(train_df, target_col="y", seed=42)

    assert eq_a == eq_b
    assert diag_a["R-squared"] == pytest.approx(diag_b["R-squared"])
    assert diag_a["MSE"] == pytest.approx(diag_b["MSE"])
    assert diag_a["MAE"] == pytest.approx(diag_b["MAE"])


# Edge cases

def test_constant_target():
    """If target is already constant, BACON should find it trivially."""
    df = pd.DataFrame({
        "x": np.linspace(1, 10, 20),
        "y": np.full(20, 3.14),
    })
    _, diagnostics = BACON3F(max_depth=3).discover(df, target_col="y")
    assert diagnostics["MSE"] < 1e-10


def test_two_identical_columns():
    """Two identical columns: y = x. Should find y/x = 1 or y - x = 0."""
    vals = np.linspace(1, 10, 20)
    df = pd.DataFrame({"x": vals, "y": vals.copy()})
    equation, diagnostics = BACON3F(max_depth=3).discover(df, target_col="y")

    assert not is_failure_equation(equation)
    assert diagnostics["R-squared"] > 0.999


def test_single_column_returns_failure():
    """Only target column, no independents — should return failure, not crash."""
    df = pd.DataFrame({"y": np.linspace(1, 10, 20)})
    equation, _ = BACON3F(max_depth=3).discover(df, target_col="y")
    assert is_failure_equation(equation)
