import pytest
import numpy as np
import pandas as pd
import sympy
from symbolic_discovery.algorithms import BACON7F
from symbolic_discovery.algorithms import BACON3F
from symbolic_discovery.data.catalogue import CATALOGUE
from symbolic_discovery.data.synthetic import DatasetGenerator


def is_failure_equation(equation: str) -> bool:
    if not equation:
        return True
    return ("No law found" in equation) or ("Failed" in equation) or (equation.strip() == "Error")


@pytest.fixture
def generator():
    """Provides a fresh dataset generator for each test."""
    return DatasetGenerator(seed=42)


# Clean data: exact recovery at depth=3

@pytest.mark.parametrize("dataset_id, expected_expr", [
    ("S-1", "x1 + x2"),
    ("S-2", "x1*x2"),
    ("S-3", "x1/(x2 + 1)"),
    ("T-1", "I*R"),
    ("T-2", "k*x"),
    pytest.param("T-3", "0.5*9.81*t**2", marks=pytest.mark.xfail(
        reason="Correlation-based linearity check misclassifies t² as linear"
    )),
    pytest.param("T-4", "P*V/(n*8.314)", marks=pytest.mark.xfail(
        reason="Needs depth>=4"
    )),
    pytest.param("T-5", "5.67e-8*T**4", marks=pytest.mark.xfail(
        reason="Needs depth>=5"
    )),
])
# TBD: figure out why T-3 doesn't work.
def test_baseline_exactness(generator, dataset_id, expected_expr):
    """BACON.7F on clean data: discovered equation must simplify to the true law."""
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.0)

    solver = BACON7F(max_depth=3, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE[dataset_id].target)

    assert not is_failure_equation(equation)
    assert diagnostics["R-squared"] > 0.999

    _, rhs = equation.split("=", 1)
    discovered = sympy.simplify(sympy.sympify(rhs.strip()))
    expected = sympy.simplify(sympy.sympify(expected_expr.strip()))
    assert discovered.equals(expected), \
        f"Expected {expected}, got {discovered}"

# Deeper search for datasets that need more layers

def test_t4_deeper_search(generator):
    """T-4 (Ideal Gas) needs 4 layers: layer 3 creates PV/(n*8.314), layer 4 finds constancy."""
    train_df, _, _ = generator.generate("T-4", noise_level=0.0)
    solver = BACON7F(max_depth=4, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE["T-4"].target)

    assert not is_failure_equation(equation)
    assert diagnostics["R-squared"] > 0.999

    # Check structure is proportional to P*V/n
    _, rhs = equation.split("=", 1)
    discovered = sympy.sympify(rhs.strip())
    expected = sympy.sympify("P*V/n")
    ratio = sympy.simplify(discovered / expected)
    assert ratio.is_number is True


def test_t5_deeper_search(generator):
    """T-5 (Stefan-Boltzmann) needs 5 layers: layer 4 builds T**4, layer 5 finds constancy."""
    train_df, _, _ = generator.generate("T-5", noise_level=0.0)
    solver = BACON7F(max_depth=5, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE["T-5"].target)

    assert not is_failure_equation(equation)
    assert diagnostics["R-squared"] > 0.999

    # Check structure is proportional to 5.67e-8*T**4
    _, rhs = equation.split("=", 1)
    discovered = sympy.sympify(rhs.strip())
    expected = sympy.sympify("5.67e-8*T**4")
    ratio = sympy.simplify(discovered / expected)
    assert ratio.is_number is True


# Known structural failures

@pytest.mark.parametrize("dataset_id", [
    "S-4",   # x1² + x2²: sum of two basic squares
])
def test_expected_failures_clean(generator, dataset_id):
    """Known BACON.7F limitations: should return failure or very low R²."""
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.0)
    solver = BACON7F(max_depth=3, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE[dataset_id].target)

    is_failure = is_failure_equation(equation)
    is_poor_fit = diagnostics["R-squared"] < 0.5
    assert is_failure or is_poor_fit, \
        f"Expected failure for {dataset_id} but got R²={diagnostics['R-squared']:.4f}: {equation}"


# n_folds=1 degenerates to BACON.3F-like behaviour

@pytest.mark.parametrize("dataset_id", ["S-2", "T-1"])
def test_n_folds_1_matches_no_voting(generator, dataset_id):
    """With n_folds=1, no voting occurs — should still find the same law."""
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.0)

    solver = BACON7F(max_depth=3, n_folds=1, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col=CATALOGUE[dataset_id].target)

    assert not is_failure_equation(equation)
    assert diagnostics["R-squared"] > 0.999


# n_folds sensitivity

@pytest.mark.parametrize("n_folds", [1, 3, 5])
def test_n_folds_sensitivity(generator, n_folds):
    """Varying n_folds should all succeed on clean data."""
    train_df, _, _ = generator.generate("S-2", noise_level=0.0)

    solver = BACON7F(max_depth=3, n_folds=n_folds, verbose=True)
    equation, diagnostics = solver.discover(train_df, target_col="y")

    assert not is_failure_equation(equation)
    assert diagnostics["R-squared"] > 0.999


# Noise resilience: BACON.7F should match or beat BACON.3F

@pytest.mark.parametrize("dataset_id", ["S-2", "T-1"])
def test_noise_resilience_vs_bacon3f(generator, dataset_id):
    """
    At 2% noise, BACON.7F should achieve R² at least as good as BACON.3F.
    This is the core justification for subset voting.
    """
    train_df, _, _ = generator.generate(dataset_id, noise_level=0.02)
    target = CATALOGUE[dataset_id].target

    bacon3f = BACON3F(max_depth=3, verbose=False)
    _, diag_3 = bacon3f.discover(train_df, target_col=target)

    bacon7f = BACON7F(max_depth=3, verbose=False)
    _, diag_7 = bacon7f.discover(train_df, target_col=target)

    r2_3 = diag_3["R-squared"]
    r2_7 = diag_7["R-squared"]

    print(f"\n{dataset_id} @ 2% noise: BACON.3F R²={r2_3:.4f}, BACON.7F R²={r2_7:.4f}")

    assert r2_7 >= r2_3 - 0.05, \
        f"BACON.7F R²={r2_7:.4f} substantially worse than BACON.3F R²={r2_3:.4f}"


# Determinism

def test_determinism(generator):
    """Same seed, same data -> identical results."""
    train_df, _, _ = generator.generate("S-2", noise_level=0.0)

    eq_a, diag_a = BACON7F(max_depth=3).discover(train_df, target_col="y", seed=42)
    eq_b, diag_b = BACON7F(max_depth=3).discover(train_df, target_col="y", seed=42)

    assert eq_a == eq_b
    assert diag_a["R-squared"] == pytest.approx(diag_b["R-squared"])
    assert diag_a["MSE"] == pytest.approx(diag_b["MSE"])
    assert diag_a["MAE"] == pytest.approx(diag_b["MAE"])


# Edge cases

def test_constant_target():
    """If target is already constant, BACON.7F should find it trivially."""
    df = pd.DataFrame({
        "x": np.linspace(1, 10, 20),
        "y": np.full(20, 3.14),
    })
    _, diagnostics = BACON7F(max_depth=3).discover(df, target_col="y")
    assert diagnostics["MSE"] < 1e-10


def test_two_identical_columns():
    """Two identical columns: y = x. Should find y/x = 1 or y - x = 0."""
    vals = np.linspace(1, 10, 20)
    df = pd.DataFrame({"x": vals, "y": vals.copy()})
    equation, diagnostics = BACON7F(max_depth=3).discover(df, target_col="y")

    assert not is_failure_equation(equation)
    assert diagnostics["R-squared"] > 0.999


def test_single_column_returns_failure():
    """Only target column, no independents. Should return failure, not crash."""
    df = pd.DataFrame({"y": np.linspace(1, 10, 20)})
    equation, _ = BACON7F(max_depth=3).discover(df, target_col="y")
    assert is_failure_equation(equation)


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
    solver = BACON7F(max_depth=3, n_folds=3, r2_threshold=0.9, verbose=True)
    equation, diagnostics = solver.discover(df, target_col="y")

    assert not is_failure_equation(equation)
    assert diagnostics["R-squared"] > 0.8
