from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sympy

from symbolic_discovery.algorithms import BACON3F, BACON7F
from symbolic_discovery.data import CATALOGUE, load
from symbolic_discovery.utils import equation_to_metrics


# Clean data

@pytest.mark.parametrize("dataset_id, expected_expr", [
    ("S1", "x1 + x2"),
    ("S2", "x1*x2"),
    ("S3", "x1/(x2 + 1)"),
    ("T1", "x1*x2"),
    ("T2", "x1*x2"),
    ("T3", "0.5*9.81*x1**2"),
    ("T4", "x1*x2/(x3*8.314)"),
    ("T5", "5.67e-8*x1**4"),
])
def test_recovers_law_on_clean_data(dataset_id, expected_expr):
    #  We KNOW BACON.7F can solve clean S, T, with R^2 = 1 (at these params)
    config = CATALOGUE[dataset_id]
    train_df, test_df, _ = load(config, noise=0.0)

    solver = BACON7F(r2_threshold=0.999, scale_factor=1.0, 
                     initial_epsilon=0.01, initial_delta=0.01, 
                     log_level="quiet")
    equation, _ = solver.discover(train_df, target_col=config.target)

    assert equation != "No law found"
    r2, mse, mae = equation_to_metrics(equation, test_df, config.target)
    assert r2 > 0.999
    assert mse < 0.001
    assert mae < 0.001

    _, rhs = equation.split("=", 1)
    discovered = sympy.sympify(rhs.strip())
    expected = sympy.sympify(expected_expr.strip())
    ratio = sympy.simplify(discovered / expected)
    assert ratio.is_number is True
    assert abs(1 - float(ratio)) < 0.01


# n_folds

class TestNFoldsSensitivity:

    @pytest.mark.parametrize("dataset_id", ["S2", "T1"])
    def test_n_folds_1_still_works(self, dataset_id):
        # n_folds=1 should still solve clean data like B3F
        config = CATALOGUE[dataset_id]
        train_df, test_df, _ = load(config, noise=0.0)
        solver = BACON7F(r2_threshold=0.999, scale_factor=1.0, 
                         initial_epsilon=0.01, initial_delta=0.01, 
                         n_folds=1, log_level="quiet")
        equation, _ = solver.discover(train_df, target_col=config.target)
        assert equation != "No law found"
        r2, _, _ = equation_to_metrics(equation, test_df, config.target)
        assert r2 > 0.999

    @pytest.mark.parametrize("n_folds", [1, 3, 5])
    def test_all_n_folds_succeed_on_clean(self, n_folds):
        config = CATALOGUE["S2"]
        train_df, test_df, _ = load(config, noise=0.0)
        solver = BACON7F(r2_threshold=0.999, scale_factor=1.0, 
                         initial_epsilon=0.01,initial_delta=0.01, 
                         n_folds=n_folds, log_level="quiet")
        equation, _ = solver.discover(train_df, target_col=config.target)
        assert equation != "No law found"
        r2, _, _ = equation_to_metrics(equation, test_df, config.target)
        assert r2 > 0.999


# Noise resilience

@pytest.mark.parametrize("dataset_id", ["S2", "T1"])
def test_noise_resilience_vs_bacon3f(dataset_id):
    # Should be better than B3F under noise. 
    config = CATALOGUE[dataset_id]
    train_df, _, _ = load(config, noise=0.02)
    target = config.target

    _, diag_3 = BACON3F(log_level="quiet").discover(train_df, target_col=target)
    _, diag_7 = BACON7F(log_level="quiet").discover(train_df, target_col=target)

    r2_3 = diag_3["R-squared"]
    r2_7 = diag_7["R-squared"]
    assert r2_7 >= r2_3 - 0.05, (
        f"BACON.7F R²={r2_7:.4f} substantially worse than BACON.3F "
        f"R²={r2_3:.4f} at 2% noise on {dataset_id}"
    )


# Determinism

class TestDeterminism:
    def test_same_seed_identical_result(self):
        config = CATALOGUE["S2"]
        train_df, test_df, _ = load(config, noise=0.0)

        eq_a, _ = BACON7F(log_level="quiet").discover(
            train_df, target_col=config.target, seed=73)
        eq_b, _ = BACON7F(log_level="quiet").discover(
            train_df, target_col=config.target, seed=73)

        r2_a, _, _ = equation_to_metrics(eq_a, test_df, config.target)
        r2_b, _, _ = equation_to_metrics(eq_b, test_df, config.target)

        assert eq_a == eq_b
        assert r2_a == pytest.approx(r2_b)


# Edge cases

class TestEdgeCases:
    def test_constant_target_recovered(self):
        df = pd.DataFrame({
            "x": np.linspace(1, 10, 20),
            "y": np.full(20, 3.14),
        })
        eq, _ = BACON7F(log_level="quiet").discover(df, target_col="y")
        _, mse, _ = equation_to_metrics(eq, df, "y")
        assert mse < 1e-10

    def test_two_identical_columns(self):
        vals = np.linspace(1, 10, 20)
        df = pd.DataFrame({"x": vals, "y": vals.copy()})
        equation, _ = BACON7F(log_level="quiet").discover(df, target_col="y")
        r2, _, _ = equation_to_metrics(equation, df, "y")
        assert equation != "No law found"
        assert r2 > 0.999

    def test_single_column_returns_failure_not_crash(self):
        df = pd.DataFrame({"y": np.linspace(1, 10, 20)})
        equation, _ = BACON7F(log_level="quiet").discover(df, target_col="y")
        assert equation == "No law found"
