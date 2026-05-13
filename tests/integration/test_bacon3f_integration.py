"""Integration tests for BACON.3F."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sympy

from symbolic_discovery.algorithms import BACON3F
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
    """BACON.3F should recover clean catalogue laws."""
    config = CATALOGUE[dataset_id]
    train_df, test_df, _ = load(config, noise=0.0)

    solver = BACON3F(r2_threshold=0.999, constancy_threshold=0.01, log_level="quiet")
    equation, _ = solver.discover(train_df, target_col=config.target)

    assert equation != "No law found"
    r2, mse, mae = equation_to_metrics(equation, test_df, config.target)
    assert r2 > 0.999
    assert mse < 0.001
    assert mae < 0.001

    # Symbolic equivalence: the discovered RHS must equal the expected
    # RHS up to a constant ratio.
    _, rhs = equation.split("=", 1)
    discovered = sympy.sympify(rhs.strip())
    expected = sympy.sympify(expected_expr.strip())
    ratio = sympy.simplify(discovered / expected)
    assert ratio.is_number is True, \
        f"Discovered {discovered} not symbolically proportional to {expected}"
    assert abs(1 - float(ratio)) < 0.01


# Determinism

class TestDeterminism:

    def test_same_seed_identical_result(self):
        """Identical seeds should produce identical results."""
        config = CATALOGUE["S2"]
        train_df, test_df, _ = load(config, noise=0.0)

        eq_a, _ = BACON3F(log_level="quiet").discover(
            train_df, target_col=config.target, seed=73)
        eq_b, _ = BACON3F(log_level="quiet").discover(
            train_df, target_col=config.target, seed=73)

        r2_a, mse_a, mae_a = equation_to_metrics(eq_a, test_df, config.target)
        r2_b, mse_b, mae_b = equation_to_metrics(eq_b, test_df, config.target)

        assert eq_a == eq_b
        assert r2_a == pytest.approx(r2_b)
        assert mse_a == pytest.approx(mse_b)
        assert mae_a == pytest.approx(mae_b)


# Hand-crafted edge cases

class TestEdgeCases:

    def test_constant_target_recovered(self):
        """A constant target should be recovered."""
        df = pd.DataFrame({
            "x": np.linspace(1, 10, 20),
            "y": np.full(20, 3.14),
        })
        equation, _ = BACON3F(log_level="quiet").discover(df, target_col="y")
        _, mse, _ = equation_to_metrics(equation, df, "y")
        assert mse < 1e-10

    def test_two_identical_columns(self):
        """Identical columns should still be solved."""
        vals = np.linspace(1, 10, 20)
        df = pd.DataFrame({"x": vals, "y": vals.copy()})
        equation, _ = BACON3F(log_level="quiet").discover(df, target_col="y")
        r2, _, _ = equation_to_metrics(equation, df, "y")
        assert equation != "No law found"
        assert r2 > 0.999

    def test_single_column_returns_failure_not_crash(self):
        """A missing feature column should fail without crashing."""
        df = pd.DataFrame({"y": np.linspace(1, 10, 20)})
        equation, _ = BACON3F(log_level="quiet").discover(df, target_col="y")
        assert equation == "No law found"