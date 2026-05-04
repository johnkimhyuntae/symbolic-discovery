"""Shared pytest fixtures."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, List

import pandas as pd
import pytest

from symbolic_discovery.solvers import SOLVER_REGISTRY, SolverResult


# Marker registration

# Keep marker registration here instead of a separate pytest config file.
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running tests (real BACON / PySR runs)",
    )
    config.addinivalue_line(
        "markers", "integration: composes multiple modules (real solvers, real I/O)",
    )
    config.addinivalue_line(
        "markers", "e2e: full CLI round-trips",
    )
    config.addinivalue_line(
        "markers", "feynman: requires the feynman/ data directory on disk",
    )
    config.addinivalue_line(
        "markers", "pysr: requires pysr to be installed",
    )


# Fake solver

class FakeSolver:
    """
    Deterministic stand-in for real solvers.

    Records the kwargs it was constructed with so tests can verify that
    variant params were forwarded by the runner. Always returns the same
    SolverResult, so tests asserting on metrics are deterministic.
    """

    instances: List["FakeSolver"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        FakeSolver.instances.append(self)

    def solve(self, train_df, test_df, target_col, seed):
        return SolverResult(
            equation=f"{target_col} = fake({target_col})",
            raw_equation=f"fake|seed={seed}|kwargs={sorted(self.kwargs.items())}",
            r2=0.9876,
            mse=0.0123,
            mae=0.0099,
            time_sec=0.001,
            status="Found",
        )


@pytest.fixture
def fake_solver(monkeypatch):
    """Register FakeSolver under the name 'fake' for the duration of a test."""
    FakeSolver.instances = []
    monkeypatch.setitem(SOLVER_REGISTRY, "fake", FakeSolver)
    yield FakeSolver
    FakeSolver.instances = []


# Args helper

def _default_runner_args(**overrides) -> Namespace:
    """Build a Namespace with every flag the runner reads.

    The runner's _build_runs reads many flags via ``args.x or default``,
    so every field must be present (even if None) to avoid AttributeError.
    """
    base = dict(
        models=None,
        variant=[],
        sweep=[],
        study=None,
        datasets=None,
        target=None,
        noise=None,
        noise_types=None,
        n_samples=None,
        seeds=None,
        output_root="results",
        output="experiment_results.csv",
        log_level="default",
        feynman_root="feynman",
    )
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture
def default_runner_args():
    """Factory fixture: call to get a Namespace, optionally overriding fields."""
    return _default_runner_args


# CSV fieldnames

@pytest.fixture
def csv_fieldnames() -> List[str]:
    """The canonical column order written by runner._write_row."""
    return [
        "run_id", "dataset", "method", "variant", "params_json",
        "noise", "noise_type", "n_samples", "seed",
        "equation", "raw_equation", "r2", "mse", "mae", "time_s", "status",
    ]


# Simple DataFrames

@pytest.fixture
def simple_xy_df() -> pd.DataFrame:
    """y = 2x for x in [1..10] — minimal solvable dataset."""
    return pd.DataFrame({
        "x": list(range(1, 11)),
        "y": [2 * i for i in range(1, 11)],
    })


@pytest.fixture
def simple_train_test():
    """(train_df, test_df) split of y = 2x for wrapper tests."""
    train = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 12]})
    test = pd.DataFrame({"x": [7, 8, 9, 10], "y": [14, 16, 18, 20]})
    return train, test