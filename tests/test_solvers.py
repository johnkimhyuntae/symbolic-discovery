import pytest
import pandas as pd
from symbolic_discovery.solvers.registry import SOLVER_REGISTRY
from symbolic_discovery.solvers.base import SolverResult


@pytest.mark.parametrize("solver_name", ["bacon3f", "bacon7f"])
def test_solver_returns_result(solver_name):
    cls = SOLVER_REGISTRY[solver_name]
    solver = cls()
    train_df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    test_df = pd.DataFrame({"x": [4, 5], "y": [8, 10]})
    result = solver.solve(train_df, test_df, target_col="y", seed=42)
    assert isinstance(result, SolverResult)
