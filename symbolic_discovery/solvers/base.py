from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class SolverResult:
    """TODO: docstring."""
    equation: str
    raw_equation: str
    r2: float
    mse: float
    mae: float
    time_sec: float
    status: str
    logs: list[str] = field(default_factory=list)


class BaseSolver(ABC):
    """TODO: docstring."""
    @abstractmethod
    def solve(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, seed: int) -> SolverResult:
        raise NotImplementedError
