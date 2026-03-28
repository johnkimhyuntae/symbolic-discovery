from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class SolverResult:
    equation: str
    raw_equation: str
    train_r2: float
    mse: float
    time_sec: float
    status: str


class BaseSolver(ABC):
    @abstractmethod
    def solve(self, train_df: pd.DataFrame, target_col: str, seed: int) -> SolverResult:
        raise NotImplementedError
