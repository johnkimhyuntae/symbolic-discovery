"""
Defines the :class:`SolverResult` dataclass and the :class:`BaseSolver` abstract class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd
from typing import Any


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
    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def solve(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, seed: int) -> SolverResult:
        raise NotImplementedError
