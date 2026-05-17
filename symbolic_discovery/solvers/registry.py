from __future__ import annotations

from symbolic_discovery.solvers.base import BaseSolver

from .bacon3f import BACON3FSolver
from .bacon7f import BACON7FSolver
from .pysr import PySRSolver

SOLVER_REGISTRY: dict[str, type[BaseSolver]] = {
    "bacon3f": BACON3FSolver,
    "bacon7f": BACON7FSolver,
    "pysr": PySRSolver,
}
