from __future__ import annotations

from .bacon3f import BACON3FSolver
from .bacon7f import BACON7FSolver
from .pysr import PySRSolver

SOLVER_REGISTRY = {
    "bacon3f": BACON3FSolver,
    "bacon7f": BACON7FSolver,
    "pysr": PySRSolver,
}
