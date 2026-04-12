from __future__ import annotations

from .bacon3f import Bacon3FSolver
from .bacon7f import Bacon7FSolver
from .pysr import PySRSolver

SOLVER_REGISTRY = {
    "bacon3f": Bacon3FSolver,
    "bacon7f": Bacon7FSolver,
    "pysr": PySRSolver,
}
