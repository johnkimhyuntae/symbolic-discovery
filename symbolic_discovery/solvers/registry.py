from __future__ import annotations

from .bacon3 import Bacon3Wrapper
from .bacon7 import Bacon7Wrapper
from .pysr import PySRWrapper

SOLVER_REGISTRY = {
    "bacon3": Bacon3Wrapper,
    "bacon7": Bacon7Wrapper,
    "pysr": PySRWrapper,
}
