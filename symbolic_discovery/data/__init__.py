"""
symbolic_discovery.data: unified dataset interface.

All public API is defined in :mod:`.api`; this file re-exports it so
that consumers can write ``from symbolic_discovery.data import resolve``.
"""

from .api import (
    expand_datasets,
    get_exclusion_reason,
    load,
    pretty_equation,
    resolve,
)
from .config import DatasetConfig
from .synthetic import CATALOGUE

__all__ = [
    "CATALOGUE",
    "DatasetConfig",
    "expand_datasets",
    "get_exclusion_reason",
    "load",
    "pretty_equation",
    "resolve",
]
