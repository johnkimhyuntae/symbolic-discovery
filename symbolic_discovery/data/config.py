"""
Defines the :class:`DatasetConfig` dataclass used in the data layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Unified descriptor for any dataset in the framework.

    Attributes:
        key:       User-facing label: S1, T3, F8, B2.
        eq_id:     Internal / filename identifier.  Same as *key* for S/T;
                   the Feynman Filename (e.g. I.12.1) for F/B.
        family:    One of S, T, F, B, C.
        variables: Column / physics-symbol names for the independent vars.
        target:    Name of the dependent-variable column.
        formula:   Human-readable equation string.
        domain:    Variable -> (low, high) ranges.  Populated for S/T
                   (used by the generator); empty for F/B/C.
    """

    key: str
    eq_id: str
    family: str
    variables: list[str]
    target: str
    formula: str
    domain: dict[str, tuple[float, float]] = field(default_factory=dict)
