"""
Dataset catalogue.

Defines the :class:`DatasetConfig` dataclass used by every part of the
framework, and the built-in synthetic (S) and textbook (T) datasets.
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
        family:    One of S, T, F, B.
        variables: Column / physics-symbol names for the independent vars.
        target:    Name of the dependent-variable column.
        formula:   Human-readable equation string.
        domain:    Variable -> (low, high) ranges.  Populated for S/T
                   (used by the generator); empty for F/B.
    """

    key: str
    eq_id: str
    family: str
    variables: list[str]
    target: str
    formula: str
    domain: dict[str, tuple[float, float]] = field(default_factory=dict)


# Built-in datasets
CATALOGUE: dict[str, DatasetConfig] = {
    # Synthetic
    "S1": DatasetConfig("S1", "S1", "S", ["x1", "x2"], "y", "x1 + x2",
                        {"x1": (-5, 5), "x2": (-5, 5)}),
    "S2": DatasetConfig("S2", "S2", "S", ["x1", "x2"], "y", "x1 * x2",
                        {"x1": (1, 5), "x2": (1, 5)}),
    "S3": DatasetConfig("S3", "S3", "S", ["x1", "x2"], "y", "x1 / (x2 + 1)",
                        {"x1": (1, 10), "x2": (1, 10)}),
    "S4": DatasetConfig("S4", "S4", "S", ["x1", "x2"], "y", "x1**2 + x2**2",
                        {"x1": (-3, 3), "x2": (-3, 3)}),

    # Textbook laws
    "T1": DatasetConfig("T1", "T1", "T", ["I", "R"], "V", "I * R",
                        {"I": (0, 2), "R": (1, 10)}),
    "T2": DatasetConfig("T2", "T2", "T", ["k", "x"], "F", "k * x",
                        {"k": (1, 10), "x": (-1, 1)}),
    "T3": DatasetConfig("T3", "T3", "T", ["t"], "s", "0.5 * 9.81 * t**2",
                        {"t": (0, 2)}),
    "T4": DatasetConfig("T4", "T4", "T", ["P", "V", "n"], "T",
                        "(P * V) / (n * 8.314)",
                        {"P": (1, 5), "V": (10, 30), "n": (1, 2)}),
    "T5": DatasetConfig("T5", "T5", "T", ["T"], "P", "5.67e-8 * T**4",
                        {"T": (100, 500)}),
}