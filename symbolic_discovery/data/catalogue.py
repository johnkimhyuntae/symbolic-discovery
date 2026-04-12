from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    id: str
    variables: list[str]
    target: str
    formula: str
    domain: dict[str, tuple[float, float]]
    n_samples: int = 400


CATALOGUE: dict[str, DatasetConfig] = {
    # Synthetic Functions
    "S-1": DatasetConfig("S-1", ["x1", "x2"], "y", "x1 + x2", {"x1": (-5, 5), "x2": (-5, 5)}),
    "S-2": DatasetConfig("S-2", ["x1", "x2"], "y", "x1 * x2", {"x1": (1, 5), "x2": (1, 5)}),
    "S-3": DatasetConfig("S-3", ["x1", "x2"], "y", "x1 / (x2 + 1)", {"x1": (1, 10), "x2": (1, 10)}),
    "S-4": DatasetConfig("S-4", ["x1", "x2"], "y", "x1**2 + x2**2", {"x1": (-3, 3), "x2": (-3, 3)}),

    # Textbook Laws
    "T-1": DatasetConfig("T-1", ["I", "R"], "V", "I * R", {"I": (0, 2), "R": (1, 10)}),
    "T-2": DatasetConfig("T-2", ["k", "x"], "F", "k * x", {"k": (1, 10), "x": (-1, 1)}),
    "T-3": DatasetConfig("T-3", ["t"], "s", "0.5 * 9.81 * t**2", {"t": (0, 2)}),
    "T-4": DatasetConfig(
        "T-4",
        ["P", "V", "n"],
        "T",
        "(P * V) / (n * 8.314)",
        {"P": (1, 5), "V": (10, 30), "n": (1, 2)},
    ),
    "T-5": DatasetConfig("T-5", ["T"], "P", "5.67e-8 * T**4", {"T": (100, 500)}),
}
