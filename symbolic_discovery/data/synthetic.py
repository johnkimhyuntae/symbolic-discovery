"""
Synthetic data generation and noise injection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

from .catalogue import DatasetConfig


def inject_noise(
    df: pd.DataFrame,
    target: str,
    noise_level: float,
    seed: int,
) -> pd.DataFrame:
    """Add Gaussian noise scaled to the target column's range.

    Returns *df* unchanged when *noise_level* <= 0.
    """
    if noise_level <= 0.0 or target not in df.columns:
        return df
    y = df[target].to_numpy()
    y_range = float(np.ptp(y))
    scale = y_range if y_range > 1e-9 else 1.0
    rng = np.random.default_rng(seed)
    out = df.copy()
    # TBD: made multiplicative for now
    out[target] = y * (1 + rng.normal(0.0, noise_level, len(y)))
    return out


def generate(
    config: DatasetConfig,
    *,
    noise_level: float = 0.0,
    n_samples: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic DataFrame from a catalogue config."""
    rng = np.random.default_rng(seed)

    data: dict[str, np.ndarray] = {}
    for var in config.variables:
        low, high = config.domain[var]
        data[var] = rng.uniform(low, high, n_samples)

    df = pd.DataFrame(data)
    df[config.target] = df.eval(config.formula)

    if noise_level > 0.0:
        df = inject_noise(df, config.target, noise_level, seed)

    return df