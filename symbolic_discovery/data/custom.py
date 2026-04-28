"""
Custom CSV dataset loading.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str, n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Read a CSV file, optionally truncated to *n_samples* rows."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(p)
    if n_samples is not None and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed)
    return df