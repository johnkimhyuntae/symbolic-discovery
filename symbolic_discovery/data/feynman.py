"""
Feynman & Bonus benchmark I/O.

Handles all interaction with the ``feynman_root`` directory tree:
metadata CSVs, whitespace-delimited data files, and the exclusion list.

"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


_META_FILES: dict[str, str] = {
    "F": "FeynmanEquations.csv",
    "B": "BonusEquations.csv",
}

_DATA_DIRS: dict[str, str] = {
    "F": "Feynman_with_units",
    "B": "bonus_with_units",
}

_EXCLUSIONS_PATH = Path(__file__).parent / "feynman_exclusions.json"


# Metadata

@lru_cache(maxsize=4)
def load_metadata(root_dir: str, family: str) -> pd.DataFrame:
    """Load and cache the metadata CSV for *family* (``"F"`` or ``"B"``)."""
    if family not in _META_FILES:
        raise ValueError(f"Unknown benchmark family: {family!r}")
    path = Path(root_dir) / _META_FILES[family]
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark metadata: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def get_metadata_row(
    root_dir: str,
    family: str,
    *,
    eq_id: str | None = None,
    number: int | None = None,
) -> pd.Series:
    """Look up a single equation by Filename (*eq_id*) or *number*."""
    meta = load_metadata(root_dir, family)
    if eq_id is not None:
        match = meta.loc[meta["Filename"].str.strip() == eq_id]
    elif number is not None:
        match = meta.loc[meta["Number"].str.strip() == str(number)]
    else:
        raise ValueError("Provide either eq_id or number")
    if match.empty:
        raise ValueError(
            f"No {family} equation matching eq_id={eq_id!r}, number={number!r}"
        )
    return match.iloc[0]


def list_equation_ids(
    root_dir: str,
    family: str,
    *,
    require_data_file: bool = True,
) -> list[str]:
    """Return Filename strings for every equation in *family*."""
    meta = load_metadata(root_dir, family)
    ids = [
        str(x).strip()
        for x in meta.get("Filename", pd.Series(dtype=str)).tolist()
        if str(x).strip()
    ]
    if not require_data_file:
        return ids
    data_dir = Path(root_dir) / _DATA_DIRS[family]
    return [eq for eq in ids if (data_dir / eq).exists()]


def equation_numbers(root_dir: str, family: str) -> list[int]:
    """Return sorted list of equation numbers that have data files."""
    ids = list_equation_ids(root_dir, family, require_data_file=True)
    meta = load_metadata(root_dir, family)
    nums: list[int] = []
    for eq_id in ids:
        row = meta.loc[meta["Filename"].str.strip() == eq_id]
        if not row.empty:
            try:
                nums.append(int(row.iloc[0]["Number"].strip()))
            except (ValueError, TypeError):
                pass
    return sorted(nums)


def extract_variables(row: pd.Series) -> list[str]:
    """Pull the physics-symbol variable names from a metadata row."""
    n_vars = None
    try:
        n_vars = int(str(row.get("# variables", "")).strip())
    except (ValueError, TypeError):
        pass

    keys = [f"v{i}_name" for i in range(1, 11)]
    if n_vars is not None:
        keys = keys[: min(10, n_vars)]

    variables: list[str] = []
    for k in keys:
        v = row.get(k, "")
        if isinstance(v, str) and v.strip():
            variables.append(v.strip())
    return variables


# Data file loading

def load_data(
    eq_id: str,
    root_dir: str,
    family: str,
    n_samples: int,
    seed: int,
    target: str,
) -> pd.DataFrame:
    """Read *n_samples* rows from a whitespace-delimited data file.

    Returns a DataFrame with columns ``x1, x2, ..., y``, shuffled.
    """
    path = Path(root_dir) / _DATA_DIRS[family] / eq_id
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    rows: list[np.ndarray] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if len(rows) >= n_samples:
                break
            arr = np.fromstring(line.strip(), sep=" ")
            if arr.size:
                rows.append(arr)

    if len(rows) < n_samples:
        raise ValueError(
            f"{path.name}: only {len(rows)} usable rows, requested {n_samples}"
        )

    mat = np.vstack(rows)
    n_cols = mat.shape[1]

    data: dict[str, np.ndarray] = {}
    if n_cols == 1:
        data[target] = mat[:, 0]
    else:
        for i in range(n_cols - 1):
            data[f"x{i + 1}"] = mat[:, i]
        data[target] = mat[:, -1]

    df = pd.DataFrame(data)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# Exclusions

@lru_cache(maxsize=1)
def load_exclusions() -> dict[str, list[str]]:
    """Load ``feynman_exclusions.json`` (reason: list of eq_ids)."""
    with open(_EXCLUSIONS_PATH) as f:
        return json.load(f)


def get_exclusion_reason(eq_id: str) -> str | None:
    """Return the exclusion reason for *eq_id*, or ``None``."""
    for reason, ids in load_exclusions().items():
        if eq_id in ids:
            return reason
    return None