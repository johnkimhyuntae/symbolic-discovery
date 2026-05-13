"""
Shared post-hoc analysis utilities.
"""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

import sympy
from symbolic_discovery.data import resolve


# I/O

def load_csvs(paths):
    """Concatenate runner CSVs, de-dupe on the experiment key, add ``found``."""
    paths = [Path(p) for p in paths]
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    df = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
    df = df.drop_duplicates(
        subset=["variant", "dataset", "noise", "noise_type", "n_samples", "seed"],
        keep="first",
    ).reset_index(drop=True)
    df["found"] = df["status"] == "Found"
    print(f"loaded {len(df)} rows from {[str(p) for p in existing]} "
          f"| {df['variant'].nunique()} variants "
          f"| {df['dataset'].nunique()} datasets "
          f"| seeds={sorted(df['seed'].unique())}")
    return df


# Aggregation

def aggregate(df: pd.DataFrame, keys) -> pd.DataFrame:
    """
    Collapse the seed dimension for each cell defined by ``keys``.
    """
    g    = df.groupby(keys, dropna=False)
    succ = df[df["found"]].groupby(keys, dropna=False)

    out = pd.concat([
        g.size().rename("n_runs"),
        g["found"].mean().rename("success_rate"),
        succ["r2"].mean().rename("r2_mean"),
        succ["r2"].std().rename("r2_std"),
        succ["r2"].sem().rename("r2_sem"),
        g["time_s"].mean().rename("time_mean"),
        g["time_s"].std().rename("time_std"),
        g["time_s"].sem().rename("time_sem"),
        succ["time_s"].mean().rename("time_mean_found"),
        succ["time_s"].std().rename("time_std_found"),
        succ["time_s"].sem().rename("time_sem_found"),
    ], axis=1).reset_index()
    return out


def parse_variants(df, regex, param_map, baseline_variant, baseline_params):
    """Split each ``variant`` string into ``(param, param_value)`` columns
    and echo the baseline row once per parameter so the baseline point
    appears on every sensitivity curve."""
    pat = re.compile(regex)

    def extract(name):
        m = pat.match(name)
        if not m:
            return pd.Series([None, None])
        full, dtype = param_map[m.group(1)]
        return pd.Series([full, dtype(m.group(2))])

    df[["param", "param_value"]] = df["variant"].apply(extract)

    baseline = df[df["variant"] == baseline_variant]
    extra_rows = []
    for full_name, baseline_value in baseline_params.items():
        if baseline.empty:
            break
        copy = baseline.copy()
        copy["param"] = full_name
        copy["param_value"] = baseline_value
        extra_rows.append(copy)
    return pd.concat([df] + extra_rows, ignore_index=True)


def equivalence_ratio(discovered: str, key: str, *, feynman_root: str = "feynman") -> float:
    """
    If the discovered formula is symbolically proportional to the expected
    formula, return the constant of proportionality.  Otherwise, return 0.0.
    """
    expected = resolve(key, feynman_root=feynman_root).formula
    rhs = lambda s: s.split("=", 1)[1] if "=" in s else s
    ratio = sympy.simplify(sympy.sympify(rhs(discovered))
                           / sympy.sympify(rhs(expected)))
    if ratio.free_symbols:
        return 0.0
    try:
        return float(ratio)
    except (TypeError, ValueError):
        return 0.0