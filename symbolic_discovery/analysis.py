"""
Post-hoc analysis of experiment runner CSV output.

All functions take and return DataFrames, with no I/O beyond the 
top-level :func:`load_results` helper. Use them from a notebook or 
a small ``make_figures.py`` script to build the tables and plots 
that go in the dissertation.

The runner writes one row per (variant, dataset, noise, noise_type,
n_samples, seed). The functions here collapse, pivot, and reshape that
long-format CSV into the views useful for evaluation:

    aggregate_seeds        collapse seed dimension to mean/std
    success_rate           fraction of runs that produced an equation
    summarise_by_variant   one row per variant, sorted by R²
    pivot_compare          datasets x variants table for a chosen metric
    noise_curve            long-format frame ready for line plots
    expand_params          promote sweep params from JSON to columns
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import pandas as pd


# I/O

def load_results(path: str) -> pd.DataFrame:
    """Load a runner CSV. Parses ``params_json`` into a ``params`` dict column."""
    df = pd.read_csv(path)
    if "params_json" in df.columns:
        df["params"] = df["params_json"].apply(
            lambda s: json.loads(s) if isinstance(s, str) and s else {}
        )
    return df


# Filtering

def successful(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only runs that produced an equation (``status == 'Found'``)."""
    return df[df["status"] == "Found"].copy()


# Param expansion (for hyperparameter sweeps)

def expand_params(df: pd.DataFrame, params: Sequence[str]) -> pd.DataFrame:
    """
    Promote selected params from ``params_json`` into top-level columns.

    Useful for sweep analysis: after expanding, you can pivot directly on
    the swept hyperparameters, e.g.::

        df = expand_params(df, ["max_depth", "scale_factor"])
        pivot_compare(df, rows="max_depth", cols="scale_factor")
    """
    df = df.copy()
    if "params" not in df.columns:
        df["params"] = df["params_json"].apply(
            lambda s: json.loads(s) if isinstance(s, str) and s else {}
        )
    for p in params:
        df[p] = df["params"].apply(lambda d: d.get(p))
    return df


# Aggregation

DEFAULT_GROUP_KEYS = (
    "variant", "method", "dataset", "noise", "noise_type", "n_samples",
)
DEFAULT_METRICS = ("r2", "mse", "mae", "time_s")


def aggregate_seeds(
    df: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
    group_by: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> pd.DataFrame:
    """
    Collapse the seed dimension (and any other axes dropped from
    ``group_by``): for each cell, compute mean and std of each metric.
    Output columns are ``{metric}_mean``, ``{metric}_std``, plus
    ``n_runs`` (count of rows in the cell).

    Failed runs have ``mse=inf`` / ``mae=inf``; those are mapped to NaN
    before aggregation so a single failure doesn't poison the mean.
    Note: failed runs still contribute ``r2=0.0`` to the R² mean by
    design — call :func:`successful` first if you want quality
    conditional on success.
    """
    df = df.replace({"mse": {np.inf: np.nan}, "mae": {np.inf: np.nan}})
    keys = [k for k in group_by if k in df.columns]
    metrics = [m for m in metrics if m in df.columns]

    g = df.groupby(list(keys), dropna=False)
    agg = g[list(metrics)].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg["n_runs"] = g.size()
    return agg.reset_index()


def success_rate(
    df: pd.DataFrame,
    by: Sequence[str] = ("variant", "dataset"),
) -> pd.DataFrame:
    """Fraction of runs with ``status == 'Found'`` per group."""
    keys = [k for k in by if k in df.columns]
    tmp = df.assign(_success=(df["status"] == "Found").astype(float))
    g = tmp.groupby(list(keys), dropna=False)
    return pd.DataFrame({
        "success_rate": g["_success"].mean(),
        "n_runs": g.size(),
    }).reset_index()


def summarise_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per variant: success rate, mean/std R², mean/total time.
    Sorted by mean R² descending — useful as a top-level leaderboard
    for the evaluation chapter.
    """
    tmp = df.assign(_success=(df["status"] == "Found").astype(float))
    g = tmp.groupby("variant", dropna=False)
    out = pd.DataFrame({
        "n_runs": g.size(),
        "success_rate": g["_success"].mean(),
        "r2_mean": g["r2"].mean(),
        "r2_std": g["r2"].std(),
        "time_s_mean": g["time_s"].mean(),
        "time_s_total": g["time_s"].sum(),
    }).reset_index()
    return out.sort_values("r2_mean", ascending=False).reset_index(drop=True)


# Pivots and reshapes for plots/tables

def pivot_compare(
    df: pd.DataFrame,
    metric: str = "r2",
    rows: str = "dataset",
    cols: str = "variant",
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """
    Pivot for side-by-side comparison: e.g. datasets on rows, variants
    on columns, mean R² in cells. Drops straight into ``df.to_latex()``
    for dissertation tables, or into ``sns.heatmap`` for hyperparameter
    grids.
    """
    return df.pivot_table(values=metric, index=rows, columns=cols, aggfunc=aggfunc)


def noise_curve(
    df: pd.DataFrame,
    metric: str = "r2",
    group_by: Sequence[str] = ("variant", "noise"),
) -> pd.DataFrame:
    """
    Long-format DataFrame ready for plotting: one row per
    (variant, noise) cell, with mean/std of the chosen metric.
    Feeds directly into ``sns.lineplot(x='noise', y='r2_mean',
    hue='variant', ...)``.
    """
    keys = list(group_by)
    g = df.groupby(keys, dropna=False)[metric]
    return pd.DataFrame({
        f"{metric}_mean": g.mean(),
        f"{metric}_std": g.std(),
        "n": g.size(),
    }).reset_index()