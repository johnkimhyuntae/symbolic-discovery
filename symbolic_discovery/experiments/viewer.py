#!/usr/bin/env python3
"""
Results viewer for symbolic-discovery experiment output.

Two modes:

    default   one table per (variant, noise, noise_type, n_samples, seed),
              with datasets as rows. Shows every cell in the experiment
              grid at full per-seed granularity.

    stats     aggregates over seeds and datasets: one row per
              (variant, noise, noise_type, n_samples) cell. Printed and
              saved to ``<input>_stats.csv`` next to the input file.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

from symbolic_discovery.analysis import (
    aggregate_seeds,
    success_rate,
    successful,
)

console = Console()


# Formatting helpers

def _equation_col(df: pd.DataFrame) -> str:
    """Whichever equation column the CSV actually has."""
    for c in ("equation", "best_equation"):
        if c in df.columns:
            return c
    return ""


def _evaluation(status: str, r2_str: str) -> Text:
    """Colour-coded grade based on status and R²."""
    if status != "Found":
        return Text("Fail", style="bold red")
    try:
        v = float(r2_str)
    except (TypeError, ValueError):
        return Text("?", style="dim")
    if v > 0.999:
        return Text("Perfect", style="blue")
    if v > 0.900:
        return Text("Good", style="green")
    if v > 0.500:
        return Text("Poor", style="yellow")
    return Text("Bad", style="red")


def _fmt(val, fmt: str = ".4f") -> str:
    if pd.isna(val):
        return "—"
    try:
        v = float(val)
        if pd.isna(v):
            return "—"
        return f"{v:{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_time(val) -> str:
    f = _fmt(val, ".3f")
    return f"{f}s" if f != "—" else "—"


def _truncate(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _params_label(params_json) -> str:
    """Compact ``k=v, k=v`` rendering of a params_json string."""
    if not isinstance(params_json, str) or not params_json or params_json == "{}":
        return ""
    try:
        d = json.loads(params_json)
    except json.JSONDecodeError:
        return ""
    if not d:
        return ""
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items()))


def _cell_subtitle(noise, noise_type, n_samples, seed) -> str:
    """Format the (noise, noise_type, n_samples, seed) coordinates of a cell."""
    parts = [f"noise={_fmt(noise, '.3g')}"]
    if pd.notna(noise_type):
        parts.append(str(noise_type))
    if pd.notna(n_samples):
        try:
            parts.append(f"n={int(n_samples)}")
        except (ValueError, TypeError):
            pass
    if pd.notna(seed):
        try:
            parts.append(f"seed={int(seed)}")
        except (ValueError, TypeError):
            pass
    return "  ".join(parts)


_DS_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _dataset_sort_key(s) -> tuple:
    """Sort dataset keys naturally: S1, S2, ..., S10, T1, ..., F1, F2, ..., F11."""
    m = _DS_RE.match(str(s))
    if m:
        return (m.group(1), int(m.group(2)))
    return (str(s), 0)


# Default mode: one table per cell

DEFAULT_GROUP_COLS = ("variant", "noise", "noise_type", "n_samples", "seed")


def _view_default(df: pd.DataFrame) -> None:
    eq_col = _equation_col(df)
    group_cols = [c for c in DEFAULT_GROUP_COLS if c in df.columns]
    has_r2 = "r2" in df.columns
    has_time = "time_s" in df.columns

    for group_key, group in df.groupby(group_cols, sort=True, dropna=False):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        kv = dict(zip(group_cols, keys))

        params = ""
        if "params_json" in group.columns:
            params = _params_label(group["params_json"].iloc[0])

        title = f"{kv.get('variant', '?')}"
        if params:
            title += f"  [dim]({params})[/dim]"
        subtitle = _cell_subtitle(
            kv.get("noise"), kv.get("noise_type"),
            kv.get("n_samples"), kv.get("seed"),
        )

        group = group.assign(
            _sort=group["dataset"].map(_dataset_sort_key),
        ).sort_values("_sort").drop(columns="_sort")

        t = Table(
            title=f"  {title}  [dim]{subtitle}[/dim]",
            title_style="bold cyan",
            box=box.ROUNDED,
            show_lines=False,
            padding=(0, 1),
        )
        t.add_column("Dataset", style="bold")
        t.add_column("Grade", justify="center")
        if eq_col:
            t.add_column("Equation")
        if has_r2:
            t.add_column("R²", justify="right")
        if has_time:
            t.add_column("Time", justify="right", style="dim")

        for _, row in group.iterrows():
            cells: list[Any] = [str(row["dataset"])]
            cells.append(_evaluation(row.get("status", ""), _fmt(row.get("r2", ""))))
            if eq_col:
                cells.append(_truncate(row.get(eq_col, ""), 60))
            if has_r2:
                cells.append(_fmt(row["r2"]))
            if has_time:
                cells.append(_fmt_time(row["time_s"]))
            t.add_row(*cells)

        console.print(t)
        console.print()


# Stats mode: aggregate over seeds and datasets, save to CSV

# Group keys for stats: every cell axis except 'seed' and 'dataset'.
STATS_GROUP_COLS = ("variant", "method", "noise", "noise_type", "n_samples")


def _view_stats(df: pd.DataFrame, save_path: Path | None) -> None:
    group_by = [c for c in STATS_GROUP_COLS if c in df.columns]

    # Total counts and success rate over all rows.
    sr = success_rate(df, by=group_by)

    # Quality metrics computed only over successful runs — failed rows
    # have r2=0.0 / mse=inf / mae=inf and would poison the means.
    found = successful(df)
    quality = aggregate_seeds(found, group_by=group_by).rename(
        columns={"n_runs": "n_found"}
    )

    # Left-merge: every cell from sr; metric NaNs where zero successes.
    cell = sr.merge(quality, on=group_by, how="left")
    cell["n_found"] = cell["n_found"].fillna(0).astype(int)

    # One table per (noise, noise_type, n_samples); variants as rows.
    partition_cols = [c for c in ("noise", "noise_type", "n_samples")
                      if c in cell.columns]
    sort_cols = partition_cols + (["variant"] if "variant" in cell.columns else [])
    if sort_cols:
        cell = cell.sort_values(sort_cols).reset_index(drop=True)

    # Section header (printed once).
    console.print(
        "[bold cyan]Stats[/bold cyan]  "
        "[dim](aggregated over seeds and datasets; "
        "R²/time over successful runs only)[/dim]"
    )
    console.print()

    if partition_cols:
        groups = list(cell.groupby(partition_cols, sort=True, dropna=False))
    else:
        groups = [((), cell)]

    for key, sub in groups:
        keys = key if isinstance(key, tuple) else (key,)
        kv = dict(zip(partition_cols, keys))

        # Subtitle: noise=X  noise_type  n=N
        parts: list[str] = []
        if "noise" in kv:
            parts.append(f"noise={_fmt(kv['noise'], '.3g')}")
        if "noise_type" in kv and pd.notna(kv["noise_type"]):
            parts.append(str(kv["noise_type"]))
        if "n_samples" in kv and pd.notna(kv["n_samples"]):
            try:
                parts.append(f"n={int(kv['n_samples'])}")
            except (ValueError, TypeError):
                pass
        subtitle = "  ".join(parts)

        t = Table(
            title=f"  {subtitle}" if subtitle else None,
            title_style="bold cyan",
            box=box.ROUNDED,
            show_lines=False,
            padding=(0, 1),
        )
        t.add_column("Variant", style="bold")
        t.add_column("Runs", justify="right")
        t.add_column("Found", justify="right")
        t.add_column("Found %", justify="right")
        t.add_column("Mean R²", justify="right")
        t.add_column("Std R²", justify="right")
        t.add_column("Mean time", justify="right")

        for _, r in sub.iterrows():
            sr_val = float(r["success_rate"])
            sr_style = (
                "green" if sr_val >= 0.8
                else "yellow" if sr_val >= 0.5
                else "red"
            )
            t.add_row(
                str(r["variant"]),
                str(int(r["n_runs"])),
                Text(str(int(r["n_found"])),
                     style=sr_style if sr_val < 1.0 else ""),
                Text(f"{sr_val * 100:.1f}%", style=sr_style),
                _fmt(r.get("r2_mean")),
                _fmt(r.get("r2_std")),
                _fmt_time(r.get("time_s_mean")),
            )
        console.print(t)
        console.print()

    # Save full (un-partitioned) version to CSV.
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cell.to_csv(save_path, index=False)
        console.print(
            f"[dim]Saved {len(cell)} rows × {len(cell.columns)} cols to[/dim] "
            f"[bold]{save_path}[/bold]"
        )
    else:
        console.print(
            f"[dim]{len(cell)} rows — pass --save-path to write CSV[/dim]"
        )
    console.print()


# Entry point

def view_results(
    csv_path: str,
    mode: str = "default",
    save_path: str | None = None,
) -> None:
    p = Path(csv_path)
    if not p.exists():
        console.print(f"[red]File not found:[/red] {csv_path}")
        return

    df = pd.read_csv(p)

    required = {"variant", "method", "status"}
    missing = required - set(df.columns)
    if missing:
        console.print(
            f"[red]CSV missing required columns:[/red] {', '.join(sorted(missing))}"
        )
        return

    console.print()
    console.rule(f"[bold]{p.name}[/bold]  [dim]({len(df)} rows)[/dim]",
                 style="bright_black")
    console.print()

    if mode == "default":
        _view_default(df)
    elif mode == "stats":
        if save_path is None:
            save = p.with_name(f"{p.stem}_stats.csv")
        else:
            save = Path(save_path)
        _view_stats(df, save)
    else:
        console.print(f"[red]Unknown mode:[/red] {mode}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="View symbolic-discovery experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes:
    default   one table per (variant, noise, noise_type, n_samples, seed)
    stats     aggregate across seeds; saves <input>_stats.csv

examples:
    symbolic-discovery view results.csv
    symbolic-discovery view results.csv --mode stats
    symbolic-discovery view results.csv --mode stats --save-path my_stats.csv""",
    )
    parser.add_argument("csv_file", help="Path to results CSV")
    parser.add_argument(
        "--mode", "-m",
        choices=["default", "stats"],
        default="default",
        help="Display mode (default: default)",
    )
    parser.add_argument(
        "--save-path", default=None,
        help="In stats mode, where to write per-cell stats CSV "
             "(default: <input>_stats.csv next to the input file).",
    )
    args = parser.parse_args(argv)
    view_results(args.csv_file, args.mode, args.save_path)


if __name__ == "__main__":
    main()