"""
Results viewer for symbolic-discovery experiment output.

Default behaviour: prints one table per (variant, noise, noise_type,
n_samples, seed), with datasets as rows.

With ``--stats``: aggregates over seeds and datasets into one row per
(variant, method, noise, noise_type, n_samples) cell. Reports run
counts, success rate, R² mean/std/sem (over successful runs), and
wall-clock mean/std/sem both over all runs and over successful runs
only. Printed and saved to ``<input>_stats.csv`` next to the input file.
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

from ..utils.analysis import aggregate

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


def _fmt_mean_err(mean, err, fmt: str = ".4f", suffix: str = "") -> str:
    if pd.isna(mean):
        return "—"
    m = _fmt(mean, fmt)
    if pd.isna(err):
        return f"{m}{suffix}"
    return f"{m}{suffix} ± {_fmt(err, fmt)}{suffix}"


def _truncate(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _params_label(params_json) -> str:
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

    df = df.copy()
    df["found"] = df["status"] == "Found"

    cell = aggregate(df, group_by)

    cell["n_found"] = (cell["n_runs"] * cell["success_rate"]).round().astype(int)

    # One table per (noise, noise_type, n_samples); variants as rows.
    partition_cols = [c for c in ("noise", "noise_type", "n_samples")
                      if c in cell.columns]
    sort_cols = partition_cols + (["variant"] if "variant" in cell.columns else [])
    if sort_cols:
        cell = cell.sort_values(sort_cols).reset_index(drop=True)

    # Section header
    console.print(
        "[bold cyan]Stats[/bold cyan]  "
        "[dim](aggregated over seeds and datasets; "
        "R² and time-found over successful runs only; "
        "mean ± sem, with std shown separately)[/dim]"
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
        if "noise_type" in kv and pd.notna(kv["noise_type"]): # type: ignore
            parts.append(str(kv["noise_type"]))
        if "n_samples" in kv and pd.notna(kv["n_samples"]): # type: ignore
            try:
                parts.append(f"n={int(kv['n_samples'])}") # type: ignore
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
        t.add_column("R²", justify="right")
        t.add_column("R² sem", justify="right", style="dim")
        t.add_column("Time", justify="right")
        t.add_column("Time sem", justify="right", style="dim")
        t.add_column("Time-found", justify="right")
        t.add_column("Time-found sem", justify="right", style="dim")

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
                _fmt(r.get("r2_sem")),
                _fmt_time(r.get("time_mean")),
                _fmt_time(r.get("time_sem")),
                _fmt_time(r.get("time_mean_found")),
                _fmt_time(r.get("time_sem_found")),
            )
        console.print(t)
        console.print()

    # Save full (un-partitioned) version to CSV.
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cell.to_csv(save_path, index=False)
        console.print(
            f"[dim]Saved {len(cell)} rows x {len(cell.columns)} cols to[/dim] "
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
    stats: bool = False,
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
    console.rule(f"[bold]{p.name}[/bold]  [dim]({len(df)} rows)[/dim]", style="bright_black")
    console.print()

    if stats:
        save = Path(save_path) if save_path is not None \
            else p.with_name(f"{p.stem}_stats.csv")
        _view_stats(df, save)
    else:
        _view_default(df)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="View symbolic-discovery experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
output:
    default      one table per (variant, noise, noise_type, n_samples, seed)
    with --stats aggregate over seeds and datasets; saves <input>_stats.csv

examples:
    symbolic-discovery view results.csv
    symbolic-discovery view results.csv --stats
    symbolic-discovery view results.csv --stats --save-path my_stats.csv""",
    )
    parser.add_argument("csv_file", help="Path to results CSV")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Aggregate across seeds and datasets; saves <input>_stats.csv.",
    )
    parser.add_argument(
        "--save-path", default=None,
        help="With --stats, where to write per-cell stats CSV "
             "(default: <input>_stats.csv next to the input file).",
    )
    args = parser.parse_args(argv)
    view_results(args.csv_file, stats=args.stats, save_path=args.save_path)


if __name__ == "__main__":
    main()
    