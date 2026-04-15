#!/usr/bin/env python3
"""
Results viewer for symbolic-discovery experiment output.

Displays per-model, per-condition tables from a results CSV.
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# Helpers

def _equation_col(df: pd.DataFrame) -> str:
    """Return whichever equation column the CSV actually has."""
    for c in ("equation", "best_equation"):
        if c in df.columns:
            return c
    return ""


def _evaluation(status: str, r2: str) -> Text:
    if status == "Found":
        if float(r2) > 0.999:
            return Text("Perfect", style="blue")
        elif float(r2) > 0.900:
            return Text("Good", style="green")
        elif float(r2) > 0.500:
            return Text("Poor", style="yellow")
        else:
            return Text("Bad", style="red")
    return Text("Fail", style="bold red")


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


def _truncate(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_time(val) -> str:
    f = _fmt(val, ".3f")
    return f"{f}s" if f != "—" else "—"


def _noise_seed_title(noise, seed, has_seed: bool) -> str:
    """Format a (noise, seed) pair into a subtitle string."""
    s = f"noise={_fmt(noise, '.2f')}"
    if has_seed:
        seed_s = str(int(seed)) if pd.notna(seed) else "?"
        s += f"  seed={seed_s}"
    return s


# Stats footer

def _stats_table(group: pd.DataFrame) -> Table:
    """One-row stats table shown beneath a results table."""
    total = len(group)
    total_found = int((group["status"] == "Found").sum())
    failures = total - total_found
    ok = group[group["status"] == "Found"]
    has_r2 = "r2" in group.columns
    has_time = "time_s" in group.columns

    r2_vals = pd.to_numeric(ok["r2"], errors="coerce").dropna() if has_r2 and not ok.empty else pd.Series(dtype=float)
    time_vals = pd.to_numeric(group["time_s"], errors="coerce").dropna() if has_time else pd.Series(dtype=float)

    t = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    t.add_column("Runs", justify="right")
    t.add_column("Found", justify="right")
    t.add_column("Perfect", justify="right")
    t.add_column("Good", justify="right")
    t.add_column("Poor", justify="right")
    t.add_column("Bad", justify="right")
    t.add_column("Failed", justify="right")
    t.add_column("Fail Rate", justify="right")
    if has_r2:
        t.add_column("Mean R²", justify="right")
        t.add_column("Min R²", justify="right")
    if has_time:
        t.add_column("Mean time", justify="right")
        t.add_column("Total time", justify="right")

    row: list[str | Text] = [
        str(total),
        Text(str(total_found)),
        Text(str((r2_vals > 0.999).sum()), style="blue" if (r2_vals > 0.999).sum() else "dim"),
        Text(str(((0.999 >= r2_vals) & (r2_vals > 0.900)).sum()), style="green" if ((0.999 >= r2_vals) & (r2_vals > 0.900)).sum() else "dim"),
        Text(str(((0.900 >= r2_vals) & (r2_vals > 0.500)).sum()), style="yellow" if ((0.900 >= r2_vals) & (r2_vals > 0.500)).sum() else "dim"),
        Text(str((r2_vals <= 0.500).sum()), style="red" if (r2_vals <= 0.500).sum() else "dim"),
        Text(str(failures), style="red" if failures else "dim"),
        f"{failures / total * 100:.1f}%" if total else "—",
    ]
    if has_r2:
        row.append(_fmt(r2_vals.mean()) if not r2_vals.empty else "—")
        row.append(_fmt(r2_vals.min()) if not r2_vals.empty else "—")
    if has_time:
        row.append(f"{time_vals.mean():.3f}s" if not time_vals.empty else "—")
        row.append(f"{time_vals.sum():.2f}s" if not time_vals.empty else "—")

    t.add_row(*row)
    return t


# Group key helpers

def _model_group_cols(df: pd.DataFrame) -> list[str]:
    """Group columns for concise/full: (method, noise[, seed])."""
    cols = ["method", "noise"]
    if "seed" in df.columns:
        cols.append("seed")
    return cols


def _compare_group_cols(df: pd.DataFrame) -> list[str]:
    """Group columns for compare: (noise[, seed])."""
    cols = ["noise"]
    if "seed" in df.columns:
        cols.append("seed")
    return cols


# Concise mode

def _view_concise(df: pd.DataFrame) -> None:
    eq_col = _equation_col(df)
    has_seed = "seed" in df.columns
    group_cols = _model_group_cols(df)

    for group_key, group in df.groupby(group_cols, sort=True):
        keys: tuple[Any, ...] = group_key if isinstance(group_key, tuple) else (group_key,)
        method, noise = keys[0], keys[1]
        seed = keys[2] if has_seed else None

        subtitle = _noise_seed_title(noise, seed, has_seed)
        group = group.sort_values("dataset") # TBD: better sorting

        t = Table(
            title=f"  {method}  [dim]{subtitle}[/dim]",
            title_style="bold cyan",
            box=box.ROUNDED,
            show_lines=False,
            padding=(0, 1),
        )

        t.add_column("Dataset", style="bold")
        t.add_column("Grade", justify="center")
        if eq_col:
            t.add_column("Equation")
        if "r2" in group.columns:
            t.add_column("R²", justify="right")
        if "time_s" in group.columns:
            t.add_column("Time", justify="right", style="dim")

        for _, row in group.iterrows():
            cells: list[str | Text] = [str(row["dataset"])]
            if "r2" in group.columns:
                cells.append(_evaluation(row["status"], _fmt(row["r2"])))
            if eq_col:
                cells.append(_truncate(row.get(eq_col, ""), 52))
            if "r2" in group.columns:
                cells.append(_fmt(row["r2"]))
            if "time_s" in group.columns:
                cells.append(_fmt_time(row["time_s"]))
            t.add_row(*cells)

        console.print(t)
        console.print(_stats_table(group))
        console.print()


# Full mode

def _view_full(df: pd.DataFrame) -> None:
    eq_col = _equation_col(df)
    has_seed = "seed" in df.columns
    group_cols = _model_group_cols(df)

    # Columns to show per row (method, noise, seed are in the title)
    show_order = [
        "run_id", "dataset", "grade",
        eq_col, "raw_equation",
        "r2", "mse", "mae", "time_s",
    ]
    show_order = [c for c in show_order if c and (c in df.columns or c == "grade")]

    wide = Console(width=max(console.width, 160))

    for group_key, group in df.groupby(group_cols, sort=True):
        keys: tuple[Any, ...] = group_key if isinstance(group_key, tuple) else (group_key,)
        method, noise = keys[0], keys[1]
        seed = keys[2] if has_seed else None

        subtitle = _noise_seed_title(noise, seed, has_seed)
        group = group.sort_values("dataset")

        t = Table(
            title=f"  {method}  [dim]{subtitle}[/dim]",
            title_style="bold cyan",
            box=box.ROUNDED,
            show_lines=True,
            padding=(0, 1),
        )

        for col in show_order:
            justify = "right" if col in ("r2", "mse", "mae", "time_s") else "left"
            nowrap = col not in (eq_col, "raw_equation")
            t.add_column(col, justify=justify, no_wrap=nowrap)

        for _, row in group.iterrows():
            cells: list[str | Text] = []
            for col in show_order:
                val = row.get(col, "")
                if col == "grade":
                    cells.append(_evaluation(row["status"], _fmt(row["r2"])))
                elif col in ("r2", "mse", "mae"):
                    cells.append(_fmt(val))
                elif col == "time_s":
                    cells.append(_fmt_time(val))
                else:
                    cells.append(str(val) if pd.notna(val) else "—")
            t.add_row(*cells)

        wide.print(t)
        wide.print(_stats_table(group))
        wide.print()


# Compare mode

def _view_compare(df: pd.DataFrame) -> None:
    eq_col = _equation_col(df)
    methods = sorted(df["method"].unique())
    has_r2 = "r2" in df.columns
    has_time = "time_s" in df.columns
    has_seed = "seed" in df.columns
    group_cols = _compare_group_cols(df)

    wide = Console(width=max(console.width, 160))

    # One table per (noise, seed)

    for group_key, cond_group in df.groupby(group_cols, sort=True):
        keys: tuple[Any, ...] = group_key if isinstance(group_key, tuple) else (group_key,)
        noise = keys[0]
        seed = keys[1] if has_seed else None

        subtitle = _noise_seed_title(noise, seed, has_seed)
        datasets = sorted(cond_group["dataset"].unique())
        indexed = cond_group.set_index(["dataset", "method"])

        t = Table(
            title=f"  {subtitle}",
            title_style="bold cyan",
            box=box.ROUNDED,
            show_lines=False,
            padding=(0, 1),
        )

        t.add_column("Dataset", style="bold")

        for i, m in enumerate(methods):
            style = "dim" if i % 2 else ""
            t.add_column(m, justify="center", style=style)
            if eq_col:
                t.add_column("Eq", style=style)
            if has_r2:
                t.add_column("R²", justify="right", style=style)
            if has_time:
                t.add_column("Time", justify="right", style=style)

        for ds in datasets:
            cells: list[Any] = [ds]
            for m in methods:
                try:
                    # r is row
                    r: pd.Series = indexed.loc[(ds, m)] # type: ignore[assignment]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    if has_r2:
                        cells.append(_evaluation(r["status"], _fmt(r["r2"])))
                    if eq_col:
                        cells.append(_truncate(str(r.get(eq_col, "")), 40))
                    if has_r2:
                        cells.append(_fmt(r["r2"]))
                    if has_time:
                        cells.append(_fmt_time(r["time_s"]))
                except KeyError:
                    cells.append(Text("—", style="dim"))
                    if eq_col:
                        cells.append("—")
                    if has_r2:
                        cells.append("—")
                    if has_time:
                        cells.append("—")
            t.add_row(*cells)

        wide.print(t)
        wide.print()

    # Summary table (per method, across all data)

    st = Table(
        title="  Summary",
        title_style="bold cyan",
        box=box.ROUNDED,
        show_lines=False,
        padding=(0, 1),
    )

    st.add_column("Method", style="bold")
    st.add_column("Runs", justify="right")
    st.add_column("Found", justify="right")
    st.add_column("Perfect", justify="right")
    st.add_column("Good", justify="right")
    st.add_column("Poor", justify="right")
    st.add_column("Bad", justify="right")
    st.add_column("Failed", justify="right")
    st.add_column("Fail Rate", justify="right")
    if has_r2:
        st.add_column("Mean R²", justify="right")
        st.add_column("Min R²", justify="right")
    if has_time:
        st.add_column("Mean time", justify="right")
        st.add_column("Total time", justify="right")

    for m in methods:
        mg = df[df["method"] == m]
        total = len(mg)
        total_found = int((mg["status"] == "Found").sum())
        failures = len(mg) - total_found
        ok = mg[mg["status"] == "Found"]

        r2_vals = pd.to_numeric(ok["r2"], errors="coerce").dropna() if has_r2 and not ok.empty else pd.Series(dtype=float)
        time_vals = pd.to_numeric(mg["time_s"], errors="coerce").dropna() if has_time else pd.Series(dtype=float)

        row: list[str | Text] = [
            m,
            str(total),
            Text(str(total_found)),
            Text(str((r2_vals > 0.999).sum()), style="blue" if (r2_vals > 0.999).sum() else "dim"),
            Text(str(((0.999 >= r2_vals) & (r2_vals > 0.900)).sum()), style="green" if ((0.999 >= r2_vals) & (r2_vals > 0.900)).sum() else "dim"),
            Text(str(((0.900 >= r2_vals) & (r2_vals > 0.500)).sum()), style="yellow" if ((0.900 >= r2_vals) & (r2_vals > 0.500)).sum() else "dim"),
            Text(str((r2_vals <= 0.500).sum()), style="red" if (r2_vals <= 0.500).sum() else "dim"),
            Text(str(failures), style="red" if failures else "dim"),
            f"{failures / total * 100:.1f}%" if total else "—",
        ]
        if has_r2:
            row.append(_fmt(r2_vals.mean()) if not r2_vals.empty else "—")
            row.append(_fmt(r2_vals.min()) if not r2_vals.empty else "—")
        if has_time:
            row.append(f"{time_vals.mean():.3f}s" if not time_vals.empty else "—")
            row.append(f"{time_vals.sum():.2f}s" if not time_vals.empty else "—")

        st.add_row(*row)

    wide.print(st)
    wide.print()


# Entry point

def view_results(csv_path: str, mode: str = "concise") -> None:
    p = Path(csv_path)
    if not p.exists():
        console.print(f"[red]File not found:[/red] {csv_path}")
        return

    df = pd.read_csv(p)

    if "method" not in df.columns or "status" not in df.columns:
        console.print("[red]CSV must contain at least 'method' and 'status' columns.[/red]")
        return

    console.print()
    console.rule(f"[bold]{p.name}", style="bright_black")
    console.print()

    if mode == "concise":
        _view_concise(df)
    elif mode == "full":
        _view_full(df)
    elif mode == "compare":
        _view_compare(df)
    else:
        console.print(f"[red]Unknown mode:[/red] {mode}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="View symbolic-discovery experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
    symbolic-discovery view results.csv
    symbolic-discovery view results.csv --mode full
    symbolic-discovery view results.csv --mode compare""",
    )
    parser.add_argument("csv_file", help="Path to results CSV")
    parser.add_argument(
        "--mode", "-m",
        choices=["concise", "full", "compare"],
        default="concise",
        help="Display mode (default: concise)",
    )
    args = parser.parse_args(argv)
    view_results(args.csv_file, args.mode)


if __name__ == "__main__":
    main()