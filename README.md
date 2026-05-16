# symbolic-discovery

A pip-installable Python framework for symbolic regression research, built around BACON-style discovery algorithms. Implements **BACON.3F** and **BACON.7F** (flat-pool adaptations of Langley and Miller's originals) alongside a **PySR** integration, with a unified solver interface, curated datasets, and a YAML-driven experiment runner for reproducible benchmarking.

## Quickstart

```bash
pip install symbolic-discovery
```

This exposes the `symbolic-discovery` CLI:

```bash
symbolic-discovery run --help
symbolic-discovery view --help
```

Try it on the built-in datasets immediately:

```bash
symbolic-discovery run --models bacon3f --datasets S T --output results.csv
symbolic-discovery view results.csv
```

## Running Experiments

The runner expands a Cartesian product over models × datasets × noise levels × noise types × sample sizes × seeds, executes each cell, and appends one row per run to an output CSV.

```bash
# BACON.3F on clean synthetic + textbook data
symbolic-discovery run --models bacon3f --datasets S T --output results.csv

# Compare BACON.3F and BACON.7F under noise
symbolic-discovery run --models bacon3f bacon7f --datasets S2 T1 T2 \
    --noise 0.0 0.01 0.05 --seeds 73 74 75 --output noise_sweep.csv

# A named ablation, defined inline
symbolic-discovery run --datasets T F --noise 0.0 0.05 \
    --variant baseline=bacon7f \
    --variant no_folds=bacon7f:n_folds=1 \
    --variant no_relax=bacon7f:scale_factor=1.0

# A one-knob hyperparameter sweep
symbolic-discovery run --datasets S T --sweep bacon7f.n_folds=1,3,5,7

# A full preregistered study from a YAML file
symbolic-discovery run --study studies/validation.yaml --output validation.csv

# PySR (optional dependency)
symbolic-discovery run --models pysr --datasets S2 T1 --output pysr_results.csv

# Your own CSV
symbolic-discovery run --models bacon3f --datasets my_data.csv --target V \
    --output custom.csv
```

CLI flags override study fields per axis, so it is fine to point at a heavy YAML study and override `--seeds` for a quick re-run.

### Dataset Selectors

| Selector | Description |
|---|---|
| `S1`..`S3` | Individual synthetic datasets |
| `T1`..`T5` | Individual textbook laws (Ohm, Hooke, freefall, ideal gas, Stefan–Boltzmann) |
| `F1`..`F100` | Individual Feynman equations (by number) |
| `B1`..`B20` | Individual Bonus equations (by number) |
| `S` / `T` / `F` / `B` | All datasets in that family |
| `*.csv` | Custom CSV file (requires `--target`) |

### CLI Parameters

| Flag | Description | Default |
|---|---|---|
| `--models` | Solvers to run (`bacon3f`, `bacon7f`, `pysr`). Repeatable. | — |
| `--variant` | Named variant of the form `name=model[:k=v,...]`. Repeatable. | — |
| `--sweep` | One-knob sweep `model.param=v1,v2,...`. Repeatable. | — |
| `--study` | Path to a YAML study file. CLI flags override study fields per axis. | — |
| `--datasets` | Dataset selectors (see above). Required unless given by `--study`. | — |
| `--target` | Target column name (required for custom CSVs). | — |
| `--noise` | Noise levels to inject. | `0.0` |
| `--noise-types` | Noise distributions (`multiplicative`, `additive`). | `multiplicative` |
| `--n-samples` | Rows per dataset. | `1000` |
| `--seeds` | Random seeds. | `73` |
| `--log-level` | `default`, `verbose`, or `quiet`. | `default` |
| `--output-root` | Root directory for output CSVs. | `results` |
| `--output` | Output CSV path (relative to `--output-root`). | `experiment_results.csv` |
| `--exclude` | Skip datasets that use operators beyond `+ - * /`. | off |
| `--exclude-bacon` | Skip datasets known to be unsolvable by BACON. | off |
| `--feynman-root` | Root directory for Feynman/Bonus data files. | `feynman` |

One model must be specified — via `--models`, `--variant`, `--sweep`, or `--study`.

### YAML Studies

For multi-axis experiments, declare the full grid in a YAML file and pass it with `--study <path>`. The runner expands it into the same `list[Run]` that the direct CLI flags would produce.

```yaml
# studies/ablation_bacon7f.yaml
variants:
  - {name: baseline, model: bacon7f}
  - {name: no_relax, model: bacon7f, params: {scale_factor: 1.0}}
  - {name: no_folds, model: bacon7f, params: {n_folds: 1}}
datasets: [T, F11, F15, F34, F41, F58, F60, F64, F83, F85, F96]
noise:    [0.0, 0.025]
seeds:    [73, 74, 75, 76, 77, 78, 79, 80, 81, 82]
```

```bash
symbolic-discovery run --study studies/ablation_bacon7f.yaml --output ablation.csv
```

Top-level keys are `variants`, `datasets`, `noise`, `noise_types`, `n_samples`, and `seeds` (all optional, all falling back to their CLI defaults when omitted). CLI flags override study fields per axis, useful for one-knob variations on a heavy study.

## Viewing Results

The viewer renders experiment CSVs as rich tables in the terminal:

```bash
symbolic-discovery view results.csv                 # Concise summary (default)
symbolic-discovery view results.csv --mode full     # All columns
symbolic-discovery view results.csv --mode compare  # Side-by-side per solver
symbolic-discovery view results.csv --stats         # Aggregated statistics
```

## Project Structure

```
symbolic-discovery/
├── pyproject.toml
├── README.md
├── LICENSE
├── requirements.txt
├── symbolic_discovery/      # The pip-installable package
│   ├── algorithms/          # BACON.3F and BACON.7F implementations
│   │   ├── bacon3f.py
│   │   └── bacon7f.py
│   ├── solvers/             # Uniform solver interface
│   │   ├── base.py          # BaseSolver ABC + SolverResult
│   │   ├── registry.py      # SOLVER_REGISTRY dictionary
│   │   ├── bacon3f.py       # BACON3FSolver wrapper
│   │   ├── bacon7f.py       # BACON7FSolver wrapper
│   │   └── pysr.py          # PySRSolver wrapper
│   ├── data/                # Unified data layer
│   │   ├── api.py           # load, resolve, expand_datasets, ...
│   │   ├── config.py        # DatasetConfig dataclass
│   │   ├── synthetic.py     # generate() + built-in S/T catalogue
│   │   ├── feynman.py       # Feynman/Bonus metadata + file I/O
│   │   ├── feynman_exclusions.json
│   │   └── custom.py        # Custom CSV loader
│   ├── experiments/         # Orchestration
│   │   ├── plan.py          # Variant/Run dataclasses; CLI + YAML parsers
│   │   ├── runner.py        # Executes runs; writes output CSV
│   │   └── viewer.py        # Rich-based results viewer
│   ├── cli/                 # CLI entry point
│   │   └── main.py          # Dispatches run / view subcommands
│   └── utils/               # Helpers
│       ├── metrics.py       # R², MSE, MAE, r
│       └── analysis.py      # Post-hoc aggregation helpers
├── tests/                   # Unit, integration, and end-to-end tests
├── studies/                 # YAML studies used in the dissertation
└── results/                 # Output CSVs from studies
```

### Key Modules

- `symbolic_discovery.algorithms` — BACON.3F and BACON.7F discovery loops
- `symbolic_discovery.solvers` — `BaseSolver` / `SolverResult` contract + `SOLVER_REGISTRY`
- `symbolic_discovery.data` — Built-in S/T catalogue, Feynman/Bonus loaders, synthetic generation, noise injection, declarative exclusion list
- `symbolic_discovery.experiments.plan` — Parses CLI flags and YAML studies into `list[Run]`
- `symbolic_discovery.experiments.runner` — Executes runs, writes CSV
- `symbolic_discovery.experiments.viewer` — Rich tables (concise / full / compare / stats)

## Optional Dependencies

### PySR (Julia-backed engine)

```bash
pip install symbolic-discovery[pysr]
```

PySR is integrated as an optional solver backend (`--models pysr`). It pulls Julia via `juliapkg` / `juliacall`; the first run downloads a Julia runtime and precompiles packages, subsequent runs are fast.

### Feynman / Bonus Datasets

The Feynman/Bonus benchmarks are local assets, not bundled with `pip install`. Download from <https://space.mit.edu/home/tegmark/aifeynman.html>.

Expected layout under `--feynman-root` (default: `./feynman`):

```
feynman/
├── FeynmanEquations.csv
├── BonusEquations.csv
├── Feynman_with_units/
│   ├── I.6.2a
│   ├── I.6.2b
│   └── ...
└── bonus_with_units/
    └── ...
```

The package bundles `feynman_exclusions.json`, a curated list of equations outside BACON's algorithmic ceiling (transcendentals, square roots, complex compositions) used for fair benchmarking.

## Development

### Setup from source

```bash
git clone REDACTED (for anonymity)
cd symbolic-discovery

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e ".[dev,pysr]"
```

The module entrypoint also works: `python -m symbolic_discovery --help`.

### Tests

```bash
pytest                            # All 330 tests
pytest --cov=symbolic_discovery   # With coverage (≈88%)
pytest tests/test_bacon3f.py -v   # One file, verbose
```

Feynman-tier tests are skipped when the database is absent, so the suite stays green on a clean clone.

### Adding a new solver

1. Implement `BaseSolver` in `symbolic_discovery/solvers/your_solver.py`.
2. Add one line to `symbolic_discovery/solvers/registry.py`.
3. Add tests in `tests/`.

`solvers/pysr.py` is the canonical example of wrapping an external library; `solvers/bacon3f.py` shows the minimal in-tree case.

### Adding a new dataset

Add a `DatasetConfig` entry to `symbolic_discovery/data/synthetic.py` (for S/T-tier datasets). It immediately integrates with the CLI, runner, and viewer.

### Code style

Type hints throughout, docstrings on every public class and function, three-tier `pytest` suite (unit / integration / end-to-end).

## Programmatic Usage

```python
import pandas as pd
from symbolic_discovery.algorithms import BACON3F

df = pd.DataFrame({
    "I": [0.5, 1.0, 1.5, 2.0],
    "R": [10, 10, 10, 10],
    "V": [5.0, 10.0, 15.0, 20.0],
})

solver = BACON3F(max_depth=3)
equation, _ = solver.discover(df, target_col="V", seed=73)
print(equation)        # V = I * R
```

## License

Apache 2.0 - see [LICENSE](LICENSE)