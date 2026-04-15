# symbolic-discovery

University of Cambridge Computer Science Tripos Part II Project — Hyuntae Kim.

A pip-installable Python framework for symbolic regression research, built around BACON-style discovery algorithms. Implements BACON.3F and BACON.7F (flat-pool adaptations of Langley and Miller's originals) alongside a PySR integration, with a unified solver interface, curated datasets, and an experiment runner for reproducible benchmarking.

## Quickstart

```bash
pip install symbolic-discovery
```

This exposes the `symbolic-discovery` CLI:

```bash
symbolic-discovery --help
```

Try it on the built-in datasets immediately:

```bash
symbolic-discovery run --models bacon3f --datasets S T --noise 0.0 --output results.csv
symbolic-discovery view results.csv
```

## Running Experiments

The experiment runner iterates over the Cartesian product of models, datasets, noise levels, and seeds, writing results to a CSV in append mode.

```bash
# BACON.3F on clean synthetic + textbook data
symbolic-discovery run --models bacon3f --datasets S T --noise 0.0 --output results.csv

# Compare BACON.3F and BACON.7F under noise
symbolic-discovery run --models bacon3f bacon7f --datasets S2 T1 T2 --noise 0.0 0.01 0.05 --seeds 42 43 44 --output noise_sweep.csv

# BACON.3F on specific Feynman equations
symbolic-discovery run --models bacon3f --datasets F1 F8 F12 --seeds 42 --output feynman_results.csv

# Full Feynman sweep
symbolic-discovery run --models bacon3f bacon7f --datasets F --noise 0.0 --output feynman_full.csv

# PySR (optional dependency)
symbolic-discovery run --models pysr --datasets S2 T1 --noise 0.0 --output pysr_results.csv

# Custom CSV file
symbolic-discovery run --models bacon3f --datasets my_data.csv --target V --noise 0.0 --output custom.csv
```

### Dataset Selectors

| Selector | Description |
|---|---|
| `S1`..`S4` | Individual synthetic datasets |
| `T1`..`T5` | Individual textbook laws (Ohm, Hooke, freefall, ideal gas, Stefan-Boltzmann) |
| `F1`..`F100` | Individual Feynman equations (by number) |
| `B1`..`B20` | Individual Bonus equations (by number) |
| `S` / `T` / `F` / `B` | All datasets in a family |
| `*.csv` | Custom CSV file (requires `--target`) |

### CLI Parameters

| Flag | Description | Default |
|---|---|---|
| `--models` | Solvers to run (`bacon3f`, `bacon7f`, `pysr`) | *(required)* |
| `--datasets` | Dataset selectors (see above) | *(required)* |
| `--target` | Target column name (required for custom CSVs) | — |
| `--noise` | Noise levels to inject | `0.0` |
| `--seeds` | Random seeds for reproducibility | `42` |
| `--output` | Output CSV path | `experiment_results.csv` |
| `--verbose` | Verbose solver logs | off |
| `--n-samples` | Rows per dataset | `1000` |
| `--feynman-root` | Root directory for Feynman/Bonus data files | `feynman` |

## Viewing Results

The results viewer renders experiment CSVs using rich tables:

```bash
# Concise summary (default)
symbolic-discovery view results.csv

# Full per-row details
symbolic-discovery view results.csv --mode full

# Side-by-side solver comparison
symbolic-discovery view results.csv --mode compare
```

## Project Structure

```
symbolic_discovery/
├── algorithms/         # Core algorithm implementations (BACON3F, BACON7F)
├── solvers/            # Public API: BaseSolver, SolverResult, SOLVER_REGISTRY, wrappers
├── data/               # Unified dataset interface
│   ├── api.py          # expand_datasets, resolve, load, get_exclusion_reason, pretty_equation
│   ├── catalogue.py    # DatasetConfig + built-in S/T datasets
│   ├── feynman.py      # Feynman/Bonus metadata, data file I/O, exclusions
│   ├── feynman_exclusions.json
│   ├── synthetic.py    # generate() + inject_noise()
│   └── custom.py       # Custom CSV loader
├── experiments/        # Experiment runner + results viewer
│   ├── runner.py
│   └── viewer.py
├── cli/                # CLI entry point
│   └── main.py
└── utils/              # Metrics (R², MSE, MAE, r)
    └── metrics.py
```

### Key Modules

- `symbolic_discovery.algorithms` — BACON.3F and BACON.7F implementations
- `symbolic_discovery.solvers` — Uniform solver interface (`BaseSolver`, `SolverResult`) + registry (`SOLVER_REGISTRY`)
- `symbolic_discovery.data` — Unified data API: catalogue (S1–S4, T1–T5), Feynman/Bonus benchmark loaders, synthetic generation, noise injection, exclusion system
- `symbolic_discovery.experiments.runner` — Batch experiment runner (Cartesian product over models × datasets × noise × seeds)
- `symbolic_discovery.experiments.viewer` — Rich-based results viewer (concise/full/compare modes)

## Optional Dependencies

### PySR (Julia-backed engine)

PySR is integrated as an optional solver backend (`--models pysr`). It is not installed by default.

```bash
pip install pysr
```

PySR uses Julia via `juliapkg`/`juliacall`. The first run may download a Julia runtime and precompile Julia packages; subsequent runs are faster.

### Feynman / Bonus Datasets

The Feynman/Bonus benchmarks are local assets, not bundled with `pip install`. Download from https://space.mit.edu/home/tegmark/aifeynman.html.

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

The framework includes a `feynman_exclusions.json` file that categorises equations outside BACON's algorithmic ceiling (transcendental functions, square roots, complex expressions) for fair benchmarking.

## Development

### Setup from Source

```bash
git clone https://github.com/johnkimhyuntae/symbolic-discovery.git
cd symbolic-discovery

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```

This installs the package in editable mode with test dependencies. You can also invoke via the module entrypoint: `python -m symbolic_discovery --help`.

### Running Tests

```bash
pytest                           # All tests
pytest -q                        # Quiet mode
pytest tests/test_bacon3f.py -v  # Specific file, verbose
```

### Adding a New Solver

1. Create a wrapper in `symbolic_discovery/solvers/` implementing `BaseSolver`
2. Register it in `symbolic_discovery/solvers/registry.py`
3. Add tests in `tests/`

See `solvers/pysr.py` for an example of wrapping an external library, or `solvers/bacon3f.py` for a minimal wrapper around an internal algorithm.

### Adding a New Dataset

Add a `DatasetConfig` entry to `symbolic_discovery/data/catalogue.py`. It immediately integrates with the CLI, runner, and viewer.

### Code Style

The codebase uses type hints throughout and docstrings for public APIs.

## Programmatic Usage

```python
from symbolic_discovery.algorithms import BACON3F
import pandas as pd

df = pd.DataFrame({
    "I": [0.5, 1.0, 1.5, 2.0],
    "R": [10, 10, 10, 10],
    "V": [5.0, 10.0, 15.0, 20.0],
})

solver = BACON3F(max_depth=3, verbose=True)
equation, _ = solver.discover(df, target_col="V", seed=42)
print(equation)        # V = I*R
```
