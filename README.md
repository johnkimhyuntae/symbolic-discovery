# symbolic-discovery

University of Cambridge Computer Science Tripos Part II Project — Hyuntae (John) Kim.

This repo contains implementations of BACON-style symbolic discovery (notably BACON.3 and a BACON.7-style variant), plus a small experiment harness for running them across synthetic/textbook datasets.

## Quickstart

Create a local virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"
```

This installs the package in editable mode and exposes one console command:

- `symbolic-discovery` (CLI)

After activating the environment, you can run:

```bash
symbolic-discovery --help
```

### If your environment is confusing

If you have multiple virtualenvs, or `symbolic-discovery` isn’t found, make sure you:

1) activated the correct environment, and
2) installed this source checkout into it (`pip install -e .` for editable, or `pip install .` for a normal install).

If `pip` seems to be pointing at the wrong Python, you can force it with `python -m pip ...`.

If/when this project is published to a package index (e.g. PyPI), you could alternatively do `pip install symbolic-discovery` instead of installing from a local checkout.

Then you can always invoke the CLI via the module entrypoint:

```bash
symbolic_discovery --help
```

## Running Batch Experiments

The experiment runner lives at `symbolic_discovery.experiments.runner`. It runs one or more models across one or more datasets (catalogue IDs, benchmark selectors, or custom CSVs) and appends results to an output CSV.

If you installed the package (see Quickstart), use:

- `symbolic-discovery run ...`

### Basic usage

```bash
# Test BACON.3 on clean data
symbolic-discovery run --models bacon3 --datasets S-2 T-1 T-2 --noise 0.0 --output results.csv

# Test BACON.7 on noisy data
symbolic-discovery run --models bacon7 --datasets S-2 T-1 --noise 0.05 --output results.csv

# Test PySR (optional dependency)
symbolic-discovery run --models pysr --datasets S-2 T-1 --noise 0.0 --output results.csv

# Compare both algorithms
symbolic-discovery run --models bacon3 bacon7 --datasets S-2 T-1 T-2 T-3 --noise 0.0 0.05 --output comparison.csv
```

### Parameters

- `--models`: Which algorithms to run (`bacon3`, `bacon7`, `pysr`)
- `--datasets`: Dataset IDs from the catalogue (S-1, S-2, T-1, …), paths to `.csv` files, or benchmark selectors:
	- `feynman:dimless:<equation_id>` (e.g. `feynman:dimless:I.18.12`)
	- `feynman:dim:<equation_id>`
	- `bonus:dimless:<equation_id>` / `bonus:dim:<equation_id>`
	- `feynman:all` (dimensionless Feynman sweep)
	- `feynman:all_full` (Feynman+Bonus, dim+dimless)
	- `bonus:all` / `bonus:dim:all`
- `--noise`: Noise levels to test (e.g. `0.0 0.01 0.05 0.10`)
- `--seeds`: Random seeds for reproducibility (e.g. `42 43 44`)
- `--output`: Output CSV file path (default: `experiment_results.csv`)
- `--target`: Target column name (required for custom CSV files)
- `--verbose`: Enable verbose logs from BACON implementations

Feynman-specific options:

- `--feynman-root`: Root directory containing the Feynman CSV metadata and data folders (default: `feynman`)
- `--feynman-n-samples`: Number of rows to sample per equation file (default: `400`)
- `--feynman-target`: Target column name used when loading Feynman datasets (default: `y`)

### Examples

Test all “easy/solvable” catalogue datasets (clean):

```bash
symbolic-discovery run \
	--models bacon3 bacon7 \
	--datasets S-2 T-1 T-2 T-3 \
	--noise 0.0 \
	--output clean_results.csv
```

Noise robustness sweep:

```bash
symbolic-discovery run \
	--models bacon3 bacon7 \
	--datasets S-2 T-1 T-2 \
	--noise 0.0 0.01 0.05 0.10 \
	--seeds 42 43 44 \
	--output noise_test.csv
```

Custom CSV:

```bash
symbolic-discovery run \
	--models bacon3 bacon7 \
	--datasets my_data.csv \
	--target V \
	--noise 0.0 \
	--output custom_results.csv
```

Requirements for custom CSV files:

- Must have column headers (first row)
- Specify target variable with `--target <column_name>`
- Works best with multiplicative/power laws (e.g. `V = I×R`, `PV/T = constant`)
- Additive laws are partially supported:
	- `bacon3` supports simple `+`/`-` composites (e.g. `y = x1 + x2`)
	- `bacon7` is still primarily tuned for multiplicative/ratio-style invariants

Feynman dataset bundle:

Feynman/Bonus data is treated as a local (non-PyPI) asset. If you have the files under `./feynman/`, you can run a sweep over all available equation files:

```bash
symbolic-discovery run \
	--models bacon3 bacon7 \
	--datasets feynman:all \
	--noise 0.0 \
	--seeds 42 \
	--feynman-root feynman \
	--feynman-n-samples 400 \
	--feynman-target y \
	--output results/feynman_all_bacon3_bacon7_seed42_n400.csv
```

Full benchmark sweep (Feynman+Bonus, dim+dimless):

```bash
symbolic-discovery run \
	--models bacon3 bacon7 \
	--datasets feynman:all_full \
	--noise 0.0 \
	--seeds 42 \
	--feynman-root feynman \
	--feynman-n-samples 400 \
	--feynman-target y \
	--output results/feynman_all_full_bacon3_bacon7_seed42_n400.csv
```

### Output format

The output CSV has columns:

- `run_id`: Unique identifier (method_dataset_noise_seed)
- `dataset`: Dataset ID
- `method`: Algorithm name (`bacon3` / `bacon7`)
- `noise`: Noise level
- `seed`: Random seed
- `found_eq`: Discovered equation (or failure message)
- `found_eq_raw`: Backend raw equation string (useful for debugging/pretty-printing)
- `r2`: Reported $R^2$ metric (model-dependent)
- `time_s`: Runtime in seconds
- `status`: `Success` / `Failed` / `Error`

Note: rerunning `symbolic-discovery run` with the same `--output` appends to the file.

## Viewing Results

The results viewer lives at `symbolic_discovery.experiments.view_results`.

It prints results in a few different views:

```bash
# Summary view (default)
symbolic-discovery view results.csv

# Full details with equations
symbolic-discovery view results.csv --mode full

# Statistical summary
symbolic-discovery view results.csv --mode stats

# Side-by-side comparison
symbolic-discovery view results.csv --mode compare

# Failures/errors only (full equations)
symbolic-discovery view results.csv --mode failures

# Failures + low-R² “successes”
symbolic-discovery view results.csv --mode interesting
```

## Project Structure

```
symbolic_discovery/
├── _core/              # Private BACON implementations (bacon3.py, bacon7.py)
├── solvers/            # Public API: BaseSolver, SOLVER_REGISTRY, wrappers
├── data/               # Dataset loaders, catalogue, benchmarks
├── experiments/        # Experiment runner + results viewer
├── cli/                # Command-line interface
└── utils/              # Metrics, helpers
```

**Key modules:**

- `symbolic_discovery._core` — Core algorithm implementations (BACON.3, BACON.7)
- `symbolic_discovery.solvers` — Uniform solver interface + registry (bacon3, bacon7, pysr)
- `symbolic_discovery.data.catalogue` — Synthetic dataset catalogue (S-*, T-*)
- `symbolic_discovery.data.benchmarks` — Feynman/Bonus benchmark loaders
- `symbolic_discovery.data.feynman_exclusions` — Equation exclusion categories for fair benchmarking
- `symbolic_discovery.experiments.runner` — Batch experiment runner

## Optional installs / assets

This project supports optional engines and local benchmark assets which are not always installed by default.

### PySR (Julia-backed engine)

PySR is integrated as an optional solver backend (`--models pysr`). It is not installed by default.

Install PySR into your environment:

```bash
pip install pysr
```

Notes:

- PySR uses Julia via `juliapkg`/`juliacall`. The first run is often slow because it may:
	- download and install a Julia runtime (hundreds of MB), and
	- install/precompile Julia packages (notably `SymbolicRegression.jl`).
	Subsequent runs are typically much faster once Julia + packages are cached.
- By default, the `pysr` solver is configured to avoid writing persistent artifact directories (to match the BACON solvers). The experiment runner remains the canonical place where results are written (CSV).
- Some dataset column names collide with SymPy built-ins (e.g. `I`); the PySR wrapper sanitizes feature names during fitting and maps the final expression back.

### Feynman / Bonus datasets (local files)

The Feynman/Bonus benchmarks are treated as local assets. The `feynman:*` / `bonus:*` dataset selectors require that you have the benchmark files on disk.

Download/source: https://space.mit.edu/home/tegmark/aifeynman.html

Expected layout under `--feynman-root` (default: `./feynman`):

- `FeynmanEquationsDimensionless.csv`
- `FeynmanEquations.csv`
- `BonusEquationsDimensionless.csv`
- `BonusEquations.csv`
- `Feynman_without_units/` and/or `Feynman_with_units/`
- `bonus_without_units/` and/or `bonus_with_units/`

If you installed the code via `pip` from a source checkout, you typically already have this folder. If you installed from somewhere that doesn’t ship the data (e.g. a minimal package build), copy/download the `feynman/` folder separately and point the runner at it:

```bash
symbolic-discovery run \
	--models bacon3 bacon7 pysr \
	--datasets feynman:all \
	--noise 0.0 \
	--seeds 42 \
	--feynman-root /path/to/feynman \
	--feynman-n-samples 400 \
	--feynman-target y \
	--output results/feynman_sweep.csv
```

## Development

### Running tests

```bash
pytest                    # Run all tests
pytest -q                 # Quiet mode
pytest tests/test_bacon3.py -v  # Specific test file, verbose
```

### Code style

The codebase uses:
- Type hints throughout (validated by mypy)
- Black-compatible formatting
- Docstrings for public APIs

### Adding a new solver

1. Create a wrapper in `symbolic_discovery/solvers/` implementing `BaseSolver`
2. Register it in `symbolic_discovery/solvers/registry.py`
3. Add tests in `tests/`

See `solvers/pysr.py` for an example of wrapping an external library.

## License

Part II Project — University of Cambridge. See dissertation for full details.
