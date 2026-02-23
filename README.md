# symbolic-discovery

University of Cambridge Computer Science Tripos Part II Project — Hyuntae (John) Kim.

This repo contains implementations of BACON-style symbolic discovery (notably BACON.3 and a BACON.7-style variant), plus a small experiment harness for running them across synthetic/textbook datasets.

## Quickstart

Create a local virtual environment and install dependencies:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -e ".[dev]"
```

This installs the package in editable mode and exposes one console command:

- `symbolic-discovery` (umbrella CLI)

You can also run the umbrella CLI via Python:

```bash
./.venv/bin/python -m symbolic_discovery --help
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

# Compare both algorithms
symbolic-discovery run --models bacon3 bacon7 --datasets S-2 T-1 T-2 T-3 --noise 0.0 0.05 --output comparison.csv
```

### Parameters

- `--models`: Which algorithms to run (`bacon3`, `bacon7`)
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
- May fail on additive relationships (e.g. `y = x + z`) due to limitations of the search space

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

## Where things live now

- Solvers: `symbolic_discovery.algorithms` (BACON.3 / BACON.7)
- Synthetic dataset catalogue: `symbolic_discovery.data.catalogue` (S-* / T-*)
- Synthetic generator: `symbolic_discovery.data.synthetic`
- Feynman/Bonus loaders: `symbolic_discovery.data.benchmarks`
- Experiment runner + viewer: `symbolic_discovery.experiments`

## Development

```bash
./.venv/bin/python -m pytest -q
```
