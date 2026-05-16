from typing import Tuple
import numpy as np
import pandas as pd
from itertools import permutations
from sympy import Symbol, Expr, symbols
from scipy.stats import iqr
import sympy
from dataclasses import dataclass
from ..utils import calculate_r, calculate_r2, calculate_mse, calculate_mae


@dataclass
class Term:
    """
    A symbolic expression paired with its evaluated numerical data.

    The symbol tracks the algebraic form (e.g. V/P) while values holds
    the corresponding numeric array computed from the dataset. This
    separation lets the discovery loop manipulate symbolic expressions
    and numeric data in lockstep without re-evaluating expressions
    from scratch.
    """
    symbol: Expr
    values: np.ndarray


class BACON3F:
    """
    BACON.3F: a flat tabular adaptation of Langley's BACON.3.

    Discovers empirical laws by iterating directed pairwise heuristic
    checks over a pool of symbolic terms. New composites are promoted 
    each layer and laws are collected whenever a composite is found to 
    be constant and contains the target variable.

    - IQR-based linearity: slope constancy is assessed via the 
    interquartile range of finite differences.

    - Early stopping: discovery halts as soon as a law exceeding the 
    R² threshold is found, avoiding unnecessary combinatorial 
    exploration at deeper layers.

    - Deduplication: a tried-permutations set prevents re-checking 
    the same directed pair across layers, and a known-expressions 
    set prevents proposing duplicate composites.
    """
    def __init__(self, 
                 max_depth: int = 6,
                 constancy_threshold: float = 0.1,
                 r_threshold: float = 0.45,
                 r2_threshold: float = 0.9,
                 log_level: str = "default"):
        """
        Initialise the BACON.3F solver.

        Args:
            max_depth: Maximum number of discovery layers before stopping.
                Each layer tests all novel directed pairs and promotes
                non-constant composites.
            constancy_threshold: Unified threshold for all constancy
                checks, used for variable constancy, slope constancy, 
                and intercept negligibility.
            r_threshold: Minimum absolute Pearson correlation coefficient
                for a pair to be considered correlated.
            r2_threshold: Minimum predictive R² for early stopping. When
                a discovered law meets or exceeds this threshold, the
                search halts immediately.
            log_level: Controls the verbosity of the decision log. Options are
                "default", "verbose", and "quiet".
        """
        self.constancy_threshold = constancy_threshold
        self.max_depth = max_depth
        self.r_threshold = r_threshold
        self.r2_threshold = r2_threshold
        self.log_level = log_level
        
        # Internal states
        self.logs: list[str] = []
        self.variable_pool: list[Term] = []
        self.target_var: Symbol | None = None
        self.target_values: np.ndarray | None = None
        self.discovered_laws: list[Tuple[str, dict[str, float]]] = []
        # Dedup: avoid duplicate laws (by expression string)
        self.discovered_strs: set[str] = set()
        # Dedup: avoid re-checking the same directed pair across layers
        self.tried_permutations: set[Tuple[str, str]] = set()
        # Dedup: avoid proposing composites whose expression string
        # already exists in the pool (original variables + promoted)
        self.known_expressions: set[str] = set()
        # Maps original variable symbols to their data arrays, used by
        # _get_diagnostics to evaluate rearranged laws on original data
        self.sym_to_vals: dict[str, np.ndarray] = {}


    def _log(self, message: str, level: str = "default"):
        """Append to the decision log / print depending on log_level.

            quiet:      nothing logged or printed.
            default:    only level="default" calls are emitted.
            verbose:    all calls (default + verbose) are emitted.
        """
        if self.log_level == "quiet":
            return
        if level == "verbose" and self.log_level != "verbose":
            return

        self.logs.append(message)
        print(f"[BACON.3F] {message}")
        return

    
    def _check(self, dependent: Term, independent: Term) -> Tuple[Term | None, str]:
        """
        Applies four checks in this order:

        1. **Constancy**: is the dependent already constant within 
            constancy_threshold?

        2. **Linearity**: do dependent and independent have a near 
            perfect linear relationship (determined by slope constancy)?

        3. **Uncorrelatedness**: if the Pearson correlation coefficient 
           |r| < self.r_threshold, the variables are considered uncorrelated, 
           and no meaningful relationship is proposed.

        4. **Monotonic trend**: if non-linear and correlated, the Pearson 
           correlation sign determines whether to propose a ratio 
           (r > 0, co-varying) or product (r < 0, inversely varying).

        Args:
            dependent: The term treated as y.
            independent: The term treated as x.

        Returns:
            A (Term, relation_type) pair where relation_type is one of
            "Constant", "Linear", "Ratio", "Product" or (None, "Null")
            if no relation was found or the proposed expression is a
            known duplicate.
        """

        X = independent.values
        Y = dependent.values

        # Guard against division by zero in ratio operations
        safe_X = np.where(np.abs(X) < 1e-9, 1e-9, X)

        # Constancy check
        self._log(f"  Checking constancy...", level="verbose")

        M_Y = float(np.mean(Y))
        if np.abs(M_Y) < 1e-4:
            # Near-zero mean: strict inequality on zero-width bounds
            # would always fail, so check absolute spread instead
            if np.mean(np.abs(Y) < 1e-4) > 0.95:
                self._log(f"    -> Constant (near-zero, frac|Y|<1e-4={float(np.mean(np.abs(Y)<1e-4)):.3f})", level="verbose")
                return (dependent, "Constant")
        else:
            lo, hi = sorted([M_Y * (1 - self.constancy_threshold), M_Y * (1 + self.constancy_threshold)])
            self._log(f"    band=[{lo:.4g}, {hi:.4g}], frac_in={float(np.mean((Y>lo)&(Y<hi))):.3f}", level="verbose")
            if np.mean((Y > lo) & (Y < hi)) > 0.95:
                self._log(f"    -> Constant", level="verbose")
                return (dependent, "Constant")

        # Linearity check
        self._log(f"  Checking linearity...", level="verbose")

        # Sort by X, compute finite-difference slopes, test constancy
        sorted_idx = np.argsort(X)
        X_s = X[sorted_idx]
        Y_s = Y[sorted_idx]
        dx = np.diff(X_s)
        dy = np.diff(Y_s)
        safe_dx = np.where(np.abs(dx) < 1e-9, 1e-9, dx)
        slopes = dy / safe_dx

        median_slope = np.median(slopes)
        slope_cv = iqr(slopes) / (np.abs(median_slope) + 1e-8)
        self._log(f"    median_slope={median_slope:.4g}, slope_cv={slope_cv:.4g} vs thr={self.constancy_threshold}", level="verbose")

        if slope_cv < self.constancy_threshold:
            m = float(median_slope)
            intercept = float(np.median(Y - m * X))
            intercept_cv = np.abs(intercept) / (np.abs(M_Y) + 1e-8)
            self._log(f"    m={m:.4g}, intercept={intercept:.4g}, intercept_cv={intercept_cv:.4g}", level="verbose")

            if intercept_cv < self.constancy_threshold:
                # Negligible intercept: y = mx -> invariant is y/x
                new_expr = (dependent.symbol) / (independent.symbol) # type: ignore[operator]
                if str(new_expr) in self.known_expressions:
                    self._log(f"    -> Null (Ratio {new_expr} already known)", level="verbose")
                    return (None, "Null")
                vals = Y / safe_X
                self._log(f"    -> Linear (negligible intercept) => {new_expr}", level="verbose")
                return (Term(new_expr, vals), "Linear")
            else:
                # Significant intercept: invariant is y − mx
                m_sym = float(f"{m:.4g}")
                new_expr = (dependent.symbol) - m_sym * (independent.symbol) # type: ignore[operator]
                vals = Y - m * X
                self._log(f"    -> Linear (significant intercept) => {new_expr}", level="verbose")
                return (Term(new_expr, vals), "Linear")
        
        # Uncorrelatedness check
        self._log(f"  Checking uncorrelatedness...", level="verbose")

        r = calculate_r(X, Y)

        self._log(f"    Pearson r={r:.4f} vs r_threshold={self.r_threshold}", level="verbose")

        if np.abs(r) < self.r_threshold:
            self._log(f"    -> Null (uncorrelated)", level="verbose")
            return (None, "Null")

        # Monotonic trend check
        self._log(f"  Checking for monotonic trend...", level="verbose")
        if r > 0:
            # co-varying: divide to get invariant
            new_expr = (dependent.symbol) / (independent.symbol) # type: ignore[operator]
            if str(new_expr) in self.known_expressions:
                self._log(f"    -> Null (Ratio {new_expr} already known)", level="verbose")
                return (None, "Null")
            vals = Y / safe_X
            self._log(f"    -> Ratio => {new_expr}", level="verbose")
            return (Term(new_expr, vals), "Ratio")
        elif r < 0:
            # inversely varying: multiply to get invariant
            new_expr = (dependent.symbol) * (independent.symbol) # type: ignore[operator]
            if str(new_expr) in self.known_expressions:
                self._log(f"    -> Null (Product {new_expr} already known)", level="verbose")
                return (None, "Null")
            vals = Y * X
            self._log(f"    -> Product => {new_expr}", level="verbose")
            return (Term(new_expr, vals), "Product")
        
        self._log(f"    -> Null (r=0)", level="verbose")
        return (None, "Null")
    

    def _contains_target(self, expr: Expr) -> bool:
        """Check whether the target variable appears in expr's free symbols."""
        if self.target_var is None:
            return False
        return self.target_var in expr.free_symbols
    

    def _rearrange(self, term: Term) -> Expr | None:
        """
        Solve a discovered law for the target variable.

        Given a term whose values are approximately constant (e.g.
        V/(I*R) = k), sets up the equation term.symbol = k and solves
        for the target variable to obtain an explicit prediction
        formula (e.g. V = k*I*R).

        Near-zero constants (|k| < 1e-9) are snapped to exactly zero
        to avoid floating-point artifacts. All floats in the result
        are rounded to 4 significant figures for readability.

        Args:
            term: A Term whose values are constant, as detected by
                the constancy check in _check.

        Returns:
            A SymPy expression for the target variable, or None if
            SymPy cannot solve the equation.
        """
        k = float(np.mean(term.values))
        if np.abs(k) < 1e-9:
            k = 0.0
        equation = sympy.Eq(term.symbol, k)
        solutions = sympy.solve(equation, self.target_var)
        
        if not solutions:
            return None
        
        expr = solutions[0]
        
        # Round all numeric atoms to 4 significant figures
        expr = expr.xreplace({
            n: sympy.Float(float(n), 4)
            for n in expr.atoms(sympy.Number)
            if n.is_Float
        })
        return expr


    def _get_diagnostics(self, expr: Expr | None) -> dict[str, float]:
        """
        Compute predictive quality metrics for a rearranged law.

        Evaluates the rearranged expression on the original data
        (stored in sym_to_vals during initialisation) and compares
        predictions against the actual target values.

        Args:
            expr: A SymPy expression for the target variable, as
                returned by _rearrange. If None, returns zero-quality
                diagnostics indicating complete failure.

        Returns:
            Dict with keys "R-squared", "MSE", "MAE".
        """
        if expr is None or self.target_values is None:
            return {
                "R-squared": float("nan"),
                "MSE": float("nan"),
                "MAE": float("nan"),
            }
        
        free_syms = list(expr.free_symbols)
        func = sympy.lambdify(free_syms, expr, modules=["numpy"])

        try:
            args = [self.sym_to_vals[str(s)] for s in free_syms]
            y_pred = func(*args)
        except (KeyError, Exception):
            return {"R-squared": float("nan"), "MSE": float("nan"), "MAE": float("nan")}

        return {
            "R-squared": calculate_r2(self.target_values, y_pred),
            "MSE": calculate_mse(self.target_values, y_pred),
            "MAE": calculate_mae(self.target_values, y_pred),
        }
    

    def discover(self, data: pd.DataFrame, target_col: str, seed: int = 73) -> tuple[str, dict[str, float]]:
        """
        Run the BACON.3F discovery loop.

        Iterates up to max_depth layers. At each layer, all novel
        directed pairs of pool variables are checked via _check.
        Constant composites containing the target are recorded as
        discovered laws. If one exceeds the R² threshold the search
        halts immediately (early stopping). Non-constant composites
        are promoted into the pool for the next layer.

        Args:
            data: DataFrame containing all variables including target.
            target_col: Name of the target column.
            seed: Random seed for reproducibility (currently used only
                for np.random.seed so the algorithm is deterministic).

        Returns:
            A (equation_string, diagnostics) tuple for the best
            discovered law, where diagnostics contains "R-squared",
            "MSE", and "MAE". Returns ("No law found", {...}) on
            failure.
        """

        np.random.seed(seed)

        # Reinitialise all internal states for a fresh discovery run.
        # TODO: do these really need to be instance variables if they're all reset here? Could they just be local variables within discover()?
        self.logs = []
        self.variable_pool = []
        self.target_var = None
        self.target_values = None
        self.discovered_laws = []
        self.discovered_strs = set()
        self.tried_permutations = set()
        self.known_expressions = set()
        self.sym_to_vals = {}
        
        self.target_var = symbols(target_col)
        self.target_values = np.asarray(data[target_col].values)

        # Initialise pool with one Term per column
        for col in data.columns:
            sym = symbols(col)
            col_values = np.asarray(data[col].values)
            var = Term(sym, col_values)
            self.variable_pool.append(var)
            self.known_expressions.add(str(sym))
            self.sym_to_vals[str(sym)] = col_values

        self._log(f"Pool init: {len(self.variable_pool)} vars: {[str(v.symbol) for v in self.variable_pool]}", level="verbose")
        
        if self.target_var is None:
            self._log("Stop: target variable not found")
            return ("No law found", {"R-squared": float("nan"), "MSE": float("nan"), "MAE": float("nan")})

        self._log(f"Starting discovery. Target: '{str(self.target_var)}'. Seed: {seed}. Shape: {data.shape}")

        # Main loop
        for i in range(self.max_depth):
            self._log(f"--- Layer {i+1} ---")

            candidates_this_layer = []

            for (dependent, independent) in permutations(self.variable_pool, 2):
                self._log(f"Pair: ({dependent.symbol}, {independent.symbol})", level="verbose")

                # Skip pairs already checked in previous layers
                if (str(dependent.symbol), str(independent.symbol)) in self.tried_permutations:
                    self._log(f"  Skip: already tried", level="verbose")
                    continue

                self.tried_permutations.add((str(dependent.symbol), str(independent.symbol)))

                result, relation_type = self._check(dependent, independent)

                if result is None:
                    self._log(f"  No composite produced", level="verbose")
                    continue

                if relation_type == "Constant":
                    self._log(f"  Constant: {result.symbol}", level="verbose")
                    # Only record laws that involve the target variable
                    if self._contains_target(result.symbol) and str(result.symbol) not in self.discovered_strs:
                        self._log(f"  Contains target & novel -> evaluating as candidate law", level="verbose")
                        self.discovered_strs.add(str(result.symbol))
                        self.known_expressions.add(str(result.symbol))

                        # Rearrange to target = f(other vars) and evaluate
                        rearranged = self._rearrange(result)
                        if rearranged is None:
                            eq_str = f"{result.symbol} = {np.mean(result.values):.4g}"
                        else:
                            eq_str = f"{self.target_var} = {rearranged}"

                        diagnostics = self._get_diagnostics(rearranged)
                        self.discovered_laws.append((eq_str, diagnostics))
                        self._log(f"  Candidate: {eq_str}  (R²={diagnostics['R-squared']:.4f}, MSE={diagnostics['MSE']:.4g}, MAE={diagnostics['MAE']:.4g})", level="verbose")

                        # Early stop if law is good enough
                        if diagnostics["R-squared"] >= self.r2_threshold:
                            self._log(f"Discovery complete: {eq_str} with R²={diagnostics['R-squared']:.4f}. Early stop at layer {i+1}.")
                            return eq_str, diagnostics
                        
                    # Constants are never promoted as candidates
                    continue

                # Non-constant: promote composite into pool for next layer
                self.known_expressions.add(str(result.symbol))
                self._log(f"  Promote -> {result.symbol}", level="verbose")
                candidates_this_layer.append(result)

            if not candidates_this_layer:
                self._log("Stop: no new composites generated")
                break
            
            self.variable_pool.extend(candidates_this_layer)
            if self.log_level == "verbose":
                self._log(f"Layer {i+1} complete. Promoted: {[str(v.symbol) for v in candidates_this_layer]}", level="verbose")
            else:
                self._log(f"Layer {i+1} complete. Promoted {len(candidates_this_layer)} candidates.")
        
        # Post-loop: return best of whatever was found
        if not self.discovered_laws:
            self._log(f"Failed: No law found after {self.max_depth} layers")
            return ("No law found", {"R-squared": float("nan"), "MSE": float("nan"), "MAE": float("nan")})

        self.discovered_laws.sort(key=lambda x: (-x[1]["R-squared"], x[1]["MSE"]))
        self._log(f"Discovery complete: {self.discovered_laws[0][0]} with R²={self.discovered_laws[0][1]['R-squared']:.4f} with {len(self.variable_pool)} total expressions in pool.")
        return self.discovered_laws[0]
    