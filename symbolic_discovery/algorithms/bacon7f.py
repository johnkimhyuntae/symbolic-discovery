from itertools import permutations
import numpy as np
import pandas as pd
import sympy
from dataclasses import dataclass
from sympy import Symbol, Expr, symbols
from typing import Tuple
from collections import Counter
from numpy.polynomial import Polynomial
from ..utils import calculate_mae, calculate_r2, calculate_mse, calculate_r


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


class BACON7F:
    """
    BACON.7F: a flat tabular data adaptation of Miller's BACON.7.

    Extends BACON.3F's pool-based discovery with noise-resilience
    mechanisms from Miller and more, adapted for flat tabular data:

    1. Fold partitioning and voting: each directed pair is classified on
    multiple random folds, and a majority vote determines the winning
    relation type.

    2. Symmetry-apply: the winning structural form is 
    re-applied to the full data via _apply, so no 
    rows are lost.

    3. Per-layer threshold relaxation: epsilon and delta are 
    multiplied by scale_factor at the start of each layer, 
    compensating for noise propagation through composed variables.
    """
    def __init__(self, 
                 max_depth: int = 6,
                 initial_epsilon: float = 0.01,
                 initial_delta: float = 0.1,    
                 c_val: float = 0.2,
                 r_threshold: float = 0.45,
                 scale_factor: float = 1.2,
                 n_folds: int = 3,
                 r2_threshold: float = 0.9,
                 log_level: str = "default"):
        """
        Initialise the BACON.7F solver.

        Args:
            max_depth: Maximum number of discovery layers before stopping.
                Each layer tests all novel directed pairs and promotes
                non-constant composites.
            initial_epsilon: Linearity threshold for the heuristic layer.
                A pair is deemed linear if 1 - |r| < epsilon.
            initial_delta: Constancy threshold for the heuristic layer.
                A variable is constant if all values fall within
                mean x (1 ± delta).
            c_val: Intercept negligibility threshold. If
                |intercept / mean(Y)| < c_val, the intercept is treated
                as zero and a ratio y/x is preferred over a linear
                residual y - mx.
            r_threshold: Minimum absolute Pearson correlation coefficient
                for a pair to be considered correlated.
            scale_factor: Multiplicative relaxation applied to epsilon
                and delta at the start of each layer.
            n_folds: Number of random folds to partition each
                directed pair into for voting. Defaults to 5.
                Automatically reduces n_folds when data is too small for 
                reliable per-fold classification.
            r2_threshold: Minimum predictive R² for early stopping. When
                a discovered law meets or exceeds this threshold, the
                search halts immediately.
            log_level: Controls the verbosity of the decision log. Options are
                "default", "verbose", and "quiet".
        """
        
        self.max_depth = max_depth
        self.epsilon = initial_epsilon
        self.delta = initial_delta
        self.c_val = c_val
        self.r_threshold = r_threshold
        self.scale_factor = scale_factor
        self.n_folds = n_folds
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
        print(f"[BACON.7F] {message}")
        return

    
    def _check(self, dependent: Term, independent: Term) -> str:
        """
        Classifies the relationship between a dependent and independent
        term. Returns only a relation type string. The Term construction 
        is deferred to _apply, which runs once on full data.

        Applies four checks in this order:

        1. **Constancy**: is the dependent already constant within 
            delta?

        2. **Linearity**: is |r| close to 1 (within epsilon)? AKA, do 
            dependent and independent have a near-perfect linear relationship?

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
            One of {"Constant", "Linear", "Ratio", "Product", "Null"}.
        """

        X = independent.values
        Y = dependent.values

        # Constancy check
        self._log("    Checking constancy...", level="verbose")

        M_Y = float(np.mean(Y))
        if np.abs(M_Y) < 1e-4:
            if np.mean(np.abs(Y) < 1e-4) > 0.95:
                self._log(f"      -> Constant (near-zero, frac|Y|<1e-4={float(np.mean(np.abs(Y)<1e-4)):.3f})", level="verbose")
                return "Constant"
        else:
            lo, hi = sorted([M_Y * (1 - self.delta), M_Y * (1 + self.delta)])
            self._log(f"      band=[{lo:.4g}, {hi:.4g}], frac_in={float(np.mean((Y>lo)&(Y<hi))):.3f}", level="verbose")
            if np.mean((Y > lo) & (Y < hi)) > 0.95:
                self._log("      -> Constant", level="verbose")
                return "Constant"

        # Linearity check
        self._log("    Checking linearity...", level="verbose")

        r = calculate_r(X, Y)
        self._log(f"      Pearson r={r:.4f}, 1-|r|={1-abs(r):.4g} vs epsilon={self.epsilon}", level="verbose")
        if (1 - abs(r)) < self.epsilon:
            self._log("      -> Linear", level="verbose")
            return "Linear"

        # Uncorrelatedness check
        self._log("    Checking uncorrelatedness...", level="verbose")
        if abs(r) < self.r_threshold:
            self._log("      -> Null (uncorrelated)", level="verbose")
            return "Null"

        # Monotonic trend checks
        self._log("    Checking for monotonic trend...", level="verbose")
        if r > 0:
            key = f"({dependent.symbol})/({independent.symbol})"
            if key in self.known_expressions:
                self._log(f"      -> Null (Ratio {key} already known)", level="verbose")
                return "Null"
            self._log(f"      -> Ratio => {key}", level="verbose")
            return "Ratio"
        elif r < 0:
            key1 = f"({dependent.symbol})*({independent.symbol})"
            key2 = f"({independent.symbol})*({dependent.symbol})"
            if key1 in self.known_expressions or key2 in self.known_expressions:
                self._log(f"      -> Null (Product {key1} or {key2} already known)", level="verbose")
                return "Null"
            self._log(f"      -> Product => {key1}", level="verbose")
            return "Product"
        
        self._log("      -> Null (r=0)", level="verbose")
        return "Null"
    

    def _election(self, votes: list[str]) -> str:
        """
        Determine the winning relation type from fold votes.

        Args:
            votes: List of relation type strings from each fold's
                classification of the same directed pair.

        Returns:
            The winning relation type after majority vote and tiebreak.
        """
        # Constant > Linear > Ratio > Product (most specific wins)
        RELATION_PRIORITY = {"Constant": 0, "Linear": 1, "Ratio": 2, "Product": 3, "Null": 4}

        vote_counts = Counter(votes)
        self._log(f"    Vote_counts={dict(vote_counts)}", level="verbose")
        
        top_count = vote_counts.most_common()[0][1]
        tied = [rel for rel, count in vote_counts.items() if count == top_count]
        self._log(f"    top_count={top_count}, tied={tied}", level="verbose")

        if len(tied) == 1:
            winning_rel = tied[0]
        else:
            winning_rel = min(tied, key=lambda r: RELATION_PRIORITY.get(r, 99))
            self._log("    Tiebreak applied", level="verbose")
        
        self._log(f"    Election winner: {winning_rel}", level="verbose")
        return winning_rel


    def _apply(self, winning_rel: str, dependent: Term, independent: Term) -> Term | None:
        """
        Constructs the composite Term on full data.

        The fold voting (Steps 2, 3 of discover) determines which
        relation type to apply. This method applies that operation to
        all rows, preserving dataset size.

        For "Linear", the slope is computed fresh on full data via
        Polynomial.fit. A sub-check on intercept negligibility (c_val)
        determines whether to produce a ratio (y/x) or a linear
        residual (y - mx).

        For "Ratio" and "Product", known-expression dedup prevents
        returning composites that already exist in the pool.

        Args:
            winning_rel: Relation type from the majority vote 
                (one of {"Constant", "Linear", "Ratio", "Product"}).
            dependent: The full-data dependent Term.
            independent: The full-data independent Term.

        Returns:
            A new Term with the composite symbol and full-data values,
            or None if the composite is a known duplicate.
        """

        X = independent.values
        Y = dependent.values
        safe_X = np.where(np.abs(X) < 1e-9, 1e-9, X)

        if winning_rel == "Constant":
            self._log(f"      -> Constant: returning {dependent.symbol} unchanged", level="verbose")
            return dependent

        if winning_rel == "Ratio":
            term = (dependent.symbol) / (independent.symbol) # type: ignore[operator]
            if str(term) in self.known_expressions:
                self._log(f"      -> None (Ratio {term} already known)", level="verbose")
                return None
            self._log(f"      -> Ratio applied to full data ({len(Y)} rows)", level="verbose")
            return Term(term, Y / safe_X)

        if winning_rel == "Product":
            term = (dependent.symbol) * (independent.symbol) # type: ignore[operator]
            if str(term) in self.known_expressions:
                self._log(f"      -> None (Product {term} already known)", level="verbose")
                return None
            self._log(f"      -> Product applied to full data ({len(Y)} rows)", level="verbose")
            return Term(term, Y * X)

        if winning_rel == "Linear":
            coef = Polynomial.fit(X, Y, 1).convert().coef
            c, m = float(coef[0]), float(coef[1])
            self._log(f"      Linear fit on full data: m={m:.4g}, c={c:.4g}", level="verbose")
            M_Y = float(np.mean(Y))
            self._log(f"      intercept_ratio=|c/M_Y|={abs(c / (M_Y + 1e-9)):.4g} vs c_val={self.c_val}", level="verbose")
            if abs(c / (M_Y + 1e-9)) < self.c_val:
                # Negligible intercept: y ≈ mx -> invariant is y/x
                term = (dependent.symbol) / (independent.symbol) # type: ignore[operator]
                if str(term) in self.known_expressions:
                    self._log(f"      -> None ({term} already known)", level="verbose")
                    return None
                self._log(f"      -> Linear (negligible intercept) => {term}", level="verbose")
                return Term(term, Y / safe_X)
            else:
                # Significant intercept: invariant is y − mx
                m_sym = float(f"{m:.4g}")
                term = (dependent.symbol) - m_sym * (independent.symbol) # type: ignore[operator]
                if str(term) in self.known_expressions:
                    self._log(f"      -> None ({term} already known)", level="verbose")
                    return None
                self._log(f"      -> Linear (significant intercept) => {term}", level="verbose")
                return Term(term, Y - m * X)
            
        self._log(f"      -> None (unhandled relation: {winning_rel})", level="verbose")
        return None
    

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
        are rounded to 4 significant figures of precision for readability.

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
        Run the BACON.7F discovery loop.

        Iterates up to max_depth layers. At each layer, every novel
        directed pair of pool variables is randomly partitioned into
        n_folds folds. The heuristic layer classifies each fold and a
        majority vote determines the winning relation type. _apply
        applies that operation to the full data to construct the
        composite Term.

        Constant composites containing the target are recorded as
        discovered laws. If one exceeds the R² threshold the search
        halts immediately (early stopping). Non-constant composites
        are promoted into the pool for the next layer.

        The fold-voting architecture means each pair costs n_folds
        heuristic-layer calls instead of one, but these are cheap
        (no sympy construction).

        Args:
            data: DataFrame containing all variables including target.
            target_col: Name of the target column.
            seed: Random seed for reproducibility. Controls both
                np.random.seed and the per-pair RNG used for 
                fold partitioning.

        Returns:
            A (equation_string, diagnostics) tuple for the best
            discovered law, where diagnostics contains "R-squared",
            "MSE", and "MAE". Returns ("No law found", {...}) on
            failure.

        """
        np.random.seed(seed)

        # Reinitialise all internal states for a fresh discovery run.
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
        
        # Decrease number of folds when data is too small for reliable per-fold classification
        effective_folds = self.n_folds
        while len(data) // effective_folds < 10 and effective_folds > 1:
            effective_folds -= 1

        if effective_folds < self.n_folds:
            self._log(f"Data too small for folding ({len(data)} rows, {self.n_folds} folds). Falling back to n_folds={effective_folds}.")

        self._log(f"Starting discovery. Target: '{str(self.target_var)}'. Seed: {seed}. Shape: {data.shape}")

        # Main loop
        for i in range(self.max_depth):
            self._log(f"--- Layer {i+1} ---")

            candidates_this_layer = []

            # Step 1: Relaxation of parameters
            # Consolidation for the propagation of noise per layer.
            self.epsilon *= self.scale_factor
            self.delta *= self.scale_factor
            if self.scale_factor != 1.0:
                self._log(f"Relaxed parameters: epsilon={self.epsilon:.4g}, delta={self.delta:.4g}", level="verbose")

            for (dependent, independent) in permutations(self.variable_pool, 2):
                self._log(f"Pair: ({dependent.symbol}, {independent.symbol})", level="verbose")

                # Skip pairs already checked in previous layers
                if (str(dependent.symbol), str(independent.symbol)) in self.tried_permutations:
                    self._log("  Skip: already tried", level="verbose")
                    continue

                self.tried_permutations.add((str(dependent.symbol), str(independent.symbol)))
                
                # Partition into random folds for voting
                rng = np.random.default_rng(seed)
                indices = rng.permutation(len(dependent.values))
                idx_folds = np.array_split(indices, effective_folds)
                dep_folds = [dependent.values[idx] for idx in idx_folds]
                ind_folds = [independent.values[idx] for idx in idx_folds]
                self._log(f"  Partitioned into {effective_folds} folds, sizes={[len(f) for f in idx_folds]}", level="verbose")

                # Step 2: Classify on each fold
                votes = []
                for subset_idx in range(effective_folds):
                    dep_partitioned = Term(dependent.symbol, dep_folds[subset_idx])
                    ind_partitioned = Term(independent.symbol, ind_folds[subset_idx])
                    relation_type = self._check(dep_partitioned, ind_partitioned)
                    votes.append(relation_type)
                    self._log(f"    Fold {subset_idx} (n={len(dep_folds[subset_idx])}): {relation_type}", level="verbose")

                if not votes:
                    continue
                
                self._log(f"  Votes collected: {votes}", level="verbose")

                # Step 3: Majority vote with priority tiebreak
                winning_rel = self._election(votes)

                if winning_rel is None or winning_rel == "Null":
                    self._log(f"  -> Skip (winner={winning_rel})", level="verbose")
                    continue
                    
                
                # Step 4: Apply winning structure to full data 
                full_term = self._apply(winning_rel, dependent, independent)

                if full_term is None:
                    self._log("  -> Skip (apply produced None)", level="verbose")
                    continue

                # Handle discovered laws
                if winning_rel == "Constant":
                    self._log(f"  Constant: {full_term.symbol}", level="verbose")
                    self.known_expressions.add(str(full_term.symbol))
                    if self._contains_target(full_term.symbol) and str(full_term.symbol) not in self.discovered_strs:
                        self._log("  Contains target & novel -> evaluating as candidate law", level="verbose")
                        self.discovered_strs.add(str(full_term.symbol))

                        # Rearrange to target = f(other vars) and evaluate
                        rearranged = self._rearrange(full_term)
                        if rearranged is None:
                            eq_str = f"{full_term.symbol} = {np.mean(full_term.values):.4g}"
                        else:
                            eq_str = f"{self.target_var} = {rearranged}"

                        diagnostics = self._get_diagnostics(rearranged)
                        self.discovered_laws.append((eq_str, diagnostics))
                        self._log(f"  Candidate: {eq_str}  (R²={diagnostics['R-squared']:.4f}, MSE={diagnostics['MSE']:.4g}, MAE={diagnostics['MAE']:.4g})", level="verbose")

                        # Early stop if law is good enough
                        if diagnostics["R-squared"] >= self.r2_threshold:
                            self._log(f"Discovery complete: {eq_str} with R²={diagnostics['R-squared']:.4f}. Early stop at layer {i+1}.")
                            return eq_str, diagnostics
                        
                    # Either way, don't promote constants as candidates
                    continue

                # Non-constant: promote composite into pool for next layer
                self.known_expressions.add(str(full_term.symbol))
                self._log(f"  Promote -> {full_term.symbol}", level="verbose")
                candidates_this_layer.append(full_term)

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
