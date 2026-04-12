from itertools import permutations
import numpy as np
import pandas as pd
import sympy
from dataclasses import dataclass
from sympy import Symbol, Expr, symbols
from typing import Tuple
from collections import Counter
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
    BACON.7F: flat tabular adaptation of Miller's BACON.7 (2024).

    Extends BACON.3F's pool-based discovery with noise-resilience
    mechanisms from Miller's factorial-design algorithm, adapted for
    flat tabular data:

    1. Subset voting: each directed pair's data is randomly 
    partitioned into n_folds subsamples. The heuristic layer 
    classifies each subsample independently, and a majority vote 
    determines the winning relation type. This filters noisy 
    misclassifications: if 2 of 3 folds say "Ratio" and 1 says 
    "Linear" due to noise, "Ratio" wins.

    2. Symmetry-apply/Averaging: the winning structural form is 
    re-applied to the full (unpartitioned) data via _average, so no 
    rows are lost. For linear relations the slope is recomputed on 
    full data rather than inheriting a fold-local estimate.

    3. Managing-layer retry (TBD): if no candidates are found, 
    epsilon and delta are relaxed by a multiplicative scale factor 
    and the search is retried.

    Key differences from Miller's original BACON.7:

    - Miller's subsets arise from factorial conditioning (holding 
    other variables fixed at specific levels). Our flat tabular 
    adaptation uses random partitioning instead, yielding correlated 
    but distinct estimates of the same bivariate relationship.

    - Miller's linearity check uses correlation coefficient thresholds. 
    This is more lenient than BACON.3F's IQR-based slope check, 
    which can cause misclassification of polynomial relationships 
    as linear (a known limitation documented in Chapter 3).

    - The voting only determines the relation type, not a full Term. 
    This avoids expensive sympy expression construction in the fold loop. 
    Terms are only built once by _average on the full data.
    """
    def __init__(self, 
                 max_depth: int = 5,
                 initial_epsilon: float = 0.05,
                 initial_delta: float = 0.05,    
                 c_val: float = 0.05,
                 scale_factor: float = 1.2,
                 big_delta: float = 0.1,
                 n_folds: int = 3,
                 r2_threshold: float = 0.990,
                 verbose: bool = False):
        """
        Initialise the BACON.7F solver.

        Args:
            max_depth: Maximum number of discovery layers before stopping.
                Each layer tests all novel directed pairs and promotes
                non-constant composites. Deeper layers find more complex
                laws but scale combinatorially.
            initial_epsilon: Linearity threshold for the heuristic layer.
                A pair is deemed linear if 1 - |r| < epsilon. More
                lenient than BACON.3F's IQR-based check.
            initial_delta: Constancy threshold for the heuristic layer.
                A variable is constant if all values fall within
                mean x (1 ± delta).
            c_val: Intercept negligibility threshold. If
                |intercept / mean(Y)| < c_val, the intercept is treated
                as zero and a ratio y/x is preferred over a linear
                residual y - mx.
            scale_factor: Multiplicative relaxation applied to epsilon
                and delta after each layer (and on managing-layer retry).
            big_delta: Separate threshold for the sufficiency check
                (Delta-pruning). Not auto-scaled. Currently unused (TBD).
            n_folds: Number of random folds to partition each
                directed pair into for voting. Defaults to 3,
                matching Miller's typical 3-value factorial design.
                Automatically falls back to 1 (no folding) when data
                is too small for reliable per-fold classification.
            r2_threshold: Minimum predictive R² for early stopping. When
                a discovered law meets or exceeds this threshold, the
                search halts immediately. Set by the wrapper based on
                the expected noise level.
            verbose: If True, print the decision log to stdout.
        """
        
        self.max_depth = max_depth
        self.epsilon = initial_epsilon
        self.delta = initial_delta
        self.c_val = c_val
        self.scale_factor = scale_factor
        self.big_delta = big_delta
        self.n_folds = n_folds
        self.r2_threshold = r2_threshold
        self.verbose = verbose
        
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


    def _log(self, message: str):
        """Append to the decision log; print if verbose."""
        self.logs.append(message)
        if self.verbose:
            if message:
                print(f"[BACON.7F] {message}")
            else:
                print("")

    
    def _check(self, dependent: Term, independent: Term) -> str:
        """
        BACON.1 heuristic layer (Miller, 2024).

        Classifies the relationship between a dependent and independent
        term. Unlike BACON.3F's _check (which returns a Term), this
        returns only a relation type string. The Term construction is
        deferred to _average, which runs once on full data.

        Applies three checks in priority order:

        1. **Constancy**: is the dependent already constant within 
        delta? Near-zero means are handled with an absolute-spread 
        check to avoid the zero-width-bounds problem.

        2. **Linearity**: is |r| close to 1 (within epsilon)? This 
        correlation-based check is more lenient than BACON.3F's 
        IQR-based slope check.

        3. **Monotonic trend**: if non-linear, the correlation sign 
        determines Ratio (r > 0) or Product (r < 0). Known 
        expressions are checked to avoid duplicates.

        Args:
            dependent: The term treated as y.
            independent: The term treated as x.

        Returns:
            One of {"Constant", "Linear", "Ratio", "Product", "Null"}.
        """

        X = independent.values
        Y = dependent.values


        # Constancy check

        M_Y = float(np.mean(Y))
        if np.abs(M_Y) < 1e-9:
            # Near-zero mean: strict inequality on zero-width bounds
            # would always fail, so check absolute spread instead
            if np.all(np.abs(Y) < 1e-9):
                return "Constant"
        else:
            lo, hi = sorted([M_Y * (1 - self.delta), M_Y * (1 + self.delta)])
            if np.all((Y > lo) & (Y < hi)):
                return "Constant"


        # Linearity check

        r = calculate_r(X, Y)
        if (1 - abs(r)) < self.epsilon:
            return "Linear"


        # Monotonic trend check

        if r > 0:
            key = f"{dependent.symbol}/{independent.symbol}"
            if key in self.known_expressions:
                return "Null"
            return "Ratio"
        elif r < 0:
            key = f"{dependent.symbol}*{independent.symbol}"
            if key in self.known_expressions:
                return "Null"
            return "Product"
        
        return "Null"


    def _managing_layer(self, dependent: Term, independent: Term) -> str:
        """
        Miller's managing layer (TBD: currently a passthrough).

        Wraps the heuristic layer. In Miller's original, this adds an
        iteration counter j (capping expression complexity at
        |n| + |m| ≤ j + 1) and retry logic with epsilon/delta scaling
        per Figure 7 of the manuscript.

        TBD: implement iteration counter and retry logic.

        Args:
            dependent: The term treated as y.
            independent: The term treated as x.

        Returns:
            Relation type string from _check.
        """
        # TBD: For now we just directly call the check function and return its result.
        return self._check(dependent, independent)
    

    def _contains_target(self, expr: Expr) -> bool:
        """Check whether the target variable appears in expr's free symbols."""
        if self.target_var is None:
            return False
        return self.target_var in expr.free_symbols


    def _average(self, winning_rel: str, dependent: Term, independent: Term) -> Term | None:
        """
        Constructs the composite Term on full data.

        The fold voting (Steps 2, 3 of discover) determines which
        relation type to apply. This method applies that operation to
        all rows, preserving dataset size.

        For "Linear", the slope is computed fresh on full data via
        polyfit. A sub-check on intercept negligibility (c_val)
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
            return dependent

        if winning_rel == "Ratio":
            term = dependent.symbol / independent.symbol # type: ignore[operator]
            if str(term) in self.known_expressions:
                return None
            return Term(term, Y / safe_X)

        if winning_rel == "Product":
            term = dependent.symbol * independent.symbol # type: ignore[operator]
            if str(term) in self.known_expressions:
                return None
            return Term(term, Y * X)

        if winning_rel == "Linear":
            m, c = np.polyfit(X, Y, 1)
            M_Y = float(np.mean(Y))
            if abs(c / (M_Y + 1e-9)) < self.c_val:
                # Negligible intercept: y ≈ mx -> invariant is y/x
                term = dependent.symbol / independent.symbol # type: ignore[operator]
                if str(term) in self.known_expressions:
                    return None
                return Term(term, Y / safe_X)
            else:
                # Significant intercept: invariant is y − mx
                m_sym = float(f"{m:.4g}")
                term = dependent.symbol - m_sym * independent.symbol # type: ignore[operator]
                return Term(term, Y - m * X)

        return None
    

    def _snap_coefficients(self, expr: Expr) -> Expr:
        """
        TBD: currently unused.
        Snap near-integer numeric coefficients to exact integers.
        E.g. 0.997 to 1, 2.003 to 2, 8.314 to 8.314 (too far from 8).
        Uses self.delta as the tolerance for snapping.

        Args:
            expr: A SymPy expression to clean up.

        Returns:
            The expression with near-integer floats replaced by integers.
        """
        replacements = {}
        for n in expr.atoms(sympy.Number):
            if not n.is_Float:
                continue
            val = float(n)
            if abs(val) < 1e-9:
                replacements[n] = sympy.Integer(0)
                continue
            nearest = round(val)
            if nearest == 0:
                continue
            if abs(1 - val / nearest) < self.delta:
                replacements[n] = sympy.Integer(nearest)
        return expr.xreplace(replacements)


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
                "R-squared": 0.0,
                "MSE": float("inf"),
                "MAE": float("inf"),
            }
        
        free_syms = list(expr.free_symbols)
        func = sympy.lambdify(free_syms, expr, modules=["numpy"])

        try:
            args = [self.sym_to_vals[str(s)] for s in free_syms]
            y_pred = func(*args)
        except (KeyError, Exception):
            return {"R-squared": 0.0, "MSE": float("inf"), "MAE": float("inf")}

        return {
            "R-squared": calculate_r2(self.target_values, y_pred),
            "MSE": calculate_mse(self.target_values, y_pred),
            "MAE": calculate_mae(self.target_values, y_pred),
        }


    def discover(self, data: pd.DataFrame, target_col: str, seed: int = 42) -> tuple[str, dict[str, float]]:
        """
        Run the BACON.7F discovery loop.

        Iterates up to max_depth layers. At each layer, every novel
        directed pair of pool variables is randomly partitioned into
        n_folds folds. The heuristic layer classifies each fold and a
        majority vote determines the winning relation type. _average
        applies that operation to the full data to construct the
        composite Term.

        Constant composites containing the target are recorded as
        discovered laws. If one exceeds the R² threshold the search
        halts immediately (early stopping). Non-constant composites
        are promoted into the pool for the next layer.

        The fold-voting architecture means each pair costs n_folds
        heuristic-layer calls instead of one, but these are cheap
        (no sympy construction). The main scalability bottleneck
        remains pool growth: deduplication via tried_permutations
        and known_expressions mitigates this, but deep searches 
        on multi-variable datasets can be slow.

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

        # TBD: either reinitialise all or none
        self.logs = []
        
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

        if self.target_var is None:
            self._log("Stop: target variable not found")
            return ("No law found", {"R-squared": 0.0, "MSE": float("inf"), "MAE": float("inf")})
        
        # TBD: small fold sizes break EVERYTHING
        # Fall back to no folding when data is too small for reliable per-fold classification
        min_fold_size = 10
        if len(data) // self.n_folds < min_fold_size:
            effective_folds = 1
            self._log(f"  Data too small for folding ({len(data)} rows, {self.n_folds} folds). Falling back to n_folds=1.")
        else:
            effective_folds = self.n_folds

        self._log(f"Starting discovery. Target: '{str(self.target_var)}'. Seed: {seed}. Shape: {data.shape}")


        # Main loop

        for i in range(self.max_depth):
            self._log(f"--- Layer {i+1} ---")

            candidates_this_layer = []


            # Step 1: Sufficiency check (TBD)

            if i != (self.max_depth-1):
                # TBD
                pass

            for (dependent, independent) in permutations(self.variable_pool, 2):
                # Skip pairs already checked in previous layers
                if (str(dependent.symbol), str(independent.symbol)) in self.tried_permutations:
                    continue
                self.tried_permutations.add((str(dependent.symbol), str(independent.symbol)))
                
                # Partition into random folds for voting
                rng = np.random.default_rng(seed)
                indices = rng.permutation(len(dependent.values))
                idx_folds = np.array_split(indices, effective_folds)
                dep_folds = [dependent.values[idx] for idx in idx_folds]
                ind_folds = [independent.values[idx] for idx in idx_folds]


                # Step 2: Classify on each fold (Space of Laws)

                votes = []
                for subset_idx in range(effective_folds):
                    dep_partitioned = Term(dependent.symbol, dep_folds[subset_idx])
                    ind_partitioned = Term(independent.symbol, ind_folds[subset_idx])
                    relation_type = self._managing_layer(dep_partitioned, ind_partitioned)
                    votes.append(relation_type)

                if not votes:
                    continue
                

                # Step 3: Majority vote with priority tiebreak

                # Constant > Linear > Ratio > Product (most specific wins)
                RELATION_PRIORITY = {"Constant": 0, "Linear": 1, "Ratio": 2, "Product": 3, "Null": 4}

                vote_counts = Counter(votes)
                top_count = vote_counts.most_common()[0][1]
                tied = [rel for rel, count in vote_counts.items() if count == top_count]

                if len(tied) == 1:
                    winning_rel = tied[0]
                else:
                    # Tiebreak: most specific relation wins
                    winning_rel = min(tied, key=lambda r: RELATION_PRIORITY.get(r, 99))

                if winning_rel is None or winning_rel == "Null":
                    self._log(f"  No consensus for {dependent.symbol}, {independent.symbol}")
                    continue
                    
                
                # Step 4: Apply winning structure to full data 

                averaged_term = self._average(winning_rel, dependent, independent)

                if averaged_term is None:
                    continue

                # Handle discovered laws
                if winning_rel == "Constant":
                    if self._contains_target(averaged_term.symbol) and str(averaged_term.symbol) not in self.discovered_strs:
                        self._log(f"  Discovered law: {str(averaged_term.symbol)} = {np.mean(averaged_term.values)}")
                        self.discovered_strs.add(str(averaged_term.symbol))
                        self.known_expressions.add(str(averaged_term.symbol))

                        # Rearrange to target = f(other vars) and evaluate
                        rearranged = self._rearrange(averaged_term)
                        if rearranged is None:
                            eq_str = f"{averaged_term.symbol} = {np.mean(averaged_term.values):.4g}"
                        else:
                            eq_str = f"{self.target_var} = {rearranged}"

                        diagnostics = self._get_diagnostics(rearranged)
                        self.discovered_laws.append((eq_str, diagnostics))

                        # Early stop if law is good enough
                        if diagnostics["R-squared"] >= self.r2_threshold:
                            self._log(f"Discovery complete: {eq_str} with R²={diagnostics['R-squared']:.4f}. Early stop at layer {i+1}.")
                            return eq_str, diagnostics
                        
                    # Either way, don't promote constants as candidates
                    continue

                # Non-constant: promote composite into pool for next layer
                self._log(f"  Candidate: {averaged_term.symbol} via {winning_rel}")
                self.known_expressions.add(str(averaged_term.symbol))
                candidates_this_layer.append(averaged_term)

            if not candidates_this_layer:
                self._log("Stop: no new composites generated")
                break
            
            self.variable_pool.extend(candidates_this_layer)
            self._log(f"Layer {i+1} complete. Promoted: {[str(v.symbol) for v in candidates_this_layer]}")
        
        # Post-loop: return best of whatever was found
        if not self.discovered_laws:
            self._log(f"Failed: No law found after {self.max_depth} layers")
            return ("No law found", {"R-squared": 0.0, "MSE": float("inf"), "MAE": float("inf")})

        self.discovered_laws.sort(key=lambda x: (-x[1]["R-squared"], x[1]["MSE"]))
        self._log(f"Discovery complete: {self.discovered_laws[0][0]} with R²={self.discovered_laws[0][1]['R-squared']:.4f}")
        return self.discovered_laws[0]
