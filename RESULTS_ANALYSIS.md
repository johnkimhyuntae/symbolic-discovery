# Experimental Results: BACON.3 vs BACON.7

## Executive Summary

We conducted a comprehensive empirical comparison of BACON.3 (Langley et al., 1987) and BACON.7 (Miller & Banerjee, 2024) across 9 benchmark datasets with varying noise levels. The experiments reveal complementary strengths: **BACON.7 excels at discovering clean multi-variable laws** (e.g., ideal gas law), while **BACON.3 demonstrates superior noise resilience** for simple laws.

**Key Findings:**
- **BACON.3**: 58.3% overall success rate; higher success at 5% noise (61.1% vs 22.2%)
- **BACON.7**: 50.0% overall success rate; much stronger on clean data (77.8% vs 55.6%) and ~4.8× faster
- **Both algorithms fail on additive laws** (structural limitation of BACON framework)

---

## Experimental Setup

### Dataset Catalogue

| ID | Law | Formula | Variables | Type |
|----|-----|---------|-----------|------|
| S-1 | Simple Sum | y = x₁ + x₂ | 2 | Additive |
| S-2 | Product | y = x₁ × x₂ | 2 | Multiplicative |
| S-3 | Ratio with Offset | y = x₁/(x₂+1) | 2 | Mixed |
| S-4 | Sum of Squares | y = x₁² + x₂² | 2 | Additive |
# Experimental Results: BACON.3 vs BACON.7 (Extended Sweep)

## Executive Summary

We ran an extended empirical comparison of BACON.3 (Langley et al., 1987) and BACON.7 (Miller & Banerjee, 2024) across 9 benchmark datasets, 5 noise levels, and 10 random seeds (900 total runs). The headline result is a clear tradeoff:

- **BACON.7 dominates on perfectly clean data** and is consistently faster.
- **BACON.3 is far more robust once any noise is introduced**, but frequently returns *self-referential* equations at moderate/high noise.

**Key Findings (900 runs):**
- **Overall discovery rate** (any law returned): BACON.3 **64.4%** (290/450), BACON.7 **33.6%** (151/450)
- **Clean data (0% noise)**: BACON.7 **77.8%** vs BACON.3 **55.6%**
- **Noise sensitivity**: BACON.7 falls to **0%** at 10% noise; BACON.3 still returns laws **37.8%** of the time
- **Runtime**: BACON.7 averages **0.0034s/run** vs BACON.3 **0.0207s/run** (~**6.1×** faster)
- **Structural limitation**: both still fail on the purely additive benchmark S-1

---

## Experimental Setup

### Dataset Catalogue

| ID | Law | Formula | Vars | Type |
|----|-----|---------|------|------|
| S-1 | Simple Sum | y = x₁ + x₂ | 2 | Additive |
| S-2 | Product | y = x₁ × x₂ | 2 | Multiplicative |
| S-3 | Ratio with Offset | y = x₁/(x₂+1) | 2 | Mixed |
| S-4 | Sum of Squares | y = x₁² + x₂² | 2 | Additive |
| T-1 | Ohm's Law | V = I × R | 2 | Multiplicative |
| T-2 | Hooke's Law | F = k × x | 2 | Multiplicative |
| T-3 | Free Fall | s = 4.905t² | 1 | Power |
| T-4 | Ideal Gas | T = PV/(nR) | 3 | Multiplicative |
| T-5 | Stefan–Boltzmann | P = 5.67×10⁻⁸T⁴ | 1 | High Power |

### Configuration

- **Noise levels**: 0, 0.01, 0.02, 0.05, 0.10
- **Seeds**: 42–51 (10 seeds)
- **Sample size**: 300 points per dataset
- **Total runs**: 900 (9 datasets × 2 methods × 5 noise × 10 seeds)

### Noise Model

Additive Gaussian noise: `y_noisy = y_true + N(0, σ)` where `σ = noise × range(y)`.

### What “Success” Means Here

This analysis uses the runner’s run-level status: a run is **Success** if the algorithm returns a non-empty equation (i.e., not “No law found”). Predictive $R^2$ is reported for quality, but **success does not require $R^2$ ≥ 0.99**.

For that reason, we also report a *high-quality subset* metric: **HQ Success** = Success with predictive $R^2 \ge 0.99$.

---

## Overall Performance

### Discovery Rate (Any Law Returned)

```
BACON.3: 290/450 (64.4%)
BACON.7: 151/450 (33.6%)
Total:   441/900 (49.0%)
```

### Predictive R² Quality (Successful Runs Only)

```
BACON.3: mean 0.9672 (median 0.9891; σ=0.0352)
BACON.7: mean 0.9472 (median 0.9929; σ=0.0844)
```

The medians are high for both, but BACON.7 has a wider spread because it sometimes returns approximate invariants (notably S-3).

### Runtime

```
BACON.3: mean 0.0207s/run (median 0.0100s)
BACON.7: mean 0.0034s/run (median ~0.0000s)
Speed:   BACON.7 ~6.1× faster on average
```

---

## Performance vs Noise

### Success Rate by Noise Level

| Noise | BACON.3 | BACON.7 |
|------:|--------:|--------:|
| 0.00  | 50/90 (55.6%) | 70/90 (77.8%) |
| 0.01  | 77/90 (85.6%) | 31/90 (34.4%) |
| 0.02  | 72/90 (80.0%) | 30/90 (33.3%) |
| 0.05  | 57/90 (63.3%) | 20/90 (22.2%) |
| 0.10  | 34/90 (37.8%) |  0/90 (0.0%)  |

### HQ Success Rate (R² ≥ 0.99 among successes)

| Noise | BACON.3 HQ | BACON.7 HQ |
|------:|-----------:|-----------:|
| 0.00  | 50/50 (100%) | 60/70 (85.7%) |
| 0.01  | 50/77 (64.9%) | 19/31 (61.3%) |
| 0.02  | 39/72 (54.2%) | 10/30 (33.3%) |
| 0.05  |  0/57 (0.0%)  |  0/20 (0.0%)  |
| 0.10  |  0/34 (0.0%)  |    —          |

Interpretation: by 5% noise, both methods rarely reach near-perfect predictive fits ($R^2 \ge 0.99$) even when they return a “law”.

---

## Performance by Dataset (All Noise Levels)

| Dataset | BACON.3 Success | BACON.7 Success | Notes |
|--------:|----------------:|----------------:|------|
| S-1 | 0/50 (0%) | 0/50 (0%) | Pure additive: unsupported |
| S-2 | 48/50 (96%) | 40/50 (80%) | Product is easiest case |
| S-3 | 16/50 (32%) | 30/50 (60%) | BACON.7 often finds an *approximate* invariant |
| S-4 | 16/50 (32%) | 0/50 (0%) | Additive + powers breaks both; BACON.3 finds spurious closures |
| T-1 | 34/50 (68%) | 11/50 (22%) | Noise hurts BACON.7 strongly |
| T-2 | 44/50 (88%) | 10/50 (20%) | Similar to T-1 |
| T-3 | 50/50 (100%) | 10/50 (20%) | BACON.7 succeeds only at 0% noise |
| T-4 | 32/50 (64%) | 40/50 (80%) | BACON.7’s standout multi-var win (until high noise) |
| T-5 | 50/50 (100%) | 10/50 (20%) | BACON.7 succeeds only at 0% noise |

---

## Representative Equations (Best R² examples)

These are representative best-$R^2$ equations from the sweep (exact strings from the run outputs):

### S-3 (Ratio with Offset)

- BACON.7 (clean): `x2*y/x1 = 0.8026` (R² ≈ 0.817)
- BACON.3 (1% noise, best): `y = 0.578 * (x1/x2) + 0.2219` (R² ≈ 0.963)

Interpretation: BACON.7 tends to return a *homogeneous approximation* to a non-homogeneous law (the `+1` offset). BACON.3 occasionally returns a higher-$R^2$ regression-style approximation.

### T-4 (Ideal Gas Law)

- BACON.7 (clean): `T*n/(P*V) = 0.1203` (R² = 1.000)
- BACON.7 (1% noise, best): `0.05985*n + T/(P*V) = 0.1728` (R² ≈ 0.993)

At 10% noise, BACON.7 returned no laws for T-4 in this sweep.

### T-5 (Stefan–Boltzmann)

- BACON.3 (clean): `P = 5.67e-08 * (T**4) + 0` (R² = 1.000)
- BACON.7 (clean): `P/(T*T³) = 5.67e-08` (R² = 1.000)

From 1% noise upward, BACON.7 returned no laws for T-5 in this sweep.

---

## Key Insights

### 1) Structural Limits Remain

S-1 (pure addition) is still a complete failure mode for both methods, consistent with the BACON family’s multiplicative/divisive term bias.

### 2) Clean Multi-Variable Discovery Is BACON.7’s Sweet Spot

On noise-free data, BACON.7 is reliably strong (77.8% success overall at 0% noise) and solves the 3-variable ideal gas invariant perfectly.

### 3) BACON.7’s Noise Collapse Is Abrupt

Across the sweep, BACON.7 success drops from 77.8% (0% noise) to 0% (10% noise). Many of its classic “wins” (T-3, T-5) vanish immediately once noise is non-zero.

### 4) BACON.3’s “Robustness” Has a Caveat

BACON.3 keeps returning equations at higher noise rates, but the forms frequently become self-referential (e.g., involving the target on both sides such as `P = a*(P*T) + b`). These can still yield high $R^2$ on the sampled data but are not mechanistic rediscoveries of the ground-truth law.

---

## Case Studies (Strengths and Weaknesses)

This section highlights a few representative “stories” from the extended sweep, using success rates over the 10 seeds and example equations with their predictive $R^2$.

### Case Study A — Clean 3-variable law (T-4 Ideal Gas): BACON.7’s core strength

**What this tests**: multi-variable multiplicative structure and deeper search.

- **At 0% noise (10 seeds)**:
	- BACON.7: **10/10 successes**, best example `T*n/(P*V) = 0.1203` ($R^2=1.000$)
	- BACON.3: **0/10 successes** (no closing relation found)

- **At 5% noise (10 seeds)**:
	- BACON.7: **10/10 successes**, example `T*n/(P*V) = 0.1221` ($R^2\approx0.950$)
	- BACON.3: **10/10 successes**, but typical form is self-referential, e.g. `T = 0.06102 * (T**2*n) + 2.247` ($R^2\approx0.929$)

- **At 10% noise (10 seeds)**:
	- BACON.7: **0/10 successes**
	- BACON.3: **2/10 successes**, best example `T = 0.05894 * (T**2*n) + 2.238` ($R^2\approx0.910$)

**Takeaway**: BACON.7 can exactly recover the clean invariant and stays competitive at moderate noise, but collapses by 10% noise. BACON.3 fails on the clean multi-var case yet often “succeeds” under noise with target-involving closures.

### Case Study B — Clean-to-noisy 1-variable power law (T-3 Free Fall): BACON.3’s robustness vs BACON.7’s brittleness

**What this tests**: power-law discovery under increasing noise.

- **At 1% noise (10 seeds)**:
	- BACON.3: **10/10 successes**, best example `s = 4.913 * (t**2) - 0.03301` ($R^2\approx0.999$)
	- BACON.7: **0/10 successes**

- **At 5% noise (10 seeds)**:
	- BACON.3: **10/10 successes**, but example becomes self-referential: `s = 0.5103 * (s*t) + 1.552` ($R^2\approx0.968$)
	- BACON.7: **0/10 successes**

**Takeaway**: BACON.7 is excellent on perfectly clean invariants, but in this implementation it is extremely noise-sensitive. BACON.3 keeps returning relationships as noise rises, but interpretability often degrades.

### Case Study C — High-power law (T-5 Stefan–Boltzmann): “correct under tiny noise”, then self-referential

**What this tests**: higher-order term generation and stability under noise.

- **At 0% noise**:
	- BACON.3: `P = 5.67e-08 * (T**4) + 0` ($R^2=1.000$)
	- BACON.7: `P/(T*T³) = 5.67e-08` ($R^2=1.000$)

- **At 1% noise (10 seeds)**:
	- BACON.3: **10/10 successes**, best example `P = 5.68e-08 * (T**4) - 5.577` ($R^2\approx0.999$)
	- BACON.7: **0/10 successes**

- **At 5% noise (10 seeds)**:
	- BACON.3: **10/10 successes**, and **100%** of these contain the target on both sides (e.g. `P = 0.002059 * (P*T) + 127.9`, $R^2\approx0.988$)
	- BACON.7: **0/10 successes**

**Takeaway**: BACON.3 can still fit the correct $T^4$ form at very low noise, but for moderate noise it tends to return a different kind of “closure” that is predictive yet not an interpretable law. BACON.7 does not tolerate even 1% noise here.

### Case Study D — Non-homogeneous law (S-3 Ratio with Offset): approximate invariants and low-$R^2$ “successes”

**What this tests**: handling of offsets like `(x2 + 1)` that break simple homogeneity.

- **At 0% noise (10 seeds)**:
	- BACON.7: **10/10 successes**, typical invariant `x2*y/x1 \approx 0.80` with best $R^2\approx0.817$
	- BACON.3: **0/10 successes**

- **At 1% noise (10 seeds)**:
	- BACON.7: **10/10 successes**, but still approximate: best `x2*y/x1 = 0.7999` ($R^2\approx0.804$)
	- BACON.3: **10/10 successes**, best example `y = 0.578 * (x1/x2) + 0.2219` ($R^2\approx0.963$)

**Takeaway**: BACON.7 consistently returns an elegant but approximate invariant (low predictive $R^2$). BACON.3 can fit better predictive approximations under slight noise, but the discovered expression is not the true offset law.

### Case Study E — Simple multiplicative law under noise (S-2 Product): interpretability vs self-reference

**What this tests**: how each method behaves on an “easy” multiplicative law when noise is present.

- **At 5% noise (10 seeds)**:
	- BACON.7: **10/10 successes**, interpretable invariant form, e.g. `y/(x1*x2) = 0.9744` ($R^2\approx0.961$)
	- BACON.3: **10/10 successes**, but a common best form is self-referential, e.g. `y = 0.04354 * (y**2) + 4.073` ($R^2\approx0.938$)

**Takeaway**: on a straightforward multiplicative law, BACON.7 tends to stay in a physically meaningful “constant ratio” form. BACON.3 often returns a closure in terms of the target itself once noise is introduced.

### Case Study F — Additive law (S-1): shared structural limitation

- **At 0% noise (10 seeds)**:
	- BACON.3: **0/10 successes**
	- BACON.7: **0/10 successes**

**Takeaway**: pure addition remains outside the representational bias of both approaches.

---

## Conclusion

Across 900 runs, **BACON.7 is best viewed as a fast, clean-data invariant finder** (especially for multi-variable multiplicative laws), while **BACON.3 is more likely to return *some* high-$R^2$ relationship under noise**, albeit often in less interpretable/self-referential forms.

*Generated: February 4, 2026*  
*Experiment: results/extended_2026-02-04_900runs/combined.csv (900 runs)*  
*Algorithms: BACON.3 (Langley et al., 1987), BACON.7 (Miller & Banerjee, 2024)*  
```


