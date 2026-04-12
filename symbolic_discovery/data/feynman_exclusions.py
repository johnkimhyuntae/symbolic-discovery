"""
TBD: NEED TO BE REDONE
Feynman Equations Expected to Fail for BACON Algorithms.

This module documents which Feynman equations are expected to fail for BACON-style
symbolic regression algorithms, along with reasons for exclusion. These equations
should be filtered out when computing success rates for fair benchmarking.

Categories of exclusion:
1. DEGENERATE: 0 independent variables in dimensionless form (pure mathematical constants)
2. TRANSCENDENTAL: Contains exp, sin, cos, arcsin, tanh, log, etc. that BACON cannot discover
3. SQRT_ONLY: Contains only sqrt which may be partially discoverable via x^0.5 but unreliable

References:
- Feynman Symbolic Benchmark: https://space.mit.edu/home/tegmark/aifeynman.html
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Optional


class ExclusionReason(Enum):
    """Reason why an equation is excluded from BACON benchmarks."""
    DEGENERATE = "degenerate"           # 0 independent variables (pure constant)
    TRANSCENDENTAL = "transcendental"   # exp, sin, cos, log, arcsin, tanh, etc.
    SQRT_ONLY = "sqrt_only"             # sqrt without other discoverable structure


@dataclass
class ExcludedEquation:
    """Information about an excluded Feynman equation."""
    eq_id: str
    reason: ExclusionReason
    original_formula: str
    notes: str


# DEGENERATE EQUATIONS
# These have 0 independent variables in dimensionless form - they're pure
# mathematical constants like 1/(4*pi*epsilon0*c**2) that don't vary with any input.

DEGENERATE_EQUATIONS: Dict[str, ExcludedEquation] = {
    "I.32.5": ExcludedEquation(
        eq_id="I.32.5",
        reason=ExclusionReason.DEGENERATE,
        original_formula="q**2*a**2/(6*pi*epsilon*c**3)",
        notes="Larmor formula - all variables factor out in dimensionless form"
    ),
    "I.38.12": ExcludedEquation(
        eq_id="I.38.12",
        reason=ExclusionReason.DEGENERATE,
        original_formula="4*pi*epsilon*(h/(2*pi))**2/(m*q**2)",
        notes="Bohr radius - pure constant in dimensionless form"
    ),
    "II.4.23": ExcludedEquation(
        eq_id="II.4.23",
        reason=ExclusionReason.DEGENERATE,
        original_formula="q/(4*pi*epsilon*r)",
        notes="Coulomb potential - variables cancel in dimensionless form"
    ),
    "II.13.17": ExcludedEquation(
        eq_id="II.13.17",
        reason=ExclusionReason.DEGENERATE,
        original_formula="1/(4*pi*epsilon*c**2)*2*I/r",
        notes="Magnetic field from current - constant 1/(4*pi*epsilon0*c**2)"
    ),
    "II.27.16": ExcludedEquation(
        eq_id="II.27.16",
        reason=ExclusionReason.DEGENERATE,
        original_formula="epsilon*c*Ef**2",
        notes="Poynting vector - degenerate in dimensionless form"
    ),
    "II.27.18": ExcludedEquation(
        eq_id="II.27.18", 
        reason=ExclusionReason.DEGENERATE,
        original_formula="epsilon*Ef**2",
        notes="Electric field energy density - degenerate"
    ),
    "II.34.2a": ExcludedEquation(
        eq_id="II.34.2a",
        reason=ExclusionReason.DEGENERATE,
        original_formula="q*v/(2*pi*r)",
        notes="Current from moving charge - degenerate"
    ),
    "II.34.29a": ExcludedEquation(
        eq_id="II.34.29a",
        reason=ExclusionReason.DEGENERATE,
        original_formula="q*h/(4*pi*m)",
        notes="Magnetic moment - ℏq/(2m) is constant"
    ),
    "III.7.38": ExcludedEquation(
        eq_id="III.7.38",
        reason=ExclusionReason.DEGENERATE,
        original_formula="2*mom*B/(h/(2*pi))",
        notes="Larmor frequency - degenerate in dimensionless form"
    ),
    "III.15.14": ExcludedEquation(
        eq_id="III.15.14",
        reason=ExclusionReason.DEGENERATE,
        original_formula="(h/(2*pi))**2/(2*E_n*d**2)",
        notes="Effective mass - degenerate"
    ),
    "III.21.20": ExcludedEquation(
        eq_id="III.21.20",
        reason=ExclusionReason.DEGENERATE,
        original_formula="-rho_c_0*q*A_vec/m",
        notes="Current density - degenerate"
    ),
}


# TRANSCENDENTAL EQUATIONS
# These contain transcendental functions (exp, sin, cos, log, arcsin, tanh)
# that BACON's polynomial/rational operations cannot discover.

TRANSCENDENTAL_EQUATIONS: Dict[str, ExcludedEquation] = {
    # Exponential/Gaussian
    "I.6.2a": ExcludedEquation(
        eq_id="I.6.2a",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="exp(-theta**2/2)/sqrt(2*pi)",
        notes="Gaussian distribution - requires exp()"
    ),
    "I.6.2": ExcludedEquation(
        eq_id="I.6.2",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)",
        notes="Normal distribution - requires exp()"
    ),
    "I.6.2b": ExcludedEquation(
        eq_id="I.6.2b",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)",
        notes="Shifted normal distribution - requires exp()"
    ),
    "I.40.1": ExcludedEquation(
        eq_id="I.40.1",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="n_0*exp(-m*g*x/(kb*T))",
        notes="Barometric formula - Boltzmann distribution with exp()"
    ),
    "I.41.16": ExcludedEquation(
        eq_id="I.41.16",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))",
        notes="Planck radiation law - requires exp()"
    ),
    "II.35.18": ExcludedEquation(
        eq_id="II.35.18",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))",
        notes="Paramagnetic susceptibility - cosh with exp()"
    ),
    "II.35.21": ExcludedEquation(
        eq_id="II.35.21",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="n_rho*mom*tanh(mom*B/(kb*T))",
        notes="Magnetization - requires tanh()"
    ),
    "III.4.32": ExcludedEquation(
        eq_id="III.4.32",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="1/(exp((h/(2*pi))*omega/(kb*T))-1)",
        notes="Bose-Einstein distribution - requires exp()"
    ),
    "III.4.33": ExcludedEquation(
        eq_id="III.4.33",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="(h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)",
        notes="Bose-Einstein energy - requires exp()"
    ),
    "III.14.14": ExcludedEquation(
        eq_id="III.14.14",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="I_0*(exp(q*Volt/(kb*T))-1)",
        notes="Diode equation - requires exp()"
    ),
    
    # Trigonometric (sin/cos)
    "I.12.11": ExcludedEquation(
        eq_id="I.12.11",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="q*(Ef+B*v*sin(theta))",
        notes="Lorentz force - requires sin()"
    ),
    "I.18.12": ExcludedEquation(
        eq_id="I.18.12",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="r*F*sin(theta)",
        notes="Torque - requires sin()"
    ),
    "I.18.14": ExcludedEquation(
        eq_id="I.18.14",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="m*r*v*sin(theta)",
        notes="Angular momentum - requires sin()"
    ),
    "I.29.16": ExcludedEquation(
        eq_id="I.29.16",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))",
        notes="Wave superposition distance - requires cos()"
    ),
    "I.30.3": ExcludedEquation(
        eq_id="I.30.3",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="Int_0*sin(n*theta/2)**2/sin(theta/2)**2",
        notes="Diffraction intensity - requires sin()"
    ),
    "I.37.4": ExcludedEquation(
        eq_id="I.37.4",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="I1+I2+2*sqrt(I1*I2)*cos(delta)",
        notes="Interference intensity - requires cos()"
    ),
    "I.50.26": ExcludedEquation(
        eq_id="I.50.26",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="x1*(cos(omega*t)+alpha*cos(omega*t)**2)",
        notes="Harmonic motion - requires cos()"
    ),
    "II.6.11": ExcludedEquation(
        eq_id="II.6.11",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="1/(4*pi*epsilon)*p_d*cos(theta)/r**2",
        notes="Dipole potential - requires cos()"
    ),
    "II.6.15b": ExcludedEquation(
        eq_id="II.6.15b",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3",
        notes="Dipole field - requires sin() and cos()"
    ),
    "II.11.17": ExcludedEquation(
        eq_id="II.11.17",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="n_0*(1+p_d*Ef*cos(theta)/(kb*T))",
        notes="Polarization distribution - requires cos()"
    ),
    "II.15.4": ExcludedEquation(
        eq_id="II.15.4",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="-mom*B*cos(theta)",
        notes="Magnetic potential energy - requires cos()"
    ),
    "II.15.5": ExcludedEquation(
        eq_id="II.15.5",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="-p_d*Ef*cos(theta)",
        notes="Electric dipole energy - requires cos()"
    ),
    "III.8.54": ExcludedEquation(
        eq_id="III.8.54",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="sin(E_n*t/(h/(2*pi)))**2",
        notes="Quantum transition probability - requires sin()"
    ),
    "III.9.52": ExcludedEquation(
        eq_id="III.9.52",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="(p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2",
        notes="Rabi oscillation - sinc² function"
    ),
    "III.15.12": ExcludedEquation(
        eq_id="III.15.12",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="2*U*(1-cos(k*d))",
        notes="Tight-binding energy - requires cos()"
    ),
    "III.17.37": ExcludedEquation(
        eq_id="III.17.37",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="beta*(1+alpha*cos(theta))",
        notes="Angular distribution - requires cos()"
    ),
    
    # Inverse Trigonometric
    "I.26.2": ExcludedEquation(
        eq_id="I.26.2",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="arcsin(n*sin(theta2))",
        notes="Snell's law - requires arcsin() and sin()"
    ),
    "I.30.5": ExcludedEquation(
        eq_id="I.30.5",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="arcsin(lambd/(n*d))",
        notes="Diffraction angle - requires arcsin()"
    ),
    
    # Logarithmic
    "I.44.4": ExcludedEquation(
        eq_id="I.44.4",
        reason=ExclusionReason.TRANSCENDENTAL,
        original_formula="n*kb*T*ln(V2/V1)",
        notes="Isothermal work - requires ln()"
    ),
}


# SQRT-ONLY EQUATIONS  
# These contain sqrt as the only non-polynomial operation. BACON can sometimes
# discover these via x^0.5 power terms, but it's unreliable.

SQRT_ONLY_EQUATIONS: Dict[str, ExcludedEquation] = {
    "I.8.14": ExcludedEquation(
        eq_id="I.8.14",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="sqrt((x2-x1)**2+(y2-y1)**2)",
        notes="Euclidean distance - sqrt of polynomial, hard to discover"
    ),
    "I.10.7": ExcludedEquation(
        eq_id="I.10.7",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="m_0/sqrt(1-v**2/c**2)",
        notes="Relativistic mass - Lorentz factor contains sqrt"
    ),
    "I.15.3x": ExcludedEquation(
        eq_id="I.15.3x",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="(x-u*t)/sqrt(1-u**2/c**2)",
        notes="Lorentz transformation x - contains sqrt"
    ),
    "I.15.3t": ExcludedEquation(
        eq_id="I.15.3t",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="(t-u*x/c**2)/sqrt(1-u**2/c**2)",
        notes="Lorentz transformation t - contains sqrt"
    ),
    "I.15.1": ExcludedEquation(
        eq_id="I.15.1",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="m_0*v/sqrt(1-v**2/c**2)",
        notes="Relativistic momentum - Lorentz factor"
    ),
    "I.47.23": ExcludedEquation(
        eq_id="I.47.23",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="sqrt(gamma*pr/rho)",
        notes="Speed of sound - sqrt of quotient"
    ),
    "I.48.2": ExcludedEquation(
        eq_id="I.48.2",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="m*c**2/sqrt(1-v**2/c**2)",
        notes="Relativistic energy - Lorentz factor"
    ),
    "II.13.23": ExcludedEquation(
        eq_id="II.13.23",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="rho_c_0/sqrt(1-v**2/c**2)",
        notes="Relativistic charge density - Lorentz factor"
    ),
    "II.13.34": ExcludedEquation(
        eq_id="II.13.34",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="rho_c_0*v/sqrt(1-v**2/c**2)",
        notes="Relativistic current - Lorentz factor"
    ),
    "II.24.17": ExcludedEquation(
        eq_id="II.24.17",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="sqrt(omega**2/c**2-pi**2/d**2)",
        notes="Waveguide wavenumber - sqrt of difference"
    ),
    "III.10.19": ExcludedEquation(
        eq_id="III.10.19",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="mom*sqrt(Bx**2+By**2+Bz**2)",
        notes="Magnetic energy - sqrt of sum of squares"
    ),
    "II.6.15a": ExcludedEquation(
        eq_id="II.6.15a",
        reason=ExclusionReason.SQRT_ONLY,
        original_formula="p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)",
        notes="Dipole field component - sqrt of sum of squares"
    ),
}


# COMBINED EXCLUSION SETS

def get_all_excluded_equations() -> Dict[str, ExcludedEquation]:
    """Return all excluded equations combined."""
    all_excluded = {}
    all_excluded.update(DEGENERATE_EQUATIONS)
    all_excluded.update(TRANSCENDENTAL_EQUATIONS)
    all_excluded.update(SQRT_ONLY_EQUATIONS)
    return all_excluded


def get_excluded_ids_by_reason(reason: ExclusionReason) -> Set[str]:
    """Get equation IDs excluded for a specific reason."""
    mapping = {
        ExclusionReason.DEGENERATE: DEGENERATE_EQUATIONS,
        ExclusionReason.TRANSCENDENTAL: TRANSCENDENTAL_EQUATIONS,
        ExclusionReason.SQRT_ONLY: SQRT_ONLY_EQUATIONS,
    }
    return set(mapping.get(reason, {}).keys())


def get_strict_exclusions() -> Set[str]:
    """
    Get equations that should ALWAYS be excluded from BACON benchmarks.
    
    This includes:
    - Degenerate equations (no independent variables)
    - Transcendental equations (exp, sin, cos, log, etc.)
    
    Does NOT include sqrt-only equations since these are sometimes discoverable.
    """
    strict = set()
    strict.update(DEGENERATE_EQUATIONS.keys())
    strict.update(TRANSCENDENTAL_EQUATIONS.keys())
    return strict


def get_all_exclusions() -> Set[str]:
    """
    Get ALL equations that should be excluded from BACON benchmarks.
    
    Includes all categories including sqrt-only.
    """
    return set(get_all_excluded_equations().keys())


# STATISTICS

def print_exclusion_summary():
    """Print a summary of exclusions."""
    print("=== Feynman Equation Exclusions for BACON ===\n")
    
    print(f"Degenerate (0 variables):    {len(DEGENERATE_EQUATIONS):3d}")
    print(f"Transcendental (exp/sin/..): {len(TRANSCENDENTAL_EQUATIONS):3d}")  
    print(f"Sqrt-only:                   {len(SQRT_ONLY_EQUATIONS):3d}")
    print(f"{'─'*45}")
    print(f"Total excluded:              {len(get_all_excluded_equations()):3d}")
    print(f"Strict exclusions:           {len(get_strict_exclusions()):3d}")
    print()
    
    # Assuming 100 total Feynman equations
    total_feynman = 100
    strict = len(get_strict_exclusions())
    all_exc = len(get_all_excluded_equations())
    
    print(f"Total Feynman equations:     {total_feynman}")
    print(f"Valid for BACON (strict):    {total_feynman - strict}")
    print(f"Valid for BACON (all excl):  {total_feynman - all_exc}")


# LIST OF DISCOVERABLE EQUATIONS
# These are the equations that BACON algorithms CAN potentially discover.

# Equations with polynomial/power-law relationships that BACON can handle
DISCOVERABLE_EQUATIONS = [
    # Simple products
    "I.12.1",   # F = mu*Nn (linear product)
    "I.12.5",   # F = q2*Ef (linear product)
    "I.14.3",   # U = m*g*z (product of 3 terms)
    "I.34.27",  # E_n = h*omega (linear product)
    "I.39.1",   # E_n = 3/2*pr*V (product with constant)
    "I.43.31",  # D = mob*kb*T (product of 3 terms)
    "III.12.43", # L = n*h/(2*pi) (linear)
    
    # Quotients
    "I.25.13",  # Volt = q/C (division)
    "I.29.4",   # k = omega/c (division)
    "I.43.16",  # v = mu*q*Volt/d (products and divisions)
    "II.3.24",  # flux = Pwr/(4*pi*r**2) (inverse square)
    
    # Power laws  
    "I.14.4",   # U = k*x**2 (quadratic)
    "I.24.6",   # E_n = m*omega**2*x**2 (products with squares)
    "I.34.1",   # omega = omega_0/(1-v/c) (denominator)
    "I.34.14",  # omega = (1+v/c)/sqrt(1-v**2/c**2)*omega_0 (complex but algebraic)
    
    # Linear combinations
    "I.11.19",  # A = x1*y1+x2*y2+x3*y3 (dot product)
    "I.16.6",   # v1 = (u+v)/(1+u*v/c**2) (velocity addition)
    "I.18.4",   # r = (m1*r1+m2*r2)/(m1+m2) (center of mass)
    
    # Inverse relationships
    "I.27.6",   # foc = 1/(1/d1+n/d2) (lens equation)
    "I.12.2",   # F = q1*q2/(4*pi*epsilon*r**2) (Coulomb)
    
    # More complex but polynomial
    "I.9.18",   # Newton's gravitation
    "I.13.12",  # Gravitational potential energy difference
    "I.32.17",  # Radiation scattering (complex polynomial)
    "I.39.11",  # Energy with gamma factor
    "I.39.22",  # Ideal gas law
    "I.43.43",  # Thermal conductivity
    "II.2.42",  # Heat conduction
    "II.8.7",   # Electrostatic energy
    "II.8.31",  # Electric field energy density
    "II.10.9",  # Polarization
    "II.11.3",  # Driven oscillator amplitude
    "II.11.20", # Polarization density
    "II.11.27", # Clausius-Mossotti
    "II.11.28", # Dielectric constant
    "II.21.32", # Retarded potential
    "II.34.2",  # Magnetic moment
    "II.34.11", # Gyromagnetic ratio
    "II.34.29b", # Zeeman energy
    "II.36.38", # Curie-Weiss (linear part)
    "II.37.1",  # Magnetic susceptibility
    "II.38.3",  # Hooke's law
    "II.38.14", # Shear modulus
    "III.13.18", # de Broglie velocity
    "III.15.27", # Bragg condition
    "III.19.51", # Hydrogen energy levels
]


if __name__ == "__main__":
    print_exclusion_summary()
    
    print("\n=== Excluded Equations by Category ===\n")
    
    print("DEGENERATE:")
    for eq_id in sorted(DEGENERATE_EQUATIONS.keys()):
        eq = DEGENERATE_EQUATIONS[eq_id]
        print(f"  {eq_id}: {eq.notes}")
    
    print("\nTRANSCENDENTAL:")
    for eq_id in sorted(TRANSCENDENTAL_EQUATIONS.keys()):
        eq = TRANSCENDENTAL_EQUATIONS[eq_id]
        print(f"  {eq_id}: {eq.notes}")
    
    print("\nSQRT_ONLY:")
    for eq_id in sorted(SQRT_ONLY_EQUATIONS.keys()):
        eq = SQRT_ONLY_EQUATIONS[eq_id]
        print(f"  {eq_id}: {eq.notes}")
