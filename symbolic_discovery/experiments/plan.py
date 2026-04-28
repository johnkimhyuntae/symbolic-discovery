"""
Experiment plan resolution.

Defines the Variant and Run dataclasses, and the parsers that turn CLI
flags (--variant, --sweep) and YAML study files (--study) into a flat 
list of concrete Run objects for the runner to execute.

A Variant is a named (model, kwargs) pair. A Run is one cell in the
experiment grid: a Variant combined with a dataset, noise level,
noise type, sample size, and seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import itertools
import yaml

from ..solvers import SOLVER_REGISTRY


# Dataclasses

@dataclass(frozen=True)
class Variant:
    """
    A named instantiation of a solver with concrete kwargs.

    Attributes:
        name: Human-readable label used in CSV output and viewer
            grouping (e.g. "no_voting", "bacon7f_default").
        model: Solver registry key (e.g. "bacon3f", "bacon7f", "pysr").
        params: kwargs forwarded to the solver wrapper's __init__.
    """
    name: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError if the model is not in the registry."""
        if self.model not in SOLVER_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model}' in variant '{self.name}'. "
                f"Available: {sorted(SOLVER_REGISTRY.keys())}"
            )


@dataclass(frozen=True)
class Run:
    """One concrete cell in the experiment grid."""
    variant: Variant
    dataset: str
    noise: float
    noise_type: str
    n_samples: int
    seed: int


# CLI spec parsers

def parse_variant_spec(spec: str) -> Variant:
    """
    Parse a --variant SPEC string into a Variant.

    Format: 'name=model[:k=v,k=v,...]'
    Example: 'no_voting=bacon7f:n_folds=1,scale_factor=1.0'
    """
    if "=" not in spec:
        raise ValueError(
            f"--variant expects 'name=model[:k=v,...]', got: {spec!r}"
        )
    name, _, rhs = spec.partition("=")
    model, _, kvs = rhs.partition(":")
    name, model = name.strip(), model.strip()
    if not name or not model:
        raise ValueError(f"--variant name and model must be non-empty: {spec!r}")
    params = _parse_kv_list(kvs) if kvs else {}
    variant = Variant(name=name, model=model, params=params)
    variant.validate()
    return variant


def parse_sweep_spec(spec: str) -> list[Variant]:
    """
    Parse a --sweep SPEC string into one Variant per swept value.

    Format: 'model.param=v1,v2,...'
    Example: 'bacon7f.n_folds=1,3,5,7' produces 4 Variants named
    'bacon7f_n_folds=1', 'bacon7f_n_folds=3', etc.
    """
    lhs, _, rhs = spec.partition("=")
    model, _, param = lhs.partition(".")
    model, param = model.strip(), param.strip()
    if not (model and param and rhs):
        raise ValueError(
            f"--sweep expects 'model.param=v1,v2,...', got: {spec!r}"
        )
    variants = []
    for raw in rhs.split(","):
        value = _coerce(raw.strip())
        v = Variant(
            name=f"{model}_{param}_{raw.strip()}",
            model=model,
            params={param: value},
        )
        v.validate()
        variants.append(v)
    return variants


# Study file loading

def load_study_file(path: str) -> dict:
    """Load a YAML study file describing a full experiment grid."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Study file {path!r} must be a YAML mapping at the top level")
    return data


def variants_from_study(study: dict) -> list[Variant]:
    """
    Extract a list of Variants from a parsed study dict.

    Each entry under the 'variants' key must be a mapping with at least
    'name' and 'model' fields, and an optional 'params' mapping.
    """
    out = []
    for entry in study.get("variants", []):
        v = Variant(
            name=entry["name"],
            model=entry["model"],
            params=dict(entry.get("params", {})),
        )
        v.validate()
        out.append(v)
    return out


# Grid expansion

def expand_to_runs(
    *,
    variants: Iterable[Variant],
    datasets: Iterable[str],
    noise: Iterable[float],
    noise_types: Iterable[str],
    n_samples: Iterable[int],
    seeds: Iterable[int],
) -> list[Run]:
    """Cartesian product of all axes into a flat list of Runs."""
    return [
        Run(variant=v, dataset=d, noise=ns, noise_type=nt,
            n_samples=n, seed=s)
        for v, d, ns, nt, n, s in itertools.product(
            variants, datasets, noise, noise_types, n_samples, seeds
        )
    ]


# Helpers

def _parse_kv_list(s: str) -> dict[str, Any]:
    """Parse 'k1=v1,k2=v2,...' into a dict, coercing values."""
    out = {}
    for kv in s.split(","):
        k, sep, v = kv.partition("=")
        if not sep:
            raise ValueError(f"Expected 'k=v' in variant params, got: {kv!r}")
        out[k.strip()] = _coerce(v.strip())
    return out


def _coerce(v: str) -> Any:
    """Best-effort coercion: bool -> None -> int -> float -> str."""
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if v.lower() in {"none", "null"}:
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v