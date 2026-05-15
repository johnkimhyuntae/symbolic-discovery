from __future__ import annotations

import dataclasses

import pytest
import yaml

from symbolic_discovery.solvers import SOLVER_REGISTRY
from symbolic_discovery.experiments.plan import (
    Run,
    Variant,
    expand_to_runs,
    load_study_file,
    parse_sweep_spec,
    parse_variant_spec,
    variants_from_study,
    _coerce,
    _parse_kv_list,
)


KNOWN_MODEL = next(iter(SOLVER_REGISTRY))
UNKNOWN_MODEL = "cambridge_fake_model_name"


# Variant

class TestVariant:
    def test_basic_construction(self):
        v = Variant(name="x", model=KNOWN_MODEL)
        assert v.name == "x"
        assert v.model == KNOWN_MODEL
        assert v.params == {}

    def test_with_params(self):
        v = Variant(name="x", model=KNOWN_MODEL, params={"k": 1})
        assert v.params == {"k": 1}

    def test_is_frozen(self):
        v = Variant(name="x", model=KNOWN_MODEL)
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.name = "y"  # type: ignore[misc]

    def test_default_params_not_shared(self):
        v1 = Variant(name="a", model=KNOWN_MODEL)
        v2 = Variant(name="b", model=KNOWN_MODEL)
        assert v1.params is not v2.params

    def test_validate_known_model_does_not_raise(self):
        Variant(name="x", model=KNOWN_MODEL).validate()

    def test_validate_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            Variant(name="x", model=UNKNOWN_MODEL).validate()

    def test_validate_error_message_includes_variant_name(self):
        with pytest.raises(ValueError, match="my_variant"):
            Variant(name="my_variant", model=UNKNOWN_MODEL).validate()


# Run

class TestRun:
    def _v(self) -> Variant:
        return Variant(name="x", model=KNOWN_MODEL)

    def test_basic_construction(self):
        r = Run(
            variant=self._v(), dataset="S1", noise=0.0,
            noise_type="multiplicative", n_samples=100, seed=73,
        )
        assert r.dataset == "S1"
        assert r.seed == 73
        assert r.variant.model == KNOWN_MODEL

    def test_is_frozen(self):
        r = Run(
            variant=self._v(), dataset="S1", noise=0.0,
            noise_type="multiplicative", n_samples=100, seed=73,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.seed = 99  # type: ignore[misc]


# parse_variant_spec

class TestParseVariantSpec:
    def test_no_params(self):
        v = parse_variant_spec(f"baseline={KNOWN_MODEL}")
        assert v.name == "baseline"
        assert v.model == KNOWN_MODEL
        assert v.params == {}

    def test_single_int_param(self):
        v = parse_variant_spec(f"v={KNOWN_MODEL}:n_folds=3")
        assert v.params == {"n_folds": 3}

    def test_multiple_params_mixed_types(self):
        v = parse_variant_spec(f"v={KNOWN_MODEL}:a=1,b=2.5,c=true")
        assert v.params == {"a": 1, "b": 2.5, "c": True}

    def test_param_value_coercions(self):
        v = parse_variant_spec(
            f"v={KNOWN_MODEL}:i=7,f=3.14,t=true,n=none,s=hello"
        )
        assert v.params == {
            "i": 7, "f": 3.14, "t": True, "n": None, "s": "hello",
        }

    def test_strips_surrounding_whitespace(self):
        v = parse_variant_spec(f"  baseline = {KNOWN_MODEL} ")
        assert v.name == "baseline"
        assert v.model == KNOWN_MODEL

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="--variant"):
            parse_variant_spec(f"baseline_{KNOWN_MODEL}")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_variant_spec(f"={KNOWN_MODEL}")

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_variant_spec("baseline=")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            parse_variant_spec(f"baseline={UNKNOWN_MODEL}")

    def test_malformed_param_raises(self):
        with pytest.raises(ValueError, match="k=v"):
            parse_variant_spec(f"v={KNOWN_MODEL}:n_folds")


# parse_sweep_spec

class TestParseSweepSpec:
    def test_int_sweep(self):
        variants = parse_sweep_spec(f"{KNOWN_MODEL}.n_folds=1,3,5")
        assert len(variants) == 3
        assert all(v.model == KNOWN_MODEL for v in variants)
        assert [v.params["n_folds"] for v in variants] == [1, 3, 5]

    def test_sweep_names_are_distinct_and_descriptive(self):
        variants = parse_sweep_spec(f"{KNOWN_MODEL}.k=1,2,3")
        names = [v.name for v in variants]
        assert len(set(names)) == 3, "sweep variant names must be unique"
        assert all(KNOWN_MODEL in n for n in names)
        assert all("k" in n for n in names)

    def test_float_sweep(self):
        variants = parse_sweep_spec(f"{KNOWN_MODEL}.scale=0.5,1.0,2.0")
        assert [v.params["scale"] for v in variants] == [0.5, 1.0, 2.0]

    def test_bool_sweep(self):
        variants = parse_sweep_spec(f"{KNOWN_MODEL}.flag=true,false")
        assert [v.params["flag"] for v in variants] == [True, False]

    def test_single_value_sweep_produces_one_variant(self):
        variants = parse_sweep_spec(f"{KNOWN_MODEL}.k=67")
        assert len(variants) == 1
        assert variants[0].params == {"k": 67}

    def test_missing_param_raises(self):
        with pytest.raises(ValueError, match="--sweep"):
            parse_sweep_spec(f"{KNOWN_MODEL}=1,2,3")

    def test_missing_values_raises(self):
        with pytest.raises(ValueError, match="--sweep"):
            parse_sweep_spec(f"{KNOWN_MODEL}.k=")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            parse_sweep_spec(f"{UNKNOWN_MODEL}.k=1,2,3")


# load_study_file

class TestLoadStudyFile:
    def test_loads_valid_yaml(self, tmp_path):
        path = tmp_path / "study.yaml"
        path.write_text(yaml.safe_dump({
            "datasets": ["S1", "S2"],
            "seeds": [73, 74],
        }))
        cfg = load_study_file(str(path))
        assert cfg["datasets"] == ["S1", "S2"]
        assert cfg["seeds"] == [73, 74]

    def test_empty_file_returns_empty_dict(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert load_study_file(str(path)) == {}

    def test_top_level_list_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ValueError, match="mapping"):
            load_study_file(str(path))

    def test_top_level_scalar_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("just a string\n")
        with pytest.raises(ValueError, match="mapping"):
            load_study_file(str(path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_study_file(str(tmp_path / "does_not_exist.yaml"))


# variants_from_study

class TestVariantsFromStudy:
    def test_empty_study_returns_empty_list(self):
        assert variants_from_study({}) == []

    def test_no_variants_key_returns_empty_list(self):
        assert variants_from_study({"datasets": ["S1"]}) == []

    def test_extracts_single_variant(self):
        cfg = {"variants": [{"name": "v1", "model": KNOWN_MODEL}]}
        out = variants_from_study(cfg)
        assert len(out) == 1
        assert out[0].name == "v1"
        assert out[0].params == {}

    def test_extracts_multiple_variants_with_params(self):
        cfg = {"variants": [
            {"name": "a", "model": KNOWN_MODEL, "params": {"k": 1}},
            {"name": "b", "model": KNOWN_MODEL, "params": {"k": 2}},
        ]}
        out = variants_from_study(cfg)
        assert [v.name for v in out] == ["a", "b"]
        assert [v.params["k"] for v in out] == [1, 2]

    def test_validates_model(self):
        cfg = {"variants": [{"name": "v1", "model": UNKNOWN_MODEL}]}
        with pytest.raises(ValueError, match="Unknown model"):
            variants_from_study(cfg)


# expand_to_runs

class TestExpandToRuns:
    def _v(self, name: str = "x") -> Variant:
        return Variant(name=name, model=KNOWN_MODEL)

    def test_cardinality(self):
        runs = expand_to_runs(
            variants=[self._v("a"), self._v("b")],
            datasets=["S1", "S2"],
            noise=[0.0, 0.1],
            noise_types=["multiplicative"],
            n_samples=[100],
            seeds=[1, 2, 3],
        )
        assert len(runs) == 2 * 2 * 2 * 1 * 1 * 3  # 24

    def test_returns_run_instances(self):
        runs = expand_to_runs(
            variants=[self._v()], datasets=["S1"], noise=[0.0],
            noise_types=["multiplicative"], n_samples=[100], seeds=[73],
        )
        assert len(runs) == 1
        assert isinstance(runs[0], Run)

    def test_empty_axis_returns_empty(self):
        runs = expand_to_runs(
            variants=[self._v()], datasets=[], noise=[0.0],
            noise_types=["multiplicative"], n_samples=[100], seeds=[73],
        )
        assert runs == []

    def test_variants_outermost_in_iteration_order(self):
        runs = expand_to_runs(
            variants=[self._v("a"), self._v("b")],
            datasets=["S1", "S2"],
            noise=[0.0],
            noise_types=["multiplicative"],
            n_samples=[100],
            seeds=[73],
        )
        names_in_order = [r.variant.name for r in runs]
        assert names_in_order == ["a", "a", "b", "b"]

    def test_fields_propagate_correctly(self):
        runs = expand_to_runs(
            variants=[self._v("v1")], datasets=["S1"], noise=[0.05],
            noise_types=["additive"], n_samples=[250], seeds=[123],
        )
        r = runs[0]
        assert r.variant.name == "v1"
        assert r.dataset == "S1"
        assert r.noise == 0.05
        assert r.noise_type == "additive"
        assert r.n_samples == 250
        assert r.seed == 123


# _parse_kv_list

class TestParseKvList:
    def test_single_pair(self):
        assert _parse_kv_list("k=1") == {"k": 1}

    def test_multiple_pairs(self):
        assert _parse_kv_list("k1=1,k2=2.0,k3=true") == {
            "k1": 1, "k2": 2.0, "k3": True,
        }

    def test_strips_whitespace(self):
        assert _parse_kv_list(" k1 = 1 , k2 = 2 ") == {"k1": 1, "k2": 2}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="k=v"):
            _parse_kv_list("just_a_key")


# _coerce

class TestCoerce:
    @pytest.mark.parametrize("inp,expected", [
        ("true", True), ("True", True), ("TRUE", True),
        ("false", False), ("False", False),
    ])
    def test_bool(self, inp, expected):
        assert _coerce(inp) is expected

    @pytest.mark.parametrize("inp", ["none", "None", "null", "NULL"])
    def test_none(self, inp):
        assert _coerce(inp) is None

    @pytest.mark.parametrize("inp,expected", [
        ("0", 0), ("1", 1), ("-7", -7), ("12345", 12345),
    ])
    def test_int(self, inp, expected):
        result = _coerce(inp)
        assert isinstance(result, int) and not isinstance(result, bool)
        assert result == expected

    @pytest.mark.parametrize("inp,expected", [
        ("0.0", 0.0), ("3.14", 3.14), ("-2.5", -2.5), ("1e3", 1000.0),
    ])
    def test_float(self, inp, expected):
        result = _coerce(inp)
        assert isinstance(result, float)
        assert result == expected
        
