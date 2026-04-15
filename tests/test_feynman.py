import pytest
from pathlib import Path
from symbolic_discovery.data import feynman


FEYNMAN_ROOT = "feynman"

FEYNMAN_HEADER = "Filename,Number,Output,Formula,# variables,v1_name,v1_low,v1_high,v2_name,v2_low,v2_high,v3_name,v3_low,v3_high,v4_name,v4_low,v4_high,v5_name,v5_low,v5_high,v6_name,v6_low,v6_high,v7_name,v7_low,v7_high,v8_name,v8_low,v8_high,v9_name,v9_low,v9_high,v10_name,v10_low,v10_high"

BONUS_HEADER = "Filename,Number,Name,Eqn. No.,Output,Formula,# variables,v1_name,v1_low,v1_high,v2_name,v2_low,v2_high,v3_name,v3_low,v3_high,v4_name,v4_low,v4_high,v5_name,v5_low,v5_high,v6_name,v6_low,v6_high,v7_name,v7_low,v7_high,v8_name,v8_low,v8_high,v9_name,v9_low,v9_high,v10_name,v10_low,v10_high"

# Directory structure

@pytest.mark.skipif(
    not Path(FEYNMAN_ROOT).exists(),
    reason="Feynman data directory not present"
)
def test_feynman_with_units_exists():
    assert (Path(FEYNMAN_ROOT) / "Feynman_with_units").exists()


@pytest.mark.skipif(
    not Path(FEYNMAN_ROOT).exists(),
    reason="Feynman data directory not present"
)
def test_bonus_with_units_exists():
    assert (Path(FEYNMAN_ROOT) / "bonus_with_units").exists()


# Metadata CSV header

@pytest.mark.skipif(
    not Path(FEYNMAN_ROOT).exists(),
    reason="Feynman data directory not present"
)
def test_feynman_csv_header():
    csv_path = Path(FEYNMAN_ROOT) / "FeynmanEquations.csv"
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()
    assert first_line == FEYNMAN_HEADER

def test_bonus_csv_header():
    csv_path = Path(FEYNMAN_ROOT) / "BonusEquations.csv"
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()
    assert first_line == BONUS_HEADER


# Metadata loading

@pytest.mark.skipif(
    not Path(FEYNMAN_ROOT).exists(),
    reason="Feynman data directory not present"
)
def test_load_metadata_feynman():
    meta = feynman.load_metadata(FEYNMAN_ROOT, "F")
    assert len(meta) > 0
    assert "Filename" in meta.columns


@pytest.mark.skipif(
    not Path(FEYNMAN_ROOT).exists(),
    reason="Feynman data directory not present"
)
def test_load_metadata_bonus():
    meta = feynman.load_metadata(FEYNMAN_ROOT, "B")
    assert len(meta) > 0
    assert "Filename" in meta.columns


def test_load_metadata_invalid_family():
    with pytest.raises(ValueError):
        feynman.load_metadata(FEYNMAN_ROOT, "X")
