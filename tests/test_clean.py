"""Tests for the cleaning pipeline."""
import pandas as pd

from readmit.clean import clean


def _raw_sample() -> pd.DataFrame:
    # Two patients: one with a repeat visit, plus missing values to handle
    return pd.DataFrame(
        {
            "patient_nbr": [1, 1, 2],
            "readmitted": ["<30", "NO", ">30"],
            "discharge_disposition_id": [1, 1, 1],
            "weight": [None, None, None],
            "max_glu_serum": [None, None, None],
            "A1Cresult": [None, None, ">7"],
            "medical_specialty": [None, "Cardiology", None],
            "payer_code": [None, None, "MC"],
            "race": [None, "Caucasian", "AfricanAmerican"],
            "diag_1": ["250", "401", None],
            "diag_2": ["428", None, "250"],
            "diag_3": [None, "276", "427"],
            "age": ["[50-60)", "[50-60)", "[60-70)"],
        }
    )


def test_first_encounter_only():
    out = clean(_raw_sample())
    # Patient 1 had 2 visits -> only first kept; patient 2 -> 1. Total 2 rows.
    assert len(out) == 2


def test_binary_target():
    out = clean(_raw_sample())
    assert set(out["target"].unique()) <= {0, 1}
    assert out["target"].iloc[0] == 1


def test_empty_columns_dropped():
    out = clean(_raw_sample())
    assert "weight" not in out.columns
    assert "max_glu_serum" not in out.columns


def test_no_missing_remains():
    out = clean(_raw_sample())
    assert out.isna().sum().sum() == 0


def test_identifiers_removed():
    out = clean(_raw_sample())
    assert "patient_nbr" not in out.columns
    assert "readmitted" not in out.columns
    assert "target" in out.columns


def test_expired_hospice_removed():
    raw = _raw_sample()
    raw["discharge_disposition_id"] = [1, 1, 11]  # patient 2's encounter = expired
    out = clean(raw)
    # Patient 2's only encounter ended in death -> removed entirely; only patient 1 remains
    assert len(out) == 1
    assert 11 not in list(out["discharge_disposition_id"])
