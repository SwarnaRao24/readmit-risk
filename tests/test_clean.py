"""Tests for the cleaning pipeline."""
import pandas as pd

from readmit.clean import clean


def _raw_sample() -> pd.DataFrame:
    # Two patients: one with a repeat visit, plus missing values to handle
    return pd.DataFrame(
        {
            "patient_nbr": [1, 1, 2],
            "readmitted": ["<30", "NO", ">30"],
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
    # Patient 1's first visit was "<30" -> 1; patient 2 ">30" -> 0
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
    assert "readmitted" not in out.columns  # raw target dropped; engineered 'target' kept
    assert "target" in out.columns