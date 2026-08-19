"""Tests for ICD-9 diagnosis grouping."""
import pandas as pd

from readmit.features import _map_icd9, engineer


def test_diabetes_codes():
    assert _map_icd9("250") == "Diabetes"
    assert _map_icd9("250.83") == "Diabetes"


def test_circulatory_range():
    assert _map_icd9("410") == "Circulatory"   # heart attack
    assert _map_icd9("428") == "Circulatory"   # heart failure
    assert _map_icd9("785") == "Circulatory"   # special-case code


def test_respiratory_range():
    assert _map_icd9("486") == "Respiratory"    # pneumonia
    assert _map_icd9("786") == "Respiratory"


def test_v_and_e_codes_are_other():
    assert _map_icd9("V45") == "Other"
    assert _map_icd9("E885") == "Other"


def test_unknown_passthrough():
    assert _map_icd9("Unknown") == "Unknown"
    assert _map_icd9(None) == "Unknown"


def test_out_of_range_is_other():
    assert _map_icd9("001") == "Other"   # infectious disease, not in our 9 groups
    assert _map_icd9("300") == "Other"   # mental disorder


def test_engineer_drops_constant_columns():
    df = pd.DataFrame({
        "diag_1": ["250", "410"],
        "diag_2": ["486", "V45"],
        "diag_3": ["428", "780"],
        "examide": ["No", "No"],
        "citoglipton": ["No", "No"],
    })
    out = engineer(df)
    assert "examide" not in out.columns
    assert "citoglipton" not in out.columns
    assert out["diag_1"].tolist() == ["Diabetes", "Circulatory"]
