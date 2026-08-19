"""Feature engineering: group high-cardinality ICD-9 diagnosis codes.

The diag_1/2/3 columns hold ~700 distinct ICD-9 codes each. One-hot encoding
them directly would create ~2000 sparse columns. Instead we group them into
9 clinical categories following the standard mapping (Strack et al., 2014),
turning them into interpretable, low-cardinality features.
"""
from __future__ import annotations

import pandas as pd

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

# Columns that are constant (1 unique value) -> zero information, drop them
CONSTANT_COLS = ["examide", "citoglipton", "glimepiride-pioglitazone"]


def _map_icd9(code: str) -> str:
    """Map a single ICD-9 code (first 3 digits) to a clinical category."""
    if code is None or code == "Unknown":
        return "Unknown"
    code = str(code)

    # V and E codes are supplementary -> Other
    if code.startswith("V") or code.startswith("E"):
        return "Other"

    # Diabetes: 250.xx
    if code.startswith("250"):
        return "Diabetes"

    try:
        num = float(code)
    except ValueError:
        return "Other"

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    if 460 <= num <= 519 or num == 786:
        return "Respiratory"
    if 520 <= num <= 579 or num == 787:
        return "Digestive"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Group diagnosis codes into clinical categories; drop constant columns."""
    df = df.copy()

    for col in DIAG_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_map_icd9)

    df = df.drop(columns=[c for c in CONSTANT_COLS if c in df.columns])

    return df
