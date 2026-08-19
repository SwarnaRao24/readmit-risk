"""Data loading and cleaning for the readmission dataset.

Encodes the cleaning decisions made during exploration:
- restrict to each patient's FIRST encounter (independence / no leakage)
- remove encounters ending in death or hospice (readmission undefined)
- drop near-empty columns (weight, max_glu_serum)
- treat informative missingness as its own category (A1Cresult, specialty, payer)
- binary target: readmitted within 30 days = 1
"""
from __future__ import annotations

import pandas as pd
from ucimlrepo import fetch_ucirepo

# Columns too empty to use (>94% missing)
DROP_COLS = ["weight", "max_glu_serum"]

# Missing here is informative -> fill with an explicit category, don't impute/drop
MISSING_AS_CATEGORY = {
    "A1Cresult": "not_measured",
    "medical_specialty": "Unknown",
    "payer_code": "Unknown",
    "race": "Unknown",
    "diag_1": "Unknown",
    "diag_2": "Unknown",
    "diag_3": "Unknown",
}

# Discharge dispositions meaning the patient died or entered hospice:
# readmission is undefined for these, so the rows are removed.
EXPIRED_HOSPICE = [11, 13, 14, 19, 20, 21]


def load_raw() -> pd.DataFrame:
    """Fetch the UCI dataset and assemble features + ids + target into one frame."""
    ds = fetch_ucirepo(id=296)
    df = ds.data.features.copy()
    df["patient_nbr"] = ds.data.ids["patient_nbr"].values
    df["readmitted"] = ds.data.targets["readmitted"].values
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning decisions. Returns a modelling-ready frame."""
    df = df.copy()

    # 1. Binary target: readmitted within 30 days
    df["target"] = (df["readmitted"] == "<30").astype(int)

    # 2. First encounter per patient only (independence, no patient leakage)
    df = df.sort_index().drop_duplicates(subset="patient_nbr", keep="first")

    # 2b. Remove encounters ending in death or hospice (readmission undefined)
    if "discharge_disposition_id" in df.columns:
        df = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE)]

    # 3. Drop near-empty columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 4. Informative missingness -> explicit category
    for col, fill in MISSING_AS_CATEGORY.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill)

    # 5. Drop identifiers and the raw target (keep engineered 'target')
    df = df.drop(columns=[c for c in ["patient_nbr", "readmitted"] if c in df.columns])

    return df.reset_index(drop=True)


def load_clean() -> pd.DataFrame:
    """Convenience: load and clean in one call."""
    return clean(load_raw())
