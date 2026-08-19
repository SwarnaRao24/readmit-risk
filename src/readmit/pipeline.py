"""Build the preprocessing + model pipeline and the train/test split.

Key principle: preprocessing is bundled into a sklearn Pipeline so it is
ONLY ever fit on training data (and re-fit inside each CV fold), preventing
any leakage from the test set into the model.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from readmit.clean import load_clean
from readmit.features import engineer


def load_modelling_data():
    """Full pipeline: load -> clean -> engineer -> return X, y."""
    df = engineer(load_clean())
    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y


def make_split(X, y, test_size=0.2, seed=42):
    """Stratified train/test split (preserves the ~9% positive rate in both)."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric features, one-hot encode categoricals.

    Wrapped in a ColumnTransformer so it slots into a Pipeline and is fit
    on training data only.
    """
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical),
        ]
    )
