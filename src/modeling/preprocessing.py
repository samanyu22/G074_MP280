"""Reusable preprocessing and temporal splitting for MP280."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = (
    "route_km",
    "stops",
    "passengers",
    "hour",
    "temp_c",
    "driver_exp_yr",
    "weather",
    "depot_zone",
)
NUMERIC_FEATURES = (
    "route_km",
    "stops",
    "passengers",
    "hour",
    "temp_c",
    "driver_exp_yr",
)
CATEGORICAL_FEATURES = ("weather", "depot_zone")
TARGET_COLUMN = "delay_min"


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the eight features allowed by the client's prediction schema."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    return frame.loc[:, FEATURE_COLUMNS].copy()


def build_preprocessor() -> ColumnTransformer:
    """Create numeric and categorical transformations fitted only on training data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def build_pipeline(regressor) -> Pipeline:
    """Combine preprocessing and a scikit-learn regression estimator."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def chronological_split(
    frame: pd.DataFrame,
    validation_fraction: float = 0.20,
    date_column: str = "date",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Reserve the latest dates for validation; never split a date across sets."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    required = [date_column, TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns needed for splitting: {missing}")

    ordered = frame.copy()
    dates = pd.to_datetime(ordered[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Column '{date_column}' contains invalid or missing dates")

    unique_dates = pd.Index(dates.unique()).sort_values()
    train_date_count = int(len(unique_dates) * (1.0 - validation_fraction))
    if train_date_count < 1 or train_date_count >= len(unique_dates):
        raise ValueError("Not enough distinct dates for the requested time split")

    cutoff = unique_dates[train_date_count]
    train_mask = dates < cutoff
    valid_mask = ~train_mask
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("The chronological split produced an empty partition")

    x_train = select_features(ordered.loc[train_mask])
    x_valid = select_features(ordered.loc[valid_mask])
    y_train = pd.to_numeric(ordered.loc[train_mask, TARGET_COLUMN], errors="raise")
    y_valid = pd.to_numeric(ordered.loc[valid_mask, TARGET_COLUMN], errors="raise")
    return x_train, y_train, x_valid, y_valid
