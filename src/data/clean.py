"""Data cleaning pipeline for MP280 Bus Delay Prediction.

Ingests raw training data (data/raw/train.csv), resolves the data quality issues
identified during exploratory diagnostics, and outputs clean data for model training
(data/processed/train_clean.csv).

Usage:
    python src/data/clean.py
"""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import yaml


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration and thresholds from params.yaml."""
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {params_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows from the dataset."""
    initial_rows = len(df)
    df_dedup = df.drop_duplicates().copy()
    dropped = initial_rows - len(df_dedup)
    print(f"[Deduplication] Removed {dropped} duplicate rows ({initial_rows} -> {len(df_dedup)}).")
    return df_dedup


def fix_target_typos(
    df: pd.DataFrame,
    target_col: str = "delay_min",
    threshold: float = 100.0,
    factor: float = 10.0,
) -> pd.DataFrame:
    """Correct target values typed with an extra zero (divided by factor)."""
    df = df.copy()
    if target_col in df.columns:
        typo_mask = df[target_col] >= threshold
        typo_count = int(typo_mask.sum())
        if typo_count > 0:
            df.loc[typo_mask, target_col] = df.loc[typo_mask, target_col] / factor
            print(f"[Target Typos] Corrected {typo_count} inflated '{target_col}' entries (divided by {factor}).")
    return df


def normalize_categories(
    df: pd.DataFrame,
    column: str = "weather",
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Normalize text column: lowercase, strip whitespace, and map abbreviations to canonical forms."""
    df = df.copy()
    if column in df.columns:
        if mapping is None:
            mapping = {"cle.": "clear", "rai.": "rain", "fog.": "fog"}

        initial_levels = df[column].dropna().nunique()
        # Lowercase and trim
        df[column] = df[column].astype(str).str.lower().str.strip()
        # Map abbreviations
        df[column] = df[column].replace(mapping)
        final_levels = df[column].nunique()
        print(f"[Categories] Normalized '{column}' from {initial_levels} raw variants to {final_levels} canonical levels ({sorted(df[column].unique().tolist())}).")
    return df


def handle_sentinels(
    df: pd.DataFrame,
    column: str = "passengers",
    sentinel_value: int = -1,
) -> pd.DataFrame:
    """Replace sentinel values with NaN and impute with median of valid records."""
    df = df.copy()
    if column in df.columns:
        sentinel_mask = df[column] == sentinel_value
        sentinel_count = int(sentinel_mask.sum())
        if sentinel_count > 0:
            median_val = int(round(df.loc[~sentinel_mask, column].median()))
            df.loc[sentinel_mask, column] = median_val
            df[column] = df[column].astype(int)
            print(f"[Sentinel Value] Replaced {sentinel_count} '{sentinel_value}' sentinels in '{column}' with median ({median_val}).")
    return df


def handle_stuck_sensor(
    df: pd.DataFrame,
    column: str = "driver_exp_yr",
    stuck_value: int = 20,
    start_date: str = "2026-04-03",
    end_date: str = "2026-04-23",
    date_col: str = "date",
) -> pd.DataFrame:
    """Handle stuck sensor readings during the anomalous date window by imputing with valid median."""
    df = df.copy()
    if column in df.columns and date_col in df.columns:
        stuck_mask = (
            (df[date_col] >= start_date)
            & (df[date_col] <= end_date)
            & (df[column] == stuck_value)
        )
        stuck_count = int(stuck_mask.sum())
        if stuck_count > 0:
            # Impute using median of non-stuck records
            valid_median = int(round(df.loc[~stuck_mask, column].median()))
            df.loc[stuck_mask, column] = valid_median
            df[column] = df[column].astype(int)
            print(f"[Stuck Sensor] Imputed {stuck_count} stuck readings in '{column}' between {start_date} and {end_date} with median ({valid_median}).")
    return df


def clean_dataframe(
    df: pd.DataFrame, params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Apply all cleaning transformations to an in-memory DataFrame."""
    if params is None:
        try:
            params = load_params()
        except Exception:
            params = {}

    clean_cfg = params.get("cleaning", {})
    diag_cfg = params.get("diagnostics", {})

    df_clean = drop_duplicates(df)

    df_clean = fix_target_typos(
        df_clean,
        target_col=diag_cfg.get("target_column", "delay_min"),
        threshold=clean_cfg.get("target_typo_threshold", 100.0),
        factor=clean_cfg.get("target_typo_factor", 10.0),
    )

    df_clean = normalize_categories(
        df_clean,
        column="weather",
        mapping=clean_cfg.get("weather_mapping", {"cle.": "clear", "rai.": "rain", "fog.": "fog"}),
    )

    df_clean = handle_sentinels(
        df_clean,
        column=clean_cfg.get("sentinel_column", "passengers"),
        sentinel_value=clean_cfg.get("sentinel_value", -1),
    )

    df_clean = handle_stuck_sensor(
        df_clean,
        column=clean_cfg.get("stuck_sensor_column", "driver_exp_yr"),
        stuck_value=clean_cfg.get("stuck_sensor_value", 20),
        start_date=clean_cfg.get("stuck_sensor_start_date", "2026-04-03"),
        end_date=clean_cfg.get("stuck_sensor_end_date", "2026-04-23"),
        date_col=diag_cfg.get("date_column", "date"),
    )

    return df_clean


def clean_train_data(params_path: str = "params.yaml") -> pd.DataFrame:
    """Execute complete data cleaning pipeline and save processed dataset."""
    params = load_params(params_path)
    data_cfg = params.get("data", {})

    raw_path = Path(data_cfg.get("raw_train_path", "data/raw/train.csv"))
    if not raw_path.exists():
        raw_path = Path("data/train.csv")

    output_path = Path(data_cfg.get("processed_train_path", "data/processed/train_clean.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    df_clean = clean_dataframe(df, params=params)

    # Save cleaned data
    df_clean.to_csv(output_path, index=False)
    print(f"\nSuccessfully cleaned dataset saved to: {output_path}")
    print(f"Final shape: {df_clean.shape} (Rows: {len(df_clean)}, Columns: {len(df_clean.columns)})")
    return df_clean


if __name__ == "__main__":
    clean_train_data()
