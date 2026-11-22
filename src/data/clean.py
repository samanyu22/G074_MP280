"""Data cleaning pipeline for MP280 Bus Delay Prediction.

Reads the original training file, resolves the verified data-quality issues,
and writes the processed dataset used for v1 model training.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load project settings with yaml.safe_load."""
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {params_path}")
    with path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    if not isinstance(params, dict):
        raise ValueError("params.yaml must contain a YAML mapping")
    return params


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (including only extra copies)."""
    before = len(df)
    cleaned = df.drop_duplicates().copy()
    print(f"[Deduplication] Removed {before - len(cleaned)} duplicate rows ({before} -> {len(cleaned)}).")
    return cleaned


def fix_target_typos(
    df: pd.DataFrame,
    target_col: str = "delay_min",
    threshold: float = 100.0,
    factor: float = 10.0,
) -> pd.DataFrame:
    """Fix target values typed ten times too large."""
    if factor <= 0:
        raise ValueError("Target typo correction factor must be positive")
    result = df.copy()
    if target_col in result.columns:
        typo_mask = pd.to_numeric(result[target_col], errors="coerce") >= threshold
        count = int(typo_mask.sum())
        result.loc[typo_mask, target_col] = result.loc[typo_mask, target_col] / factor
        if count:
            print(f"[Target Typos] Corrected {count} '{target_col}' values by dividing by {factor}.")
    return result


def normalize_categories(
    df: pd.DataFrame,
    column: str = "weather",
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Normalize case/spacing and abbreviations while preserving missing values."""
    result = df.copy()
    if column not in result.columns:
        return result

    if mapping is None:
        mapping = {"cle.": "clear", "rai.": "rain", "fog.": "fog"}

    raw_levels = int(result[column].nunique(dropna=True))
    normalized_mapping = {
        str(key).lower().strip(): str(value).lower().strip()
        for key, value in mapping.items()
    }

    # Use an empty temporary token for nulls, then restore them as np.nan.
    values = result[column].where(result[column].notna(), "").astype(str)
    values = values.str.lower().str.strip().replace(normalized_mapping)
    result[column] = values.replace("", np.nan)

    final_levels = int(result[column].nunique(dropna=True))
    levels = sorted(result[column].dropna().astype(str).unique().tolist())
    print(f"[Categories] Normalized '{column}' from {raw_levels} raw variants to {final_levels}: {levels}.")
    return result


def handle_sentinels(
    df: pd.DataFrame,
    column: str = "passengers",
    sentinel_value: int = -1,
) -> pd.DataFrame:
    """Replace the configured sentinel with the median of valid observations."""
    result = df.copy()
    if column not in result.columns:
        return result

    mask = result[column] == sentinel_value
    count = int(mask.sum())
    if count:
        valid = pd.to_numeric(result.loc[~mask, column], errors="coerce").dropna()
        if valid.empty:
            raise ValueError(f"Cannot impute {column}: no valid values remain")
        median_value = int(round(float(valid.median())))
        result.loc[mask, column] = median_value
        result[column] = result[column].astype(int)
        print(f"[Sentinel Value] Replaced {count} '{sentinel_value}' values in '{column}' with median {median_value}.")
    return result


def handle_stuck_sensor(
    df: pd.DataFrame,
    column: str = "driver_exp_yr",
    stuck_value: int = 20,
    start_date: str = "2026-04-03",
    end_date: str = "2026-04-23",
    date_col: str = "date",
) -> pd.DataFrame:
    """Impute the identified stuck-sensor value within the affected date window."""
    result = df.copy()
    if column not in result.columns or date_col not in result.columns:
        return result

    dates = pd.to_datetime(result[date_col], errors="coerce")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = dates.between(start, end) & (result[column] == stuck_value)
    count = int(mask.sum())
    if count:
        valid = pd.to_numeric(result.loc[~mask, column], errors="coerce").dropna()
        if valid.empty:
            raise ValueError(f"Cannot impute stuck sensor {column}: no valid readings remain")
        median_value = int(round(float(valid.median())))
        result.loc[mask, column] = median_value
        result[column] = result[column].astype(int)
        print(f"[Stuck Sensor] Imputed {count} readings between {start_date} and {end_date} with median {median_value}.")
    return result


def clean_dataframe(
    df: pd.DataFrame, params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Apply the configured cleaning sequence to an in-memory DataFrame."""
    if params is None:
        params = load_params()

    clean_cfg = params.get("cleaning", {})
    diag_cfg = params.get("diagnostics", {})
    result = drop_duplicates(df)
    result = fix_target_typos(
        result,
        target_col=diag_cfg.get("target_column", "delay_min"),
        threshold=clean_cfg.get("target_typo_threshold", 100.0),
        factor=clean_cfg.get("target_typo_factor", 10.0),
    )
    result = normalize_categories(
        result,
        column="weather",
        mapping=clean_cfg.get("weather_mapping", {"cle.": "clear", "rai.": "rain", "fog.": "fog"}),
    )
    result = handle_sentinels(
        result,
        column=clean_cfg.get("sentinel_column", "passengers"),
        sentinel_value=clean_cfg.get("sentinel_value", -1),
    )
    result = handle_stuck_sensor(
        result,
        column=clean_cfg.get("stuck_sensor_column", "driver_exp_yr"),
        stuck_value=clean_cfg.get("stuck_sensor_value", 20),
        start_date=clean_cfg.get("stuck_sensor_start_date", "2026-04-03"),
        end_date=clean_cfg.get("stuck_sensor_end_date", "2026-04-23"),
        date_col=diag_cfg.get("date_column", "date"),
    )
    return result


def clean_train_data(params_path: str = "params.yaml") -> pd.DataFrame:
    """Run the configured cleaning pipeline and save its DVC output."""
    params = load_params(params_path)
    data_cfg = params.get("data", {})
    raw_path = Path(data_cfg.get("raw_train_path", "data/raw/train.csv"))
    output_path = Path(data_cfg.get("processed_train_path", "data/processed/train_clean.csv"))
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw training file not found at {raw_path}. Restore it with DVC before cleaning."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading original training data from: {raw_path}")
    raw = pd.read_csv(raw_path)
    cleaned = clean_dataframe(raw, params=params)
    cleaned.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")
    print(f"Final shape: {cleaned.shape}")
    return cleaned


if __name__ == "__main__":
    clean_train_data()
