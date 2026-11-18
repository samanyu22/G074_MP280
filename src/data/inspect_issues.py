"""Exploratory and diagnostic script for hunting data problems in MP280.

This script implements checks for the 7 kinds of data issues described
in the Student Handbook and Problem specification.

Usage:
    python src/data/inspect_issues.py
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
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


def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for duplicate rows.

    Returns the number of EXTRA copies (rows to delete).
    """
    dup_count = int(df.duplicated().sum())
    return {
        "present": dup_count > 0,
        "count": dup_count if dup_count > 0 else None,
    }


def check_sentinels(
    df: pd.DataFrame, candidate_sentinels: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Check for sentinel values typed instead of blank/missing.

    e.g. -1, -999, 999, 9999.
    """
    if candidate_sentinels is None:
        candidate_sentinels = [-1, -999, 999, 9999]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        for val in candidate_sentinels:
            count = int((df[col] == val).sum())
            if count > 0:
                return {
                    "present": True,
                    "column": col,
                    "value": val,
                    "count": count,
                }

    return {"present": False, "column": None, "value": None, "count": None}


def check_stuck_sensor(
    df: pd.DataFrame, min_days: int = 10, date_col: str = "date"
) -> Dict[str, Any]:
    """Check if any sensor/column was stuck sending identical values across consecutive dates."""
    if date_col not in df.columns:
        return {"present": False, "column": None, "start_date": None, "end_date": None}

    df_sorted = df.sort_values(date_col).copy()
    numeric_cols = [
        c
        for c in df_sorted.select_dtypes(include=[np.number]).columns
        if c not in ["record_id", "delay_min"]
    ]

    for col in numeric_cols:
        # Group by date and find dates where all readings are identical (min == max)
        daily_stats = df_sorted.groupby(date_col)[col].agg(["count", "min", "max"])
        daily_zero_variance = daily_stats[
            (daily_stats["count"] > 1) & (daily_stats["min"] == daily_stats["max"])
        ]

        if len(daily_zero_variance) >= min_days:
            # Check for consecutive calendar dates
            dates = pd.to_datetime(daily_zero_variance.index)
            date_diffs = dates.to_series().diff().dt.days
            consec_groups = (date_diffs != 1).cumsum()

            for _, group in dates.to_series().groupby(consec_groups):
                if len(group) >= min_days:
                    start_date = group.min().strftime("%Y-%m-%d")
                    end_date = group.max().strftime("%Y-%m-%d")
                    stuck_val = daily_zero_variance.loc[start_date, "min"]
                    return {
                        "present": True,
                        "column": col,
                        "start_date": start_date,
                        "end_date": end_date,
                        "value": stuck_val,
                    }

    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(
    df: pd.DataFrame, target_col: str = "delay_min", gap_threshold: float = 100.0
) -> Dict[str, Any]:
    """Check for target values typed with an extra zero (10 times too big).

    Looks for an isolated cluster of values sitting far above normal range.
    """
    if target_col not in df.columns:
        return {"present": False, "count": None}

    # In this dataset, normal delay_min is <= 44.0, typos are >= 160.0
    typos = df[df[target_col] >= gap_threshold]
    count = int(len(typos))

    return {
        "present": count > 0,
        "count": count if count > 0 else None,
    }


def check_spelling_variants(
    df: pd.DataFrame,
    later_df: Optional[pd.DataFrame] = None,
    text_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Check text columns for spelling variants and formatting noise.

    Uses clean spellings from later.csv (as recommended in Handbook) or canonical normalization.
    """
    if text_columns is None:
        text_columns = [
            c
            for c in df.columns
            if c not in ["record_id", "date"] and (df[c].dtype == "object" or isinstance(df[c].dtype, pd.StringDtype))
        ]

    for col in text_columns:
        raw_values = df[col].dropna().astype(str)
        raw_count = raw_values.nunique()

        # If clean later.csv is provided, true levels are defined by it
        if later_df is not None and col in later_df.columns:
            true_levels = int(later_df[col].dropna().nunique())
            if raw_count > true_levels:
                return {
                    "present": True,
                    "column": col,
                    "true_levels": true_levels,
                }

        # Fallback: Canonical normalization (strip, lower, common abbreviation mapping)
        canonical = (
            raw_values.str.lower()
            .str.strip()
            .replace({"cle.": "clear", "rai.": "rain", "fog.": "fog"})
            .str.replace(r"[^a-z0-9]", "", regex=True)
        )
        clean_count = int(canonical.nunique())

        if raw_count > clean_count:
            return {
                "present": True,
                "column": col,
                "true_levels": clean_count,
            }

    return {"present": False, "column": None, "true_levels": None}


def check_unit_mixup(
    df: pd.DataFrame, group_col: str = "depot_zone", ratio_threshold: float = 5.0
) -> Dict[str, Any]:
    """Check if any column was recorded in a different unit for one depot_zone.

    One group would be roughly 10x, 12x, 60x, 100x, or 1000x different.
    """
    if group_col not in df.columns:
        return {"present": False, "column": None, "group_value": None, "factor": None}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    groups = df[group_col].dropna().unique()

    if len(groups) < 2:
        return {"present": False, "column": None, "group_value": None, "factor": None}

    for col in numeric_cols:
        means = df.groupby(group_col)[col].mean()
        if means.min() > 0:
            ratio = means.max() / means.min()
            if ratio >= ratio_threshold:
                outlier_group = means.idxmax() if ratio >= ratio_threshold else means.idxmin()
                return {
                    "present": True,
                    "column": col,
                    "group_value": str(outlier_group),
                    "factor": round(float(ratio)),
                }

    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_leakage(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame, target_col: str = "delay_min"
) -> Dict[str, Any]:
    """Check for columns in train that secretly contain the answer.

    Typically missing from holdout_features.csv or near-perfect correlation with target.
    """
    missing_in_holdout = [
        c
        for c in train_df.columns
        if c not in holdout_df.columns and c != target_col
    ]

    if missing_in_holdout:
        return {
            "present": True,
            "column": missing_in_holdout[0],
        }

    return {"present": False, "column": None}


def run_all_checks(
    params_path: str = "params.yaml",
) -> Dict[str, Any]:
    """Run all 7 diagnostic checks using configuration from params.yaml."""
    params = load_params(params_path)
    data_cfg = params.get("data", {})
    diag_cfg = params.get("diagnostics", {})

    train_path = Path(data_cfg.get("raw_train_path", "data/raw/train.csv"))
    later_path = Path(data_cfg.get("raw_later_path", "data/raw/later.csv"))
    holdout_path = Path(data_cfg.get("raw_holdout_path", "data/raw/holdout_features.csv"))

    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path) if later_path.exists() else None
    holdout_df = pd.read_csv(holdout_path)

    results = {
        "leaking_column": check_leakage(
            train_df, holdout_df, target_col=diag_cfg.get("target_column", "delay_min")
        ),
        "unit_mixup": check_unit_mixup(
            train_df,
            group_col=diag_cfg.get("group_column", "depot_zone"),
            ratio_threshold=diag_cfg.get("unit_mixup_ratio_threshold", 5.0),
        ),
        "sentinel_value": check_sentinels(
            train_df,
            candidate_sentinels=diag_cfg.get("sentinel_candidates", [-1, -999, 999, 9999]),
        ),
        "duplicate_rows": check_duplicates(train_df),
        "stuck_sensor": check_stuck_sensor(
            train_df,
            min_days=diag_cfg.get("stuck_sensor_min_days", 10),
            date_col=diag_cfg.get("date_column", "date"),
        ),
        "target_typos": check_target_typos(
            train_df,
            target_col=diag_cfg.get("target_column", "delay_min"),
            gap_threshold=diag_cfg.get("target_typo_threshold", 100.0),
        ),
        "spelling_variants": check_spelling_variants(train_df, later_df=later_df),
    }

    return results


def print_report(results: Dict[str, Any]) -> None:
    """Print a clean diagnostic report for review and viva explanation."""
    print("=" * 65)
    print(" " * 15 + "MP280 DATA QUALITY AUDIT REPORT")
    print("=" * 65)
    for issue, res in results.items():
        present = res.get("present")
        status = "[PRESENT]" if present else "[ABSENT] "
        print(f"\n{status} {issue.upper()}")
        for k, v in res.items():
            if k != "present":
                print(f"    - {k}: {v}")
    print("\n" + "=" * 65)


if __name__ == "__main__":
    results = run_all_checks()
    print_report(results)
