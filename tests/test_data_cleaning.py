"""Unit tests for the data cleaning pipeline (MP280).

These tests run without requiring raw CSV files (satisfying the handbook requirement
for CI where raw CSVs are tracked by DVC rather than git).
"""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import (
    clean_dataframe,
    drop_duplicates,
    fix_target_typos,
    handle_sentinels,
    handle_stuck_sensor,
    normalize_categories,
)


@pytest.fixture
def sample_dirty_data() -> pd.DataFrame:
    """Create a self-contained dirty DataFrame representing all data issues."""
    records = [
        # Normal row
        {
            "record_id": "R001",
            "date": "2026-01-10",
            "route_km": 15.0,
            "stops": 20,
            "passengers": 40,
            "hour": 10,
            "temp_c": 22.0,
            "driver_exp_yr": 12,
            "weather": "clear",
            "depot_zone": "North",
            "delay_min": 20.0,
        },
        # Duplicate row of R001
        {
            "record_id": "R001",
            "date": "2026-01-10",
            "route_km": 15.0,
            "stops": 20,
            "passengers": 40,
            "hour": 10,
            "temp_c": 22.0,
            "driver_exp_yr": 12,
            "weather": "clear",
            "depot_zone": "North",
            "delay_min": 20.0,
        },
        # Row with negative passenger sentinel (-1) & weather typo ('RAIN')
        {
            "record_id": "R002",
            "date": "2026-02-15",
            "route_km": 18.0,
            "stops": 25,
            "passengers": -1,
            "hour": 14,
            "temp_c": 26.0,
            "driver_exp_yr": 8,
            "weather": "RAIN",
            "depot_zone": "South",
            "delay_min": 25.0,
        },
        # Row with weather abbreviation ('cle.') & target typo (240.0 instead of 24.0)
        {
            "record_id": "R003",
            "date": "2026-03-20",
            "route_km": 21.0,
            "stops": 30,
            "passengers": 50,
            "hour": 16,
            "temp_c": 28.0,
            "driver_exp_yr": 15,
            "weather": "cle.",
            "depot_zone": "East",
            "delay_min": 240.0,
        },
        # Row with weather abbreviation ('rai.') & whitespace (' fog ')
        {
            "record_id": "R004",
            "date": "2026-03-25",
            "route_km": 12.0,
            "stops": 15,
            "passengers": 35,
            "hour": 8,
            "temp_c": 19.0,
            "driver_exp_yr": 5,
            "weather": " fog ",
            "depot_zone": "North",
            "delay_min": 18.0,
        },
        # Row during stuck sensor period (driver_exp_yr == 20)
        {
            "record_id": "R005",
            "date": "2026-04-10",
            "route_km": 16.0,
            "stops": 22,
            "passengers": 45,
            "hour": 11,
            "temp_c": 25.0,
            "driver_exp_yr": 20,
            "weather": "rai.",
            "depot_zone": "South",
            "delay_min": 22.0,
        },
    ]
    return pd.DataFrame(records)


def test_cleaner_removes_duplicates(sample_dirty_data: pd.DataFrame):
    """Test that cleaner detects and drops duplicate rows."""
    initial_count = len(sample_dirty_data)
    cleaned = drop_duplicates(sample_dirty_data)
    assert len(cleaned) == initial_count - 1
    assert cleaned.duplicated().sum() == 0


def test_no_negative_passengers_remain(sample_dirty_data: pd.DataFrame):
    """Test that sentinel passengers (-1) are imputed with a positive valid count."""
    assert (sample_dirty_data["passengers"] < 0).any()
    cleaned = handle_sentinels(sample_dirty_data, column="passengers", sentinel_value=-1)
    assert not (cleaned["passengers"] < 0).any()
    assert (cleaned["passengers"] > 0).all()


def test_weather_categories_contain_exact_levels(sample_dirty_data: pd.DataFrame):
    """Test that weather categories are normalized to exactly {'clear', 'rain', 'fog'}."""
    cleaned = normalize_categories(
        sample_dirty_data,
        column="weather",
        mapping={"cle.": "clear", "rai.": "rain", "fog.": "fog"},
    )
    unique_categories = set(cleaned["weather"].unique())
    assert unique_categories.issubset({"clear", "rain", "fog"})
    assert "RAIN" not in unique_categories
    assert "cle." not in unique_categories
    assert "rai." not in unique_categories


def test_target_typos_are_resolved(sample_dirty_data: pd.DataFrame):
    """Test that 10x target typos (>100) are divided by 10 into valid range."""
    assert (sample_dirty_data["delay_min"] >= 100.0).any()
    cleaned = fix_target_typos(sample_dirty_data, target_col="delay_min", threshold=100.0, factor=10.0)
    assert not (cleaned["delay_min"] >= 100.0).any()
    # 240.0 should be corrected to 24.0
    r003_delay = cleaned.loc[cleaned["record_id"] == "R003", "delay_min"].iloc[0]
    assert r003_delay == pytest.approx(24.0, abs=1e-3)


def test_stuck_sensor_is_handled(sample_dirty_data: pd.DataFrame):
    """Test that stuck sensor values during the anomaly window are imputed."""
    cleaned = handle_stuck_sensor(
        sample_dirty_data,
        column="driver_exp_yr",
        stuck_value=20,
        start_date="2026-04-03",
        end_date="2026-04-23",
        date_col="date",
    )
    # R005 was 20 during stuck period; should be imputed to median of non-stuck records
    r005_exp = cleaned.loc[cleaned["record_id"] == "R005", "driver_exp_yr"].iloc[0]
    assert r005_exp != 20


def test_end_to_end_clean_dataframe(sample_dirty_data: pd.DataFrame):
    """Test the complete cleaning pipeline end-to-end on synthetic data."""
    cleaned = clean_dataframe(sample_dirty_data)

    # 1. Deduplication verified
    assert cleaned.duplicated().sum() == 0

    # 2. No negative passengers
    assert (cleaned["passengers"] >= 0).all()

    # 3. Weather levels are strictly canonical
    assert set(cleaned["weather"].unique()).issubset({"clear", "rain", "fog"})

    # 4. Target typos resolved
    assert (cleaned["delay_min"] < 100.0).all()

    # 5. No missing/null values
    assert cleaned.isnull().sum().sum() == 0
