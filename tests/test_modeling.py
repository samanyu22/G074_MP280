"""Self-contained tests for MP280 feature selection and preprocessing."""

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.data.clean import normalize_categories
from src.modeling.preprocessing import (
    FEATURE_COLUMNS,
    build_pipeline,
    chronological_split,
    select_features,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for day in range(1, 11):
        rows.append(
            {
                "record_id": f"R{day:03d}",
                "date": f"2026-01-{day:02d}",
                "route_km": 10.0 + day,
                "stops": 5 + day,
                "passengers": 20 + day,
                "hour": 8 + day % 5,
                "temp_c": 25.0 + day,
                "driver_exp_yr": day,
                "weather": "clear" if day % 2 else "rain",
                "depot_zone": "North" if day % 3 else "South",
                "delay_min": float(5 + day),
            }
        )
    return pd.DataFrame(rows)


def test_model_features_are_exactly_the_eight_client_fields():
    features = select_features(sample_frame())
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert "record_id" not in features.columns
    assert "date" not in features.columns
    assert "delay_min" not in features.columns


def test_preprocessor_handles_missing_and_unseen_categories():
    train = sample_frame()
    train.loc[0, "temp_c"] = float("nan")
    train.loc[1, "weather"] = None
    model = build_pipeline(LinearRegression())
    model.fit(select_features(train), train["delay_min"])

    unseen = sample_frame().iloc[:2].copy()
    unseen["weather"] = "hail"
    predictions = model.predict(select_features(unseen))
    assert len(predictions) == 2
    assert pd.notna(predictions).all()


def test_chronological_split_keeps_validation_after_training():
    frame = sample_frame()
    x_train, y_train, x_valid, y_valid = chronological_split(
        frame, validation_fraction=0.2
    )
    train_max = pd.to_datetime(frame.loc[x_train.index, "date"]).max()
    valid_min = pd.to_datetime(frame.loc[x_valid.index, "date"]).min()
    assert train_max < valid_min
    assert len(x_train) + len(x_valid) == len(frame)
    assert len(y_train) == len(x_train)
    assert len(y_valid) == len(x_valid)


def test_chronological_split_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        chronological_split(sample_frame(), validation_fraction=1.0)


def test_weather_normalization_preserves_missing_values():
    dirty = pd.DataFrame({"weather": [" CLE. ", "Rain", None, "fog."]})
    cleaned = normalize_categories(
        dirty, mapping={"cle.": "clear", "rai.": "rain", "fog.": "fog"}
    )
    assert cleaned.loc[0, "weather"] == "clear"
    assert cleaned.loc[1, "weather"] == "rain"
    assert cleaned.loc[3, "weather"] == "fog"
    assert pd.isna(cleaned.loc[2, "weather"])
