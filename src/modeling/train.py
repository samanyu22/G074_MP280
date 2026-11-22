"""Train MP280 v1 candidates, track experiments, and register the winner.

Run from the repository root after `dvc repro clean`:
    python -m src.modeling.train
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.modeling.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_pipeline,
    chronological_split,
    select_features,
)


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Read configuration with yaml.safe_load."""
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find project configuration: {path}")
    with path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    if not isinstance(params, dict):
        raise ValueError("params.yaml must contain a YAML mapping")
    return params


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 for the untouched original training CSV."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_models(random_state: int) -> List[tuple[str, Any]]:
    """Four distinct model settings; all candidates use the same validation dates."""
    return [
        ("linear_regression", LinearRegression()),
        (
            "random_forest_baseline",
            RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "random_forest_regularized",
            RandomForestRegressor(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=3,
                max_features=0.8,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "gradient_boosting_huber",
            GradientBoostingRegressor(
                n_estimators=180,
                learning_rate=0.05,
                max_depth=2,
                loss="huber",
                random_state=random_state,
            ),
        ),
    ]


def train_v1(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Evaluate candidates, refit the winner on all cleaned train rows, and register v1."""
    params = load_params(params_path)
    data_cfg = params.get("data", {})
    training_cfg = params.get("training", {})

    raw_train_path = Path(data_cfg.get("raw_train_path", "data/raw/train.csv"))
    cleaned_path = Path(data_cfg.get("processed_train_path", "data/processed/train_clean.csv"))
    if not raw_train_path.exists():
        raise FileNotFoundError(
            f"Original training file is missing: {raw_train_path}. Restore it with DVC before training."
        )
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset is missing: {cleaned_path}. Run `dvc repro clean` first."
        )

    actual_sha = sha256_file(raw_train_path)
    expected_sha = str(params.get("data_sha256", "")).strip()
    if not expected_sha:
        raise ValueError("params.yaml must contain data_sha256 for the original train.csv")
    if actual_sha != expected_sha:
        raise ValueError(
            "Original train.csv SHA-256 does not match params.yaml. Do not modify the supplied raw dataset."
        )

    cleaned = pd.read_csv(cleaned_path)
    required = [*FEATURE_COLUMNS, TARGET_COLUMN, "date"]
    missing = [column for column in required if column not in cleaned.columns]
    if missing:
        raise ValueError(f"Cleaned training data is missing columns: {missing}")
    if cleaned.empty:
        raise ValueError("Cleaned training dataset is empty")

    validation_fraction = float(training_cfg.get("validation_fraction", 0.20))
    random_state = int(training_cfg.get("random_state", 42))
    experiment_name = str(training_cfg.get("experiment_name", "MP280_City_Bus_Delay"))
    registered_model_name = str(training_cfg.get("registered_model_name", "mp280_bus_delay"))
    model_output_path = Path(training_cfg.get("model_output_path", "models/model.joblib"))
    tracking_uri = str(training_cfg.get("tracking_uri", "sqlite:///mlflow.db"))

    x_train, y_train, x_valid, y_valid = chronological_split(
        cleaned, validation_fraction=validation_fraction
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    results: List[Dict[str, Any]] = []
    for model_name, estimator in candidate_models(random_state):
        pipeline = build_pipeline(clone(estimator))
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.set_tags(
                {
                    "project": "SCSE3040_MP280",
                    "stage": "v1_candidate",
                    "validation_strategy": "chronological",
                }
            )
            mlflow.log_params(
                {
                    "data_sha256": actual_sha,
                    "model_name": model_name,
                    "validation_fraction": validation_fraction,
                    "random_state": random_state,
                    "train_rows": len(x_train),
                    "validation_rows": len(x_valid),
                    "feature_count": len(FEATURE_COLUMNS),
                    "estimator_params": json.dumps(estimator.get_params(), sort_keys=True, default=str),
                }
            )

            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_valid)
            mae = float(mean_absolute_error(y_valid, predictions))
            rmse = float(mean_squared_error(y_valid, predictions) ** 0.5)
            mlflow.log_metrics({"mae_validation": mae, "rmse_validation": rmse})

            signature = infer_signature(x_valid, predictions)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="candidate_model",
                serialization_format="cloudpickle",
                input_example=x_valid.head(5),
                signature=signature,
            )
            results.append(
                {
                    "model_name": model_name,
                    "mae_validation": mae,
                    "rmse_validation": rmse,
                    "run_id": run.info.run_id,
                    "estimator": estimator,
                }
            )
            print(f"{model_name:30s} MAE={mae:.4f} min | RMSE={rmse:.4f} min")

    best = min(results, key=lambda item: item["mae_validation"])
    full_features = select_features(cleaned)
    full_target = pd.to_numeric(cleaned[TARGET_COLUMN], errors="raise")
    final_pipeline = build_pipeline(clone(best["estimator"]))
    final_pipeline.fit(full_features, full_target)

    # Resume the winning candidate run: this keeps its run ID as the v1 run ID.
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_param("final_fit_rows", len(cleaned))
        mlflow.set_tag("model_role", "v1_full_data_refit")
        signature = infer_signature(full_features, final_pipeline.predict(full_features))
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="model",
            serialization_format="cloudpickle",
            input_example=full_features.head(5),
            signature=signature,
            registered_model_name=registered_model_name,
            await_registration_for=120,
        )

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    versions = [
        version
        for version in client.search_model_versions(f"name='{registered_model_name}'")
        if version.run_id == best["run_id"]
    ]
    if not versions:
        raise RuntimeError(
            "The winning model was logged but no registered version was found. Check the MLflow Registry."
        )
    winning_version = max(versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=winning_version.version,
    )
    client.set_model_version_tag(
        name=registered_model_name,
        version=winning_version.version,
        key="data_sha256",
        value=actual_sha,
    )

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_output_path)

    print("\nV1 training complete")
    print(f"Best candidate: {best['model_name']}")
    print(f"Best validation MAE: {best['mae_validation']:.4f} minutes")
    print(f"Best MLflow run ID: {best['run_id']}")
    print(f"Registered model: {registered_model_name}")
    print(f"Registered version: {winning_version.version}")
    print("Alias: champion")
    print(f"Local model: {model_output_path}")
    print(f"Original data SHA-256: {actual_sha}")

    return {
        "experiment_name": experiment_name,
        "best_run_id": best["run_id"],
        "best_model": best["model_name"],
        "validation_mae": best["mae_validation"],
        "registered_model_name": registered_model_name,
        "champion_version": str(winning_version.version),
        "data_sha256": actual_sha,
        "model_path": str(model_output_path),
        "candidate_results": [
            {key: value for key, value in result.items() if key != "estimator"}
            for result in results
        ],
    }


if __name__ == "__main__":
    train_v1()
