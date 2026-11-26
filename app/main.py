import json
import logging
import sys

from pydantic import BaseModel, ConfigDict, Field
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd

from src.modeling.preprocessing import select_features
from fastapi import FastAPI

class PredictionJsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "level": record.levelname,
            "event": record.getMessage(),
            "prediction_minutes": record.prediction_minutes,
        })


prediction_logger = logging.getLogger("mp280.prediction")
prediction_logger.setLevel(logging.INFO)
prediction_logger.propagate = False

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(PredictionJsonFormatter())
prediction_logger.addHandler(handler)

class TripRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_km: float
    stops: float = Field(ge=2.5)
    passengers: int
    hour: int
    temp_c: float = Field(le=70)
    driver_exp_yr: float
    weather: str
    depot_zone: str
    
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = joblib.load(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)

@app.post("/predict")
def predict(request: TripRequest):
    row = select_features(pd.DataFrame([request.model_dump()]))
    delay_minutes = float(app.state.pipeline.predict(row)[0])

    prediction_logger.info(
        "prediction_completed",
        extra={"prediction_minutes": delay_minutes},
    )
    return {"prediction": delay_minutes}

@app.get("/health")
def health():
    return {"status": "ok"}