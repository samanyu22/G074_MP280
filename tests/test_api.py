import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def trip():
    return {
        "route_km": 12.5,
        "stops": 10,
        "passengers": 28,
        "hour": 9,
        "temp_c": 26,
        "driver_exp_yr": 5,
        "weather": "clear",
        "depot_zone": "North",
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_prediction(client, trip):
    response = client.post("/predict", json=trip)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], (int, float))


@pytest.mark.parametrize("field,value", [("stops", 2), ("temp_c", 71)])
def test_invalid_prediction(client, trip, field, value):
    trip[field] = value
    response = client.post("/predict", json=trip)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]