from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deployment.serve import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict():
    sample_input = {
        "gender": "Male",
        "age": 21.872484,
        "height": 1.699998,
        "overweight_familiar": "yes",
        "eat_hc_food": "yes",
        "eat_vegetables": 2.0,
        "main_meals": 2.970675,
        "snack": "Sometimes",
        "smoke": "no",
        "drink_water": 2.0,
        "monitoring_calories": "no",
        "physical_activity": 0.0,
        "use_of_technology": 0.169294,
        "drink_alcohol": "no",
        "transportation_type": "Public_Transportation",
    }

    sample_input = {"features": sample_input}

    response = client.post("/predict", json=sample_input)

    assert response.status_code == 200
    assert "prediction" in response.json()
