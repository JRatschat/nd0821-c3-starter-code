from fastapi.testclient import TestClient

from starter.main import app

client = TestClient(app)


def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"welcome message": "Welcome"}


def test_post_above_50(test_data_above_50):
    response = client.post("/api", json=test_data_above_50)
    assert response.status_code == 200
    assert response.json() == {"Prediction": ">50K"}


def test_post_below_50(test_data_below_50):
    response = client.post("/api", json=test_data_below_50)
    assert response.status_code == 200
    assert response.json() == {"Prediction": "<=50K"}
