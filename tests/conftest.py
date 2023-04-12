import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="session")
def data():
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    return X, y


@pytest.fixture(scope="session")
def train_and_fit_model(data):
    X, y = data
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model


@pytest.fixture(scope="session")
def test_data_above_50():
    return {
        "age": 32,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }


@pytest.fixture(scope="session")
def test_data_below_50():
    return {
        "age": 20,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Bachelors",
        "education-num": 10,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }