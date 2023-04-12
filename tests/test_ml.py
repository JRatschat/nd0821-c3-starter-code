from sklearn.ensemble import RandomForestClassifier
from starter.starter.ml.model import train_model, inference, compute_model_metrics


def test_train_model(data):
    X, y = data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference(train_and_fit_model, data):
    X, y = data
    preds = inference(train_and_fit_model, X)

    assert len(preds) == len(y)
    assert preds.any() == 1


def test_compute_model_metrics(train_and_fit_model, data):
    X, y = data
    preds = inference(train_and_fit_model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision != 0.0
    assert recall != 0.0
    assert fbeta != 0.0