from sklearn.ensemble import RandomForestClassifier
import os
import sys

sys.path.insert(0, os.getcwd())
from starter.starter.ml.model import train_rf_model, compute_model_metrics, inference


def test_train_model(preprocessed_train_data):
    X_train, y_train = preprocessed_train_data
    model = train_rf_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_inference(model, preprocessed_test_data):
    X_test, _ = preprocessed_test_data
    preds = inference(model, X_test)

    assert preds.shape[0] == X_test.shape[0]


def test_compute_model_metrics(preprocessed_test_data, predictions):
    _, y_test = preprocessed_test_data
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
