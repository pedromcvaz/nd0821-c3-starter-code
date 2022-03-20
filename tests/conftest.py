import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from fastapi.testclient import TestClient
import os
import sys

sys.path.insert(0, os.getcwd())
from starter.starter.ml.data import process_data
from main import app
from starter.starter.ml.model import train_rf_model, inference


@pytest.fixture
def data():
    return pd.read_csv("starter/data/clean_census.csv")


@pytest.fixture
def train_data(data):
    train, _ = train_test_split(data, test_size=0.20)
    return train


@pytest.fixture
def test_data(data):
    _, test = train_test_split(data, test_size=0.20)
    return test


@pytest.fixture
def cat_features():
    cat = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat


@pytest.fixture
def preprocessed_train_data(train_data, cat_features):
    X, y, _, _ = process_data(
        train_data, categorical_features=cat_features, label="salary", training=True)
    return X, y


@pytest.fixture
def encoder_lb(train_data, cat_features):
    _, _, encoder, lb = process_data(
        train_data, categorical_features=cat_features, label="salary", training=True)
    return encoder, lb


@pytest.fixture
def preprocessed_test_data(test_data, cat_features, encoder_lb):
    encoder, lb = encoder_lb
    X, y, _, _ = process_data(
        test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    return X, y


@pytest.fixture
def model(preprocessed_train_data):
    X_train, y_train = preprocessed_train_data
    model = train_rf_model(X_train, y_train)
    return model


@pytest.fixture
def predictions(model, preprocessed_test_data):
    X_test, _ = preprocessed_test_data
    preds = inference(model, X_test)
    return preds


@pytest.fixture
def client():
    with TestClient(app) as cl:
        yield cl


@pytest.fixture
def payload():
    pl = {
        "age": 26,
        "workclass": "Private",
        "fnlgt": 172987,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
        "salary": "<=50K"
    }

    return pl


@pytest.fixture
def payload_2():
    pl = {
        "age": 56,
        "workclass": "Local-gov",
        "fnlgt": 216851,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "salary": ">50K"
    }

    return pl

@pytest.fixture
def payload_3():
    pl = {
        "age": "error",
        "workclass": "Private",
        "fnlgt": 17298,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
        "salary": "<=50K"
    }

    return pl
