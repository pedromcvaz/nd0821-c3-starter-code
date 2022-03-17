import json
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_rf_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.DataFrame
        Labels.
    Returns
    -------
    rfc
        Trained machine learning model.
    """
    rfc = RandomForestClassifier()
    rfc.fit(X=X_train, y=y_train)

    return rfc


def load_rf_model(model_path):
    """
    Loads a RFC model from disk

    Inputs
    ----------
    model_path: str
        Path to the model file.
    Returns
    -------
    rfc
    """
    rfc = RandomForestClassifier()
    rfc.load_model(model_path)
    return rfc


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    print("$"*50)
    print(len(X))
    print("$" * 50)
    return np.array([round(x) for x in preds])


def save_model_metrics(metric_name, metric_value):
    """ Saves model metrics

        Inputs
        ------
        metric_name : str
            Metric name.
        metric_value : np.array
            Value for the given metric.
        Returns
        -------
        preds : float
        Predictions from the model.
        """
    with open(f'model/{metric_name}.json', 'w') as f:
        json.dump({metric_name: metric_value}, f)


def get_index(X_original, category, value):
    return X_original[X_original[category] == value].index.tolist()


def metrics_slice(categorical_list, preds, y, directory, test_original):
    results = {}
    category_list = []
    unique_list = []
    precision_list = []
    recall_list = []
    fbeta_list = []
    for category in categorical_list:
        for unique in test_original[category].unique():
            print("$" * 50)
            print(test_original[category].unique())
            print("$" * 50)
            slice_preds_indexes = get_index(test_original, category, unique)
            slice_preds = preds[slice_preds_indexes]
            slice_y = y[slice_preds_indexes]
            precision, recall, fbeta = compute_model_metrics(slice_preds, slice_y)
            category_list.append(category)
            unique_list.append(unique)
            precision_list.append(precision)
            recall_list.append(recall)
            fbeta_list.append(fbeta)

    results["Category"] = category_list
    results["Value"] = unique_list
    results["Precision"] = precision_list
    results["Recall"] = recall_list
    results["FBeta"] = fbeta_list
    pd.DataFrame(results).to_csv(f"{directory}/slice_output.txt", index=False)