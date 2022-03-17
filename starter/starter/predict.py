import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import load_rf_model, inference, compute_model_metrics, save_model_metrics, metrics_slice
from ml.data import process_data


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

with open('model/encoder_file', 'rb') as f:
    encoder = pickle.load(f)

with open('model/lb_file', 'rb') as f:
    lb = pickle.load(f)

data = pd.read_csv("data/clean_census.csv")

_, test = train_test_split(data, test_size=0.20)

test = test.reset_index(drop=True)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

rfc = pickle.load(open("model/model_file", 'rb'))
preds = inference(rfc, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
save_model_metrics("precision", precision)
save_model_metrics("recall", recall)
save_model_metrics("fbeta", fbeta)
print("#"*50)
print(test.shape[0])
print("#"*50)
metrics_slice(cat_features, preds, y_test, "model/", test)
