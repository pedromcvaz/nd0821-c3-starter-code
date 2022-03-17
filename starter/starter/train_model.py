# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import pickle
from ml.data import process_data
from ml.model import train_rf_model

# Add code to load in the data.
data = pd.read_csv("data/clean_census.csv")

train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

with open("model/encoder_file", 'wb') as f:
    pickle.dump(encoder, f)

with open("model/lb_file", 'wb') as f:
    pickle.dump(lb, f)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
model = train_rf_model(X_train, y_train)
with open("model/model_file", 'wb') as f:
    pickle.dump(model, f)
