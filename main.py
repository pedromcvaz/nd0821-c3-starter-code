# Put the code for your API here.
import os
import logging
import joblib
import pandas as pd
import uvicorn
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.starter.ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    model = joblib.load("starter/model/model_file")
    encoder = joblib.load("starter/model/encoder_file")
    lb = joblib.load("starter/model/lb_file")


@app.get("/", tags=["home"])
async def get_root() -> dict:
    return {
        "message": "Welcome! This api provides an interface for scoring the census data from Udacity's ML DevOps program"
    }


def replace_dash(string: str) -> str:
    return string.replace('_','-')


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    salary: Optional[str]

    class Config:
        alias_generator = replace_dash
        schema_extra = {
            "example": {
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
        }

@app.post('/predict')
async def predict(input: Data):
    """
    POST request that will provide sample census data and expect a prediction
    Output:
        0 or 1
    """

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

    input_data = input.dict(by_alias=True)
    input_df = pd.DataFrame(input_data, index=[0])
    logger.info(f"Input data: {input_df}")

    X_train, _, _, _ = process_data(
                input_df, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb)

    preds = int(model.predict(X_train)[0])
    logger.info(f"Preds: {preds}")
    return {"result": preds}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
