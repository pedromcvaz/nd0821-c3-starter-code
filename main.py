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

# DVC on Heroku - required code
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

# Load models on startup to speed-up POST request step
@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    model = joblib.load("starter/model/model_file")
    encoder = joblib.load("starter/model/encoder_file")
    lb = joblib.load("starter/model/lb_file")



# Home site with welcome message - GET request
@app.get("/", tags=["home"])
async def get_root() -> dict:
    return {
        "message": "Welcome! This api provides an interface for scoring the census data from Udacity's ML DevOps program"
    }

def replace_dash(string: str) -> str:
    return string.replace('_','-')

# Class definition of the data that will be provided as POST request
class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int #= Field(..., alias='education-num')
    marital_status: str #= Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int #= Field(..., alias='capital-gain')
    capital_loss: int #= Field(..., alias='capital-loss')
    hours_per_week: int #= Field(..., alias='hours-per-week')
    native_country: str #= Field(..., alias='native-country')
    salary: Optional[str]

    class Config:
        alias_generator = replace_dash



# POST request to /predict site. Used to validate model with sample census data
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

    # Read data sent as POST
    input_data = input.dict(by_alias=True)
    input_df = pd.DataFrame(input_data, index=[0])
    logger.info(f"Input data: {input_df}")

    # Process the data
    X_train, _, _, _ = process_data(
                input_df, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb)

    preds = int(model.predict(X_train)[0])
    logger.info(f"Preds: {preds}")
    return {"result": preds}

if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
