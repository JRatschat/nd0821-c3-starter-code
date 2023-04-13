import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .starter.ml.data import process_data
from .starter.ml.model import inference

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

MODEL_PATH = Path.cwd() / "starter/model/model.sav"
ENCODER_PATH = Path.cwd() / "starter/model/encoder.sav"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

app = FastAPI()


class InferenceData(BaseModel):
    age: int = Field(default= 32)
    workclass: str = Field(default="Private")
    fnlgt: int = Field(default=280464)
    education: str = Field(default="Bachelors")
    education_num: int = Field(default=13 ,alias="education-num")
    marital_status: str = Field(default="Never-married", alias="marital-status")
    occupation: str = Field(default="Adm-clerical")
    relationship: str = Field(default="Not-in-family")
    race: str = Field(default="Black")
    sex: str = Field(default="Male")
    capital_gain: int = Field(default=14084, alias="capital-gain")
    capital_loss: int = Field(default=0, alias="capital-loss")
    hours_per_week: int = Field(default=40, alias="hours-per-week")
    native_country: str = Field(default="United-States", alias="native-country")

    class Config:
        schema_extra = {
            "example": {
            "age": "32",
            "workclass": "Private",
            "fnlgt": "280464",
            "education": "Bachelors",
            "education_num": "education-num",
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "Black",
            "sex": "Male",
            "capital_gain": "0",
            "capital_loss": "0",
            "hours_per_week": "40",
            "native_country": "United-States",
            }
        }


@app.get("/")
async def read_root():
    return {"welcome message": "Welcome"}


@app.post("/api")
async def get_inference(data: InferenceData):
    data_df = pd.DataFrame.from_dict([data.dict(by_alias=True)])
    X, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=None,
    )
    pred = inference(model, X)
    pred_label = "<=50K" if pred[0] == 0 else ">50K"
    return {"Prediction": pred_label}

