import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# Load model and encoder
model_path = "./model/model.pkl"
encoder_path = "./model/encoder.pkl"

model = load_model(model_path)
encoder = load_model(encoder_path)

# Create FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
async def get_root():
    """Say hello!"""
    return {"message": "Welcome to the API!"}

# Inference endpoint
@app.post("/data/")
async def post_inference(data: Data):
    try:
        # Convert Pydantic model to dict
        data_dict = data.dict()
        data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
        data = pd.DataFrame.from_dict(data)

        # Define categorical features
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

        # Process data
        data_processed, _, _, _ = process_data(
            data,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder
        )

        # Perform inference
        _inference = inference(model, data_processed)
        return {"result": apply_label(_inference)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))