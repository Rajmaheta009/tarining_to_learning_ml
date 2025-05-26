from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model  = joblib.load("pkl files/house_price_prediction.pkl")


class Transaction(BaseModel):
    date: str
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int


@app.post("/predict")
def predict(txn: Transaction):
    df = pd.DataFrame([txn.dict()])
    # If you did one-hot encoding or other preprocessing, apply it here
    prediction = model.predict(df)
    return {"predicted_price": round(float(prediction[0]), 2)}
    # return "hi that api working"

