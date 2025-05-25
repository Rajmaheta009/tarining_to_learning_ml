from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model ,full_pipeline = joblib.load("pkl files/chat_gpt_model.pkl")


class Transaction(BaseModel):
    id: int
    date: str
    price: float
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
    df_processed = full_pipeline.transform(df)
    prediction = model.predict(df_processed)
    return {"price": prediction}
    # return "hi that api working"

