from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# ✅ Unpack the model and preprocessing pipeline
model, full_pipeline = joblib.load("../pkl_files/housing_price_prediction.pkl")

class Transaction(BaseModel):
    date: str  # This may not be used in prediction, consider dropping it if not trained
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

@app.post("/predict_housing_price")
def predict(txn: Transaction):
    df = pd.DataFrame([txn.dict()])
    # ✅ Apply the saved preprocessing pipeline before prediction
    processed_df = full_pipeline.transform(df)

    prediction = model.predict(processed_df)
    return {"predicted_price": float(prediction[0])}
