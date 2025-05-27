from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# ✅ Unpack the model and preprocessing pipeline
model, full_pipeline = joblib.load("pkl files/car_accident_prediction.pkl")

class Transaction(BaseModel):
    car_make: str  # Manufacturer of the car (e.g., Toyota, Ford)
    car_model: str  # Specific model of the car (e.g., Camry, Mustang)
    year: int  # Year the car was manufactured
    mileage: int  # Total distance the car has traveled (in kilometers or miles)
    price: float  # Current price of the car
    fuel_type: str  # Type of fuel used (e.g., Petrol, Diesel, Electric, Hybrid)
    color: str  # Color of the car
    transmission: str  # Type of transmission (e.g., Automatic, Manual)
    options_features: str  # Additional features (e.g., Sunroof, Leather Seats)
    condition: str  # Condition rating (e.g., 1–5 or custom logic)


@app.post("/predict_car_accident")
def predict(txn: Transaction):
    df = pd.DataFrame([txn.dict()])
    df.rename(columns={
        'car_make': 'Car Make',
        'car_model': 'Car Model',
        'year': 'Year',
        'mileage': 'Mileage',
        'price': 'Price',
        'fuel_type': 'Fuel Type',
        'color': 'Color',
        'transmission': 'Transmission',
        'options_features': 'Options/Features',
        'condition': 'Condition'
    }, inplace=True)

    # ✅ Apply the saved preprocessing pipeline before prediction
    processed_df = full_pipeline.transform(df)

    prediction = model.predict(processed_df)

    if float(prediction[0]) >= 1:
        return {"prediction of accident": "Yes"}
    else:
        return {"prediction of accident": "No"}
