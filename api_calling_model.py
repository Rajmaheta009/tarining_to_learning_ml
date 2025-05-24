from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("pkl files/fraud.pkl")


class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


@app.post("/predict")
def predict(txn: Transaction):
    df = pd.DataFrame([txn.dict()])

    # If you did one-hot encoding or other preprocessing, apply it here

    prediction = model.predict(df)
    return {"isFlaggedFraud": int(prediction[0])}
