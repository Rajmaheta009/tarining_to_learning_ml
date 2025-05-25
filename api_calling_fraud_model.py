from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model ,full_pipline = joblib.load("pkl files/chat_gpt_model.pkl")


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
    isFraud:float


@app.post("/predict")
def predict(txn: Transaction):
    df = pd.DataFrame([txn.dict()])
    # If you did one-hot encoding or other preprocessing, apply it here
    df_processed = full_pipline.transform(df)
    prediction = model.predict(df_processed)
    return {"isFlaggedFraud": int(prediction)}
    # return "hi that api working"

