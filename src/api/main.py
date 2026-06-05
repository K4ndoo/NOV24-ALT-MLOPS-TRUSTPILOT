from fastapi import FastAPI
from predict import predictor

from pydantic import BaseModel
from typing import List, Dict

class PredictionRequest(BaseModel):
    texts: List[str]

class LabelResult(BaseModel):
    prediction: int
    probability: float
    threshold: float

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, LabelResult]]


api = FastAPI(
    title="API de classification de reviews TrustPilot"
)


@api.get("/")
def root():
    return {"message": "Bienvenue sur l'API de classification de reviews TrustPilot !"}

@api.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    predictions = predictor.predict(request.texts)

    return {
        "predictions": predictions
    }