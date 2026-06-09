from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from src.api.predict import predictor
from pydantic import BaseModel
from typing import List, Dict

# faudra mettre la clé dans une variable d'environnement par la suite
API_KEY = "super_secret_token_trustpilot_2024"
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Accès refusé : Clé d'API manquante ou invalide"
    )

class PredictionRequest(BaseModel):
    texts: List[str]

class LabelResult(BaseModel):
    prediction: int
    probability: float
    threshold: float

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, LabelResult]]

api = FastAPI(title="API de classification de reviews TrustPilot")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup : On charge le modèle en RAM
    print("🚀 Démarrage de l'API : Chargement des poids du modèle...")
    predictor.load()
    yield
    # Shutdown : On libère la mémoire à l'arrêt du serveur
    print("🛑 Arrêt de l'API : Nettoyage...")
    predictor.model = None

@api.get("/")
def root():
    return {"message": "Bienvenue sur l'API de classification de reviews TrustPilot !"}

@api.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, api_key: str = Depends(get_api_key)):
    predictions = predictor.predict(request.texts)
    return {
        "predictions": predictions
    }