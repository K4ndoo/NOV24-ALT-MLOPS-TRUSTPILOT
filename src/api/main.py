# imports
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_403_FORBIDDEN
from src.api.predict import predictor
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

import torch
from mlflow.tracking.client import MlflowClient

# Prometheus
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response


# configuration API
API_KEY = os.getenv("API_KEY", "super_secret_token_trustpilot_2024")
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(
    name=API_KEY_NAME,
    auto_error=False
)


# authentification
def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header

    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Accès refusé : Clé d'API manquante ou invalide"
    )


# modèles Pydantic
class PredictionRequest(BaseModel):
    texts: List[str]


class LabelResult(BaseModel):
    prediction: int
    probability: float
    threshold: float


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, LabelResult]]


# ------------------------------------------------------------
# METRICS PROMETHEUS
# ------------------------------------------------------------

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Nombre total de requêtes envoyées à /predict",
)

PREDICTION_TEXTS = Counter(
    "prediction_texts_total",
    "Nombre total de textes envoyés au modèle",
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Nombre total d'erreurs lors des prédictions",
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Durée des prédictions en secondes",
)

ACTIVE_PREDICTIONS = Gauge(
    "active_predictions",
    "Nombre de prédictions actuellement en cours",
)


# ------------------------------------------------------------
# LIFESPAN
# ------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("🚀 Démarrage de l'API : Chargement des poids du modèle...")

    try:
        predictor.load()

    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        print(
            "Le modèle n'a peut-être pas encore été entraîné. "
            "Veuillez lancer 'python -m src.modelisation.train' d'abord."
        )

    yield

    print("🛑 Arrêt de l'API : Nettoyage...")

    predictor.model = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


api = FastAPI(
    title="API de classification des avis TrustPilot",
    lifespan=lifespan
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------

@api.get("/")
def root():
    return {
        "message": "Bienvenue sur l'API de classification de reviews TrustPilot !"
    }


@api.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    api_key: str = Depends(get_api_key)
):

    PREDICTION_REQUESTS.inc()
    PREDICTION_TEXTS.inc(len(request.texts))
    ACTIVE_PREDICTIONS.inc()

    try:

        with PREDICTION_LATENCY.time():

            predictions = predictor.predict(request.texts)

        return {
            "predictions": predictions
        }

    except Exception:

        PREDICTION_ERRORS.inc()
        raise

    finally:

        ACTIVE_PREDICTIONS.dec()


@api.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": predictor.model is not None
    }


@api.get("/model/info")
def model_info(
    api_key: str = Depends(get_api_key)
):

    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé"
        )

    return {
        "device": str(predictor.device),
        "labels": predictor.labels,
        "thresholds": predictor.thresholds
    }


@api.post("/model/reload")
def model_reload(
    api_key: str = Depends(get_api_key)
):

    # Appelé par Airflow après une promotion : le dossier models/ est partagé entre les conteneurs,
    # il suffit donc de recharger les poids depuis le disque
    try:
        predictor.load()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Échec du rechargement du modèle : {e}"
        )

    return {
        "status": "reloaded",
        "labels": predictor.labels,
        "thresholds": predictor.thresholds
    }


@api.get("/model/metrics")
def model_metrics(
    limit: int = 1,
    api_key: str = Depends(get_api_key)
):

    try:

        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "sqlite:///mlflow.db"
        )

        client = MlflowClient(
            tracking_uri=tracking_uri
        )

        experiment = client.get_experiment_by_name(
            "TrustPilot_CamemBERT_Classification"
        )

        if not experiment:
            return {
                "error": "Expérience MLflow introuvable"
            }

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=limit
        )

        if not runs:
            return {
                "error": "Aucun run trouvé"
            }

        return {
            "runs": [
                {
                    "run_id": run.info.run_id,
                    "metrics": run.data.metrics
                }
                for run in runs
            ]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ------------------------------------------------------------
# METRICS PROMETHEUS
# ------------------------------------------------------------

@api.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
