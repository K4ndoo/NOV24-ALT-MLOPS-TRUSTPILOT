from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.main import api, API_KEY
from src.api.predict import predictor

client = TestClient(api)

def test_root_endpoint():
    """Test que la route racine répond bien en HTTP 200"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de classification de reviews TrustPilot !"}

def test_predict_without_api_key_is_forbidden():
    """Test de sécurité : tentative d'accès sans clé d'API"""
    payload = {"texts": ["La livraison était super rapide et le produit est top !"]}
    response = client.post("/predict", json=payload)

    assert response.status_code == 403
    assert response.json() == {"detail": "Accès refusé : Clé d'API manquante ou invalide"}

@patch.object(predictor, "predict")
def test_predict_with_valid_api_key(mock_predict):
    
    # On définit ce que le faux modèle doit renvoyer instantanément
    mock_predict.return_value = [
        {
            "qualité produit": {"prediction": 1, "probability": 0.95, "threshold": 0.5},
            "service livraison": {"prediction": 1, "probability": 0.88, "threshold": 0.5},
            "service client": {"prediction": 0, "probability": 0.12, "threshold": 0.5}
        }
    ]

    payload = {"texts": ["La livraison était super rapide et le produit est top !"]}
    headers = {"X-API-Key": API_KEY}
    
    response = client.post("/predict", json=payload, headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    
    # On vérifie que notre faux modèle a bien été appelé
    mock_predict.assert_called_once_with(["La livraison était super rapide et le produit est top !"])

def test_reload_without_api_key_is_forbidden():
    """Test de sécurité : le rechargement du modèle est réservé aux appels authentifiés"""
    response = client.post("/model/reload")

    assert response.status_code == 403

@patch.object(predictor, "load")
def test_reload_with_valid_api_key(mock_load):
    headers = {"X-API-Key": API_KEY}

    response = client.post("/model/reload", headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "reloaded"

    # On vérifie que le rechargement des poids a bien été déclenché
    mock_load.assert_called_once()

@patch.object(predictor, "load", side_effect=RuntimeError("poids introuvables"))
def test_reload_failure_returns_500(mock_load):
    headers = {"X-API-Key": API_KEY}

    response = client.post("/model/reload", headers=headers)

    assert response.status_code == 500