## Installation

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS/WSL
# ou
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

# Pour CPU
```bash
pip install torch
```

# Pour GPU CUDA
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
Remplacez `cu128` par votre version de CUDA (`cu118`, `cu121`, etc.).

## Lancer l'API

Une fois le modèle entraîné, vous pouvez démarrer l'API avec la commande suivante (depuis l'environnement WSL où les dépendances sont installées) :

```bash
uvicorn src.api.main:api --reload --host 0.0.0.0 --port 8000
```

L'API sera alors accessible localement sur [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Tester l'API

FastAPI génère automatiquement le swagger qui permet de tester les requêtes directement depuis le navigateur. 

Disponible sur [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Endpoints disponibles :**
- `GET /health` : Vérifie l'état du serveur et si le modèle est bien chargé (Accès public).
- `GET /model/info` : Renvoie la configuration actuelle du modèle (labels, seuils optimisés, device).
- `GET /model/metrics` : Renvoie les métriques de la dernière exécution MLflow (F1-score, etc.).
- `POST /predict` : Effectue une prédiction de classification sur une liste de textes.

> **Note d'authentification** : La plupart des routes sont protégées. Dans l'interface Swagger, cliquez sur le bouton vert **"Authorize"** et entrez la clé d'API (par défaut : `super_secret_token_trustpilot_2024`) pour pouvoir exécuter vos requêtes de test.
