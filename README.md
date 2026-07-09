<<<<<<< HEAD
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


# Configuration Airflow
1. Installer les dépendances :

```bash
pip install -r requirements.txt
```

2. Start Airflow
airflow standalone

3. Ouvrir Airflow dans le navigateur :
http://localhost:8080

4. Lancer le DAG :
trustpilot_ml_pipeline

Le fichier du DAG se trouve dans :

airflow/dags/trustpilot_ml_pipeline.py

Le pipeline exécute deux étapes :

train_model : entraîne le modèle et enregistre les résultats dans DagsHub (MLflow).
evaluate_model : évalue le modèle entraîné.
=======
# Projet MLOps - Classification d'Avis Trustpilot

Ce projet implémente un pipeline complet de Machine Learning (MLOps) pour la classification d'avis clients, orchestré avec **Airflow**, packagé via **Docker**, et suivi avec **DagsHub / MLflow**.

---

## ⚙️ Étape 1 : Configuration Initiale (Obligatoire)

Avant de lancer le projet, vous devez configurer vos identifiants DagsHub pour permettre le suivi des métriques et des modèles.

1. Créez une copie du fichier d'exemple :
   ```bash
   cp .env.example .env
   ```
2. Ouvrez le fichier `.env` et remplacez les valeurs par vos identifiants DagsHub personnels (votre pseudo et votre Token/Mot de passe).
   *Note : Le fichier `.env` est ignoré par Git pour sécuriser vos mots de passe.*

---

## 🚀 Étape 2 : Lancement de l'Architecture (Docker & Airflow)

Assurez-vous que **Docker** est lancé sur votre machine, puis tapez la commande suivante à la racine du projet :
```bash
docker-compose up --build
```

Une fois les conteneurs démarrés, vous aurez accès à :
- **FastAPI** : [http://localhost:8000/docs](http://localhost:8000/docs)
- **Airflow** : [http://localhost:8080](http://localhost:8080)
  - *Identifiant* : `admin`
  - *Mot de passe* : `admin`

Depuis l'interface d'Airflow, vous pouvez déclencher manuellement le DAG `trustpilot_retraining_pipeline` pour observer le ré-entraînement automatique du modèle et son envoi sur DagsHub !

---

## 💻 Développement Local (Sans Docker)

Si vous souhaitez modifier le code et tester localement sans lancer toute l'architecture Docker, voici la marche à suivre :

### Installation de l'environnement Python

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS/WSL
# ou venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

**Installation de PyTorch (Au choix) :**
- Pour CPU (Plus rapide à installer) : `pip install torch`
- Pour GPU CUDA (Entraînement accéléré) : `pip install torch --index-url https://download.pytorch.org/whl/cu128`

### Lancer l'entraînement manuel
```bash
python src/modelisation/train.py
```
*(Le modèle final sera sauvegardé dans `models/camembert` et uploadé sur DagsHub).*

### Lancer l'API seule
```bash
uvicorn src.api.main:api --reload --host 0.0.0.0 --port 8000
```

---

## 🧪 Documentation de l'API

L'API documente ses propres routes via Swagger : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Endpoints disponibles :**
- `GET /health` : Vérifie l'état du serveur et si le modèle est bien chargé (Accès public).
- `GET /model/info` : Renvoie la configuration actuelle du modèle (labels, seuils optimisés, device).
- `GET /model/metrics` : Renvoie les métriques de la dernière exécution MLflow (F1-score) depuis DagsHub.
- `POST /predict` : Effectue une prédiction de classification sur une liste de textes.

> **Note d'authentification** : La plupart des routes sont protégées. Dans l'interface Swagger, cliquez sur le bouton vert **"Authorize"** et entrez la clé d'API (par défaut : `super_secret_token_trustpilot_2024`) pour pouvoir exécuter vos requêtes de test.
>>>>>>> quentin/main
