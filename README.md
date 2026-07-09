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
