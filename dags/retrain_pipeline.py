from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.models.param import Param
from datetime import datetime, timedelta
import json
import math
import os
import requests

# Adresse de l'API FastAPI (service "api" de docker-compose) et clé pour les routes protégées
API_URL = os.getenv("API_URL", "http://api:8000")
API_KEY = os.getenv("API_KEY") # lue depuis le .env via docker-compose
MODEL_CONFIG_PATH = "/opt/airflow/models/camembert/config.json"
SCRAPING_STATS_PATH = "/opt/airflow/data/last_scraping_training.json" # écrit par scraper.py
RUN_ID_PATH = "/opt/airflow/models/last_run_id.txt" # écrit par train.py


# Conditions : si une fonction renvoie False, toutes les tâches suivantes sont sautées (état "skipped")
# Le paramètre "force" (Trigger DAG w/ config) permet de lancer le pipeline complet quoi qu'il arrive, pour les démos

def has_new_reviews(params):
    if params["force"]:
        print("force=True : on continue sans vérifier le scraping")
        return True
    with open(SCRAPING_STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    print(f"Bilan du scraping : {stats}")
    if stats["new_reviews"] == 0:
        print("Aucun nouvel avis : ré-entraînement inutile, fin du pipeline")
        return False
    return True


def model_promoted(params):
    if params["force"]:
        print("force=True : on recharge l'API même si le modèle n'a pas été promu")
        return True
    # import ici pour ne pas charger mlflow à chaque lecture du DAG par le scheduler
    from mlflow.tracking.client import MlflowClient

    with open(RUN_ID_PATH, "r", encoding="utf-8") as f:
        run_id = f.read().strip()
    # tag posé par deploy.py après sa décision
    promoted = MlflowClient().get_run(run_id).data.tags.get("promoted") == "true"
    print(f"Run {run_id} promu : {promoted}")
    if not promoted:
        print("Modèle non retenu : l'API sert déjà le bon modèle, pas de rechargement")
    return promoted


def api_headers():
    if not API_KEY:
        raise RuntimeError("Variable API_KEY absente : ajoutez-la dans le fichier .env (voir .env.example)")
    return {"X-API-Key": API_KEY}


def check_api_health():
    # on vérifie que l'API est joignable avant de lancer un ré-entraînement de plusieurs minutes
    response = requests.get(f"{API_URL}/health", timeout=10)
    response.raise_for_status()
    print(f"API joignable : {response.json()}")


def reload_api_model():
    # le dossier models/ est partagé avec le conteneur de l'API : on lui demande de recharger les poids
    response = requests.post(f"{API_URL}/model/reload", headers=api_headers(), timeout=300)
    response.raise_for_status()
    print(f"Modèle rechargé dans l'API : {response.json()}")


def smoke_test_api():
    # Vérifie le déploiement réel (vrai modèle, vrai conteneur). Les routes, l'authentification
    # et le format des réponses sont déjà couverts par les tests unitaires (tests/api/test_main.py).

    # 1. le modèle est bien en mémoire
    health = requests.get(f"{API_URL}/health", timeout=10).json()
    if not health["model_loaded"]:
        raise RuntimeError("L'API répond mais le modèle n'est pas chargé")

    # 2. l'API sert bien le modèle présent sur le disque (celui promu par deploy.py)
    info = requests.get(f"{API_URL}/model/info", headers=api_headers(), timeout=10)
    info.raise_for_status()
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        disk_thresholds = json.load(f)["thresholds"]
    api_thresholds = info.json()["thresholds"]
    if len(api_thresholds) != len(disk_thresholds) or not all(
        math.isclose(a, b) for a, b in zip(api_thresholds, disk_thresholds)
    ):
        raise RuntimeError(f"Seuils de l'API {api_thresholds} différents du modèle sur disque {disk_thresholds}")

    # 3. le vrai modèle produit des probabilités cohérentes
    texts = [
        "La livraison était super rapide et le produit est top !",
        "Service client injoignable, je n'ai jamais été remboursé.",
    ]
    response = requests.post(f"{API_URL}/predict", json={"texts": texts}, headers=api_headers(), timeout=60)
    response.raise_for_status()
    predictions = response.json()["predictions"]
    for text, prediction in zip(texts, predictions):
        print(f"{text} -> {prediction}")
        for label, result in prediction.items():
            if not 0 <= result["probability"] <= 1:
                raise RuntimeError(f"Probabilité invalide pour '{label}' : {result['probability']}")

    print("Smoke test OK : l'API sert le nouveau modèle")

# Arguments par défaut pour notre DAG
default_args = {
    'owner': 'Equipe_MLOps',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Définition de notre pipeline d'orchestration
with DAG(
    'trustpilot_retraining_pipeline',
    default_args=default_args,
    description='Pipeline de ré-entraînement hebdomadaire pour Trustpilot',
    schedule_interval=timedelta(days=7), # S'exécute toutes les semaines
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'training'],
    # à cocher dans "Trigger DAG w/ config" pour exécuter toutes les tâches même sans nouveaux avis ni promotion
    params={
        "force": Param(False, type="boolean", description="Exécuter tout le pipeline sans tenir compte des conditions"),
        "promote_fast_track": Param(False, type="boolean", description="Autoriser la promotion d'un modèle FAST_TRACK (démo uniquement)"),
    },
) as dag:

    # Tâche 0 : Vérifie que l'API est disponible avant de lancer le pipeline
    check_api = PythonOperator(
        task_id='check_api_health',
        python_callable=check_api_health,
    )

    # Tâche 0 bis : Récupère depuis DagsHub le modèle et le dataset de référence (ceux des fichiers .dvc du dépôt git),
    # pour ne pas scraper ni comparer le candidat à partir de fichiers locaux périmés.
    # --force : écrase les versions locales qui ne correspondent pas aux fichiers .dvc
    sync_dvc_data = BashOperator(
        task_id='sync_dvc_data',
        bash_command=(
            'cd /opt/airflow && '
            'dvc remote modify origin --local user "$MLFLOW_TRACKING_USERNAME" && '
            'dvc remote modify origin --local password "$MLFLOW_TRACKING_PASSWORD" && '
            'dvc pull --force models/camembert.dvc data/labelled_dataset/dataset_avis.csv.dvc'
        ),
    )

    # Tâche 1 : Collecte des nouvelles données Trustpilot via scraper
    collect_new_data = BashOperator(
        task_id='collect_new_trustpilot_reviews',
        bash_command='cd /opt/airflow && python -u src/data_collection/scraper.py',
    )

    # Condition 1 : on ne continue que si le scraper a trouvé de nouveaux avis
    check_new_reviews = ShortCircuitOperator(
        task_id='has_new_reviews',
        python_callable=has_new_reviews,
    )

    # Tâche 1 bis : Versionnement des nouvelles données sur DagsHub (DVC), comme dans weekly_training.yml
    save_new_data = BashOperator(
        task_id='save_new_data_to_dvc',
        bash_command=(
            'cd /opt/airflow && '
            'dvc remote modify origin --local user "$MLFLOW_TRACKING_USERNAME" && '
            'dvc remote modify origin --local password "$MLFLOW_TRACKING_PASSWORD" && '
            'dvc add data/labelled_dataset/dataset_avis.csv && '
            'dvc push data/labelled_dataset/dataset_avis.csv.dvc -r origin'
        ),
    )

    # Tâche 2 : Entraînement réel du modèle
    retrain_model = BashOperator(
        task_id='train_camembert_model',
        bash_command='export FAST_TRACK=true && export TRAINING_SOURCE=airflow && cd /opt/airflow && python -u src/modelisation/train.py',
    )

    # Tâche 3 : Évaluation du modèle entraîné
    evaluate_model = BashOperator(
        task_id='evaluate_camembert_model',
        bash_command='cd /opt/airflow && python -u src/modelisation/evaluation.py',
    )

    # Tâche 4 : Décide si le candidat est meilleur que le modèle en place, et le promeut (DVC) si oui
    # Un modèle FAST_TRACK n'est promu que si le paramètre "promote_fast_track" est coché au déclenchement
    promote_model = BashOperator(
        task_id='promote_model_dvc',
        bash_command=(
            'export ALLOW_FAST_TRACK_PROMOTION={{ params.promote_fast_track | lower }} && '
            'cd /opt/airflow && python -u src/modelisation/deploy.py'
        ),
    )

    # Condition 2 : on ne recharge l'API que si le nouveau modèle a été promu
    check_promotion = ShortCircuitOperator(
        task_id='model_promoted',
        python_callable=model_promoted,
    )

    # Tâche 5 : Demande à l'API de recharger le nouveau modèle
    reload_api = PythonOperator(
        task_id='reload_api_model',
        python_callable=reload_api_model,
    )

    # Tâche 6 : Vérifie que l'API déployée sert bien le modèle et répond correctement
    smoke_test = PythonOperator(
        task_id='smoke_test_api',
        python_callable=smoke_test_api,
    )

    # Ordre d'exécution du pipeline
    (
        check_api >> sync_dvc_data >> collect_new_data >> check_new_reviews
        >> save_new_data >> retrain_model >> evaluate_model >> promote_model
        >> check_promotion >> reload_api >> smoke_test
    )
