from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Met à jour chaque jour le tableau de bord Evidently (service "evidently" de docker-compose, port 8001).
# Les avis d'inférence sont scrapés par GitHub Actions (daily_inference.yml, vers 8h14 UTC) et versionnés
# sur DagsHub : on les récupère, puis src/monitoring/update_dashboard.py ajoute les rapports manquants
# (dérive quotidienne jusqu'à la veille, et un rapport par nouvelle promotion du modèle).

INFERENCE_CSV = "reports/cache/dataset_inference.csv"  # copie de travail, jamais dans data/

# Le dépôt git est monté depuis l'hôte : on autorise git à le lire malgré un propriétaire différent
GIT_SAFE = "git config --global safe.directory /opt/airflow"

# Révision où lire les .dvc et les promotions : origin/main si on la connaît, sinon le dépôt local
GIT_REV = "$(git rev-parse -q --verify origin/main > /dev/null && echo origin/main || echo HEAD)"

default_args = {
    'owner': 'Equipe_MLOps',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    'trustpilot_drift_monitoring',
    default_args=default_args,
    description="Mise à jour quotidienne du tableau de bord de dérive Evidently",
    schedule_interval='0 11 * * *',  # après l'inférence quotidienne de GitHub Actions
    start_date=datetime(2025, 1, 1),
    catchup=False,  # inutile : update_dashboard rattrape lui-même tous les jours manquants
    # actif dès le premier docker compose up (lecture seule pour le projet, contrairement au ré-entraînement)
    is_paused_upon_creation=False,
    tags=['mlops', 'monitoring'],
) as dag:

    # Tâche 1 : Récupère la version des avis d'inférence référencée par origin/main du dépôt local.
    # Le conteneur ne contacte pas GitHub (dépôt privé) : faire `git fetch origin` sur l'hôte
    # pour que le tableau de bord voie les derniers jours d'inférence.
    sync_inference_data = BashOperator(
        task_id='sync_inference_data',
        bash_command=(
            f'cd /opt/airflow && {GIT_SAFE} && '
            'dvc remote modify origin --local user "$MLFLOW_TRACKING_USERNAME" && '
            'dvc remote modify origin --local password "$MLFLOW_TRACKING_PASSWORD" && '
            f'REV={GIT_REV} && echo "Révision utilisée : $REV" && '
            'mkdir -p reports/cache && '
            f'dvc get . data/daily_inference/dataset_inference.csv --rev "$REV" -o {INFERENCE_CSV} --force'
        ),
    )

    # Tâche 2 : Ajoute à l'espace Evidently les rapports manquants (aucun doublon, rien n'est recalculé)
    update_dashboard = BashOperator(
        task_id='update_evidently_dashboard',
        bash_command=(
            f'cd /opt/airflow && {GIT_SAFE} && '
            f'REV={GIT_REV} && '
            f'python -u -m src.monitoring.update_dashboard --rev "$REV" --inference {INFERENCE_CSV}'
        ),
    )

    sync_inference_data >> update_dashboard
