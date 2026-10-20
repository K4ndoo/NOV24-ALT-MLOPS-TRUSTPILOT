from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

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
) as dag:

    # Tâche 1 : Collecte des nouvelles données Trustpilot via scraper
    collect_new_data = BashOperator(
        task_id='collect_new_trustpilot_reviews',
        bash_command='cd /opt/airflow && python -u src/data_collection/scraper.py',
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

    # Tâche 4 : Décide si le candidat est meilleur que le modèle en place, et le déploie (DVC) si oui
    deploy_api = BashOperator(
        task_id='deploy_new_model_to_api',
        bash_command='cd /opt/airflow && python -u src/modelisation/deploy.py',
    )

    # Ordre d'exécution du pipeline
    collect_new_data >> save_new_data >> retrain_model >> evaluate_model >> deploy_api
