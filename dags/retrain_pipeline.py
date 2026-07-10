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

    # Tâche 1 : Simulation de la vérification de nouvelles données
    def _check_data():
        print("[SIMULATION] Connexion à l'API Trustpilot...")
        print("[SIMULATION] 500 nouveaux avis détectés !")
        print("[SIMULATION] Fusion des nouveaux avis avec l'ancien dataset.")
        return True

    check_new_data = PythonOperator(
        task_id='check_new_trustpilot_reviews',
        python_callable=_check_data,
    )

    # Tâche 2 : Entraînement réel du modèle
    retrain_model = BashOperator(
        task_id='train_camembert_model',
        bash_command='export FAST_TRACK=true && cd /opt/airflow && python src/modelisation/train.py',
    )

    # Tâche 3 : Évaluation du modèle entraîné
    evaluate_model = BashOperator(
        task_id='evaluate_camembert_model',
        bash_command='cd /opt/airflow && python src/modelisation/evaluation.py',
    )

    # Tâche 4 : Simulation du déploiement
    def _deploy():
        print("[SIMULATION] Lecture du meilleur modèle depuis DagsHub...")
        print("[SIMULATION] Redémarrage de l'API avec les nouveaux poids !")
        print("[SIMULATION] Déploiement terminé avec succès.")
        return True

    deploy_api = PythonOperator(
        task_id='deploy_new_model_to_api',
        python_callable=_deploy,
    )

    # Ordre d'exécution du pipeline
    check_new_data >> retrain_model >> evaluate_model >> deploy_api
