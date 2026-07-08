from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime


PROJECT_DIR = "/mnt/c/Users/vires/Downloads/NOV24-ALT-MLOPS-TRUSTPILOT"


with DAG(
    dag_id="trustpilot_ml_pipeline",
    start_date=datetime(2026, 7, 7),
    schedule=None,
    catchup=False,
    description="TrustPilot ML pipeline",
) as dag:


    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"""
        echo "Training task started"
    	sleep 10
    	echo "Training task finished"
    	"""
    )


    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=f"""
        cd {PROJECT_DIR}
        source ../../.venv/bin/activate
        python src/modelisation/evaluation.py
        """
    )


    train_model >> evaluate_model
