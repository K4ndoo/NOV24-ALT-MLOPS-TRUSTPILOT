import os
import json
import shutil
import tempfile
import subprocess
import mlflow
from mlflow.tracking.client import MlflowClient
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = "models/camembert"
RUN_ID_FILE = "models/last_run_id.txt" # écrit par train.py


def get_candidate_run(client):
    # on déploie le run noté par train.py, et pas le dernier run de l'expérience
    # (un autre pipeline, Kaggle ou Airflow, a pu en lancer un entre-temps)
    if not os.path.exists(RUN_ID_FILE):
        raise RuntimeError(f"'{RUN_ID_FILE}' introuvable. Lance d'abord train.py.")
    with open(RUN_ID_FILE, "r", encoding="utf-8") as f:
        run_id = f.read().strip()
    return client.get_run(run_id)


def get_current_best_f1():
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("best_f1_micro")


def configure_dvc_remote():
    # on configure l'auth DVC à partir des identifiants DagsHub déjà utilisés pour MLflow
    subprocess.run(
        ["dvc", "remote", "modify", "origin", "--local", "user", os.getenv("MLFLOW_TRACKING_USERNAME", "")],
        check=True,
    )
    subprocess.run(
        ["dvc", "remote", "modify", "origin", "--local", "password", os.getenv("MLFLOW_TRACKING_PASSWORD", "")],
        check=True,
    )


def copy_tree_contents_only(src_dir, dst_dir):
    for root, dirs, files in os.walk(src_dir):
        # ignore d'éventuels checkpoints HuggingFace laissés dans un artefact plus ancien
        dirs[:] = [d for d in dirs if not d.startswith("checkpoint-")]

        rel_path = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, rel_path) if rel_path != "." else dst_dir
        os.makedirs(target_root, exist_ok=True)
        for filename in files:
            shutil.copyfile(os.path.join(root, filename), os.path.join(target_root, filename))


def promote(run_id, test_f1_micro):
    print(f"Téléchargement du modèle candidat (run {run_id}) vers '{OUTPUT_DIR}'...")
    
    # on supprime d'abord l'index de devc avant la vrai suppression avec shutil sinon on perd completement le tracking dvc et ca se desynchronise avec git
    if os.path.exists(f"{OUTPUT_DIR}.dvc"):
        subprocess.run(["dvc", "remove", f"{OUTPUT_DIR}.dvc"], check=False)

    # on repart d'un dossier propre pour ne pas mélanger avec des restes d'une tentative précédente
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="camembert_model", dst_path=tmp_dir
        )
        copy_tree_contents_only(downloaded, OUTPUT_DIR)

    # le score qui a servi à la décision devient la référence pour la prochaine comparaison
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["best_f1_micro"] = test_f1_micro
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    configure_dvc_remote()
    subprocess.run(["dvc", "add", OUTPUT_DIR], check=True)
    
    # on force le push explicite du modèle en précisant le chemin
    subprocess.run(["dvc", "push", f"{OUTPUT_DIR}.dvc", "-r", "origin"], check=True)

    print(f"\nModèle promu dans '{OUTPUT_DIR}' et poussé sur DagsHub via DVC.")
    print("Pense à commiter et pousser 'models/camembert.dvc' sur git pour partager cette version à l'équipe.")


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    run = get_candidate_run(client)
    test_f1_micro = run.data.metrics.get("test_f1_micro")
    if test_f1_micro is None:
        raise RuntimeError(
            f"Le run {run.info.run_id} n'a pas de métrique 'test_f1_micro'. "
            "evaluation.py a-t-il bien tourné après train.py ?"
        )

    current_best_f1 = get_current_best_f1()
    is_better = current_best_f1 is None or test_f1_micro > current_best_f1

    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.set_tag("promoted", "true" if is_better else "false")

    if is_better:
        promote(run.info.run_id, test_f1_micro)
    else:
        print(
            f"\nModèle candidat NON retenu : F1 micro {test_f1_micro:.3f} <= modèle actuel ({current_best_f1:.3f}). "
            f"'{OUTPUT_DIR}' reste inchangé."
        )


if __name__ == "__main__":
    main()
