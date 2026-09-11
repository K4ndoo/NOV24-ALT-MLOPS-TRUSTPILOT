import os
import subprocess
import sys

def run(cmd):
    print(f"\n========================================\n[KAGGLE BOT] Exécution : {cmd}\n========================================")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Erreur lors de l'exécution de la commande : {cmd}")
        sys.exit(result.returncode)

# 1. Récupération des Secrets depuis le fichier injecté par GitHub Actions
try:
    import json
    with open("secrets.json", "r") as f:
        secrets = json.load(f)
    
    os.environ["MLFLOW_TRACKING_URI"] = secrets["MLFLOW_TRACKING_URI"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = secrets["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = secrets["MLFLOW_TRACKING_PASSWORD"]
    
    GITHUB_TOKEN = secrets["GITHUB_TOKEN"]
    GITHUB_USERNAME = secrets["GITHUB_USERNAME"]
    
except Exception as e:
    print("Erreur : Impossible de lire secrets.json. L'injection depuis GitHub Actions a échoué.")
    sys.exit(1)

REPO_URL = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/NOV24-ALT-MLOPS-TRUSTPILOT.git"

# 2. Cloner le dépôt GitHub (Code toujours à jour !)
run(f"git clone {REPO_URL} repo")
os.chdir("repo")

# 3. Installation des dépendances
run("pip install -r requirements.txt")

# 4. Configuration de DVC et téléchargement du Dataset
run("dvc remote modify origin --local user $MLFLOW_TRACKING_USERNAME")
run("dvc remote modify origin --local password $MLFLOW_TRACKING_PASSWORD")
run("dvc pull")

# 5. Entraînement sur le GPU Kaggle ! (Sans le FAST_TRACK, on entraîne pour de vrai)
os.environ["FAST_TRACK"] = "false"
run("python src/modelisation/train.py")

# 6. Évaluation du modèle
run("python src/modelisation/evaluation.py")

# 7. Sauvegarde et envoi du modèle sur DagsHub
run("python src/modelisation/deploy.py")
run("dvc push")

# 8. Commit du .dvc mis à jour sur GitHub
run('git config --global user.email "kaggle-bot@actions.com"')
run('git config --global user.name "Kaggle CI Bot"')
run("git add models/camembert.dvc")
run("git commit -m 'Auto-update: Nouveau modèle entraîné par le GPU Kaggle' || echo 'Aucun changement de modèle'")
run("git push")

print("\n🎉 ENTRAÎNEMENT KAGGLE TERMINÉ AVEC SUCCÈS ! Modèle envoyé en production.")
