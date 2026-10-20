import os
import subprocess
import sys
import glob
import json

def run(cmd):
    print(f"\n========================================\n[KAGGLE BOT] Exécution : {cmd}\n========================================")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Erreur lors de l'exécution de la commande : {cmd}")
        sys.exit(result.returncode)

# --- BLOC DE DÉBOGAGE ET RECHERCHE DYNAMIQUE ---
print("=== DEBUG KAGGLE INPUTS ===")
if os.path.exists('/kaggle/input/'):
    print("Dossiers dans /kaggle/input/ :", os.listdir('/kaggle/input/'))
    for folder in os.listdir('/kaggle/input/'):
        path = f'/kaggle/input/{folder}'
        if os.path.isdir(path):
            print(f"Contenu de {path} :", os.listdir(path))
else:
    print("Le dossier /kaggle/input/ n'existe pas !")
print("===========================")

# Recherche dynamique du fichier secrets
secret_files = glob.glob("/kaggle/input/**/secrets.json", recursive=True)

if not secret_files:
    print("Erreur : Impossible de trouver un fichier secrets.json dans /kaggle/input/. Le dataset de secrets est-il bien attaché au kernel ?")
    sys.exit(1)

SECRETS_PATH = secret_files[0]
print(f"✅ Fichier de secrets trouvé et sélectionné : {SECRETS_PATH}")
# -----------------------------------------------

try:
    with open(SECRETS_PATH, "r") as f:
        secrets = json.load(f)
    
    os.environ["MLFLOW_TRACKING_URI"] = secrets["MLFLOW_TRACKING_URI"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = secrets["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = secrets["MLFLOW_TRACKING_PASSWORD"]
    
    GITHUB_TOKEN = secrets["GITHUB_TOKEN"]
    GITHUB_USERNAME = secrets["GITHUB_USERNAME"]
    
except Exception as e:
    print(f"Erreur lors de la lecture des secrets : {e}")
    sys.exit(1)

REPO_URL = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/NOV24-ALT-MLOPS-TRUSTPILOT.git"

# on clone le dépôt GitHub (Code toujours à jour !)
run(f"git clone {REPO_URL} repo")
os.chdir("repo")

# on installe des dépendances
run("pip install -r requirements.txt")

# on cofigure DVC et on télécharge le dataset
run("dvc remote modify origin --local user $MLFLOW_TRACKING_USERNAME")
run("dvc remote modify origin --local password $MLFLOW_TRACKING_PASSWORD")
run("dvc pull")

# on entraîne sur le GPU Kaggle
os.environ["FAST_TRACK"] = "false"
os.environ["TRAINING_SOURCE"] = "kaggle"
run("python src/modelisation/train.py")

# on fait l'évvaluation du modèle
run("python src/modelisation/evaluation.py")

# on sauvegarde et envoi le modèle sur dagshub
run("python src/modelisation/deploy.py")

# commit le .dvc mis à jour sur github
run('git config --global user.email "kaggle-bot@actions.com"')
run('git config --global user.name "Kaggle CI Bot"')
run("git add models/camembert.dvc")
run("git commit -m 'Auto-update: Nouveau modèle entraîné par le GPU Kaggle' || echo 'Aucun changement de modèle'")
run("git push")

print("\n🎉 ENTRAÎNEMENT KAGGLE TERMINÉ AVEC SUCCÈS ! Modèle envoyé en production.")