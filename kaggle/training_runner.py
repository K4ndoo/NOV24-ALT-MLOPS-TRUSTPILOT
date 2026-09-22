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

# Recherche dynamique du fichier secrets (gère les cas où Kaggle ajoute un suffixe -1, -2...)
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