import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

MODEL_DVC = "models/camembert.dvc"
TRAINING_DATASET = "data/labelled_dataset/dataset_avis.csv"
CACHE_DIR = Path("reports/cache")
# Message des commits du kernel Kaggle
PROMOTION_SUBJECT = "Nouveau modèle entraîné"


@dataclass(frozen=True)
class Promotion:
    """Commit qui a promu un modèle en production (modification de models/camembert.dvc).

    Le kernel Kaggle ne committe ce fichier que si deploy.py a promu le modèle : le même commit
    contient donc le .dvc exact des données d'entraînement de ce modèle.
    """

    commit: str
    short: str
    date: str
    timestamp: datetime


def _git(*args):
    result = subprocess.run(["git", *args], check=True, capture_output=True, text=True, encoding="utf-8")
    return result.stdout.strip()


def list_promotions():
    """Promotions du modèle, de la plus récente à la plus ancienne."""
    if _git("rev-parse", "--is-shallow-repository") == "true":
        raise RuntimeError(
            "Historique git tronqué (clone superficiel) : impossible de retrouver les références. "
            "En CI, utiliser actions/checkout avec fetch-depth: 0."
        )
    lines = _git("log", "--format=%H|%h|%ad|%aI|%s", "--date=short", "--", MODEL_DVC).splitlines()
    promotions = []
    for line in lines:
        commit, short, day, iso, subject = line.split("|", 4)
        if PROMOTION_SUBJECT in subject:
            promotions.append(Promotion(commit, short, day, datetime.fromisoformat(iso).replace(tzinfo=None)))
    if not promotions:
        raise RuntimeError(f"Aucun commit de promotion trouvé pour {MODEL_DVC}.")
    return promotions


def fetch_reference(promotion):
    """Télécharge le jeu d'entraînement de cette promotion dans reports/cache/ (jamais dans data/)."""
    path = CACHE_DIR / f"reference_{promotion.short}.csv"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "dvc", "get", ".", TRAINING_DATASET, "--rev", promotion.commit, "-o", str(path)],
            check=True,
        )
    return path
