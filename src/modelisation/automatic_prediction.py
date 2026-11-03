
import os
import time
import random
import argparse
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[2]

# API FastAPI
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/") + "/predict"
API_KEY = os.getenv("API_KEY")

# Dataset d'inférence
INPUT_FILE = (
    BASE_DIR / "data/daily_inference/dataset_inference.csv"
)

# Dataset d'entraînement
TRAIN_FILE = (
    BASE_DIR / "data/labelled_dataset/dataset_avis.csv"
)

# Résultats des prédictions
OUTPUT_FILE = (
    BASE_DIR / "data/daily_inference/predictions.csv"
)

# Colonnes
TEXT_COL = "clean_comment"

LABEL_COLS = [
    "qualité produit",
    "service livraison",
    "service client"
]

# Paramètres d'inférence
BATCH_SIZE = 25
TIMEOUT = 120
MAX_RETRIES = 3

# Identifiant unique de cette exécution
RUN_ID = str(uuid.uuid4())


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. VALIDATION DE LA CONFIGURATION
# ============================================================

def validate_config():

    if not API_KEY:
        raise ValueError(
            "API_KEY introuvable dans le fichier .env"
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {INPUT_FILE}"
        )

    if not 20 <= BATCH_SIZE <= 32:
        raise ValueError(
            "La taille des lots doit être comprise entre 20 et 32."
        )

    logger.info("Configuration validée.")


# ============================================================
# 4. VERIFICATION DE L'API
# ============================================================

def check_api_health():

    health_url = API_URL.replace(
        "/predict",
        "/health"
    )

    try:

        response = requests.get(
            health_url,
            timeout=10
        )

        response.raise_for_status()

        health = response.json()

        if health.get("status") != "ok":
            raise RuntimeError("API indisponible.")

        if not health.get("model_loaded"):
            raise RuntimeError(
                "Le modèle CamemBERT n'est pas chargé."
            )

        logger.info(
            "API opérationnelle. Modèle CamemBERT chargé."
        )

    except requests.RequestException as error:

        raise RuntimeError(
            f"Impossible de contacter l'API : {error}"
        ) from error


# ============================================================
# 5. CHARGEMENT DU DATASET
# ============================================================


def load_dataset(input_file=INPUT_FILE):

    logger.info("Chargement du dataset : %s", input_file)

    df = pd.read_csv(
        input_file,
        sep=None,
        engine="python"
    )

    logger.info("Colonnes détectées : %s", list(df.columns))

    # Vérification de la présence d'une colonne de texte
    if TEXT_COL not in df.columns and "Commentaire" not in df.columns:
        raise ValueError(
            "Aucune colonne de texte trouvée : "
            "clean_comment ou Commentaire."
        )

    # Conversion explicite en chaînes Python
    if TEXT_COL not in df.columns:
        df[TEXT_COL] = ""

    df[TEXT_COL] = df[TEXT_COL].astype("object")

    if "Commentaire" in df.columns:

        commentaires = df["Commentaire"].astype("object")

        mask = (
            df[TEXT_COL].isna()
            | df[TEXT_COL].astype(str).str.strip().isin(
                ["", "nan", "None", "<NA>"]
            )
        )

        # Remplacement sans incompatibilité de dtype
        df[TEXT_COL] = df[TEXT_COL].where(
            ~mask,
            commentaires
        )

    # Nettoyage des textes
    df[TEXT_COL] = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Suppression des textes vides
    df = df[df[TEXT_COL] != ""].copy()

    # Vérification des métadonnées
    metadata_cols = [
        "comment_id",
        "company",
        "star",
        "date"
    ]

    for col in metadata_cols:
        if col not in df.columns:
            df[col] = None

    # Conservation de l'index original
    df = df.reset_index()
    df.rename(
        columns={"index": "source_index"},
        inplace=True
    )

    logger.info(
        "%d avis prêts pour l'inférence.",
        len(df)
    )

    return df

# ============================================================
# 6. ENVOI DES LOTS VERS FASTAPI
# ============================================================

def predict_batch(session, texts):

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "texts": texts
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.post(
                API_URL,
                json=payload,
                headers=headers,
                timeout=TIMEOUT
            )

            response.raise_for_status()

            data = response.json()

            predictions = data.get("predictions")

            if not isinstance(predictions, list):
                raise ValueError(
                    "Format de réponse API invalide."
                )

            if len(predictions) != len(texts):
                raise ValueError(
                    "Le nombre de prédictions ne correspond "
                    "pas au nombre d'avis envoyés."
                )

            return predictions

        except requests.RequestException:

            logger.exception(
                "Erreur API : tentative %d/%d",
                attempt,
                MAX_RETRIES
            )

            if attempt == MAX_RETRIES:
                raise

            time.sleep(2 ** attempt)

    raise RuntimeError("Échec des prédictions.")


# ============================================================
# 7. VERIFICATION ET TRANSFORMATION DES RESULTATS
# ============================================================

def format_prediction(prediction):

    result = {}

    for label in LABEL_COLS:

        if label not in prediction:
            raise ValueError(
                f"Label absent de la réponse API : {label}"
            )

        values = prediction[label]

        required_fields = [
            "prediction",
            "probability",
            "threshold"
        ]

        for field in required_fields:

            if field not in values:
                raise ValueError(
                    f"Champ '{field}' absent pour {label}"
                )

        result[f"{label}_prediction"] = int(
            values["prediction"]
        )

        result[f"{label}_probability"] = float(
            values["probability"]
        )

        result[f"{label}_threshold"] = float(
            values["threshold"]
        )

    return result


# ============================================================
# 8. SAUVEGARDE DES RESULTATS
# ============================================================

def save_results(results):

    if not results:
        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df = pd.DataFrame(results)

    # Ajout des résultats sans écraser les exécutions précédentes
    results_df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not OUTPUT_FILE.exists(),
        index=False,
        encoding="utf-8"
    )


# ============================================================
# 9. PIPELINE PRINCIPALE
# ============================================================

def run_inference():

    validate_config()

    check_api_health()

    df = load_dataset()

    total = len(df)

    if total == 0:

        logger.warning(
            "Aucun avis disponible pour les prédictions."
        )

        return

    successful = 0
    failed = 0

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("DEMARRAGE DE LA PIPELINE D'INFERENCE")
    logger.info("RUN ID : %s", RUN_ID)
    logger.info("TOTAL AVIS : %d", total)
    logger.info("TAILLE DES LOTS : %d", BATCH_SIZE)
    logger.info("=" * 60)

    with requests.Session() as session:

        # Découpage en lots
        for start in range(0, total, BATCH_SIZE):

            batch = df.iloc[
                start:start + BATCH_SIZE
            ]

            texts = batch[TEXT_COL].tolist()

            batch_id = start // BATCH_SIZE + 1

            batch_start = time.time()

            logger.info(
                "Envoi du lot %d : %d avis",
                batch_id,
                len(texts)
            )

            try:

                predictions = predict_batch(
                    session,
                    texts
                )

                results = []

                for (_, row), prediction in zip(
                    batch.iterrows(),
                    predictions
                ):

                    formatted = format_prediction(
                        prediction
                    )

                    result = {
                        "run_id": RUN_ID,
                        "comment_id": row["comment_id"],
                        "company": row["company"],
                        "star": row["star"],
                        "date": row["date"],
                        "source_index": row["source_index"],
                        "clean_comment": row[TEXT_COL],
                        "batch_id": batch_id,
                        "status": "success",
                        "inference_date": datetime.now(
                            timezone.utc
                        ).isoformat()
                    }

                    result.update(formatted)

                    results.append(result)

                save_results(results)

                successful += len(results)

                logger.info(
                    "Lot %d terminé : %d avis traités en %.2f secondes.",
                    batch_id,
                    len(results),
                    time.time() - batch_start
                )

            except (
                requests.RequestException,
                ValueError,
                KeyError,
                TypeError
            ) as error:

                logger.exception(
                    "Échec du lot %d : %s",
                    batch_id,
                    error
                )

                failed += len(batch)

                error_results = []

                for _, row in batch.iterrows():

                    error_results.append({
                        "run_id": RUN_ID,
                        "comment_id": row["comment_id"],
                        "company": row["company"],
                        "star": row["star"],
                        "source_index": row["source_index"],
                        "clean_comment": row[TEXT_COL],
                        "batch_id": batch_id,
                        "status": "error",
                        "error": str(error),
                        "inference_date": datetime.now(
                            timezone.utc
                        ).isoformat()
                    })

                save_results(error_results)

    # ========================================================
    # 10. BILAN FINAL
    # ========================================================

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("BILAN FINAL DES PREDICTIONS")
    logger.info("=" * 60)

    logger.info("Total des avis : %d", total)
    logger.info("Prédictions réussies : %d", successful)
    logger.info("Prédictions échouées : %d", failed)

    logger.info(
        "Taux de réussite : %.2f%%",
        (successful / total) * 100
    )

    logger.info(
        "Durée totale : %.2f secondes",
        elapsed
    )

    if elapsed > 0:

        logger.info(
            "Débit : %.2f avis/seconde",
            successful / elapsed
        )

    logger.info(
        "Fichier de sortie : %s",
        OUTPUT_FILE
    )

    logger.info("=" * 60)


# ============================================================
# 11. SIMULATION DE TRAFIC
# ============================================================

SIMULATION_SOURCES = {
    "train": [TRAIN_FILE],
    "inference": [INPUT_FILE],
    "all": [TRAIN_FILE, INPUT_FILE]
}


def load_simulation_texts(source):

    texts = []

    for input_file in SIMULATION_SOURCES[source]:

        if not input_file.exists():
            logger.warning("Dataset introuvable, ignoré : %s", input_file)
            continue

        texts.extend(load_dataset(input_file)[TEXT_COL].tolist())

    if not texts:
        raise FileNotFoundError(
            f"Aucun avis disponible pour la source '{source}'."
        )

    return texts


def run_simulation(interval, batch_size, source):

    # Envoie en boucle des avis tirés au hasard vers l'API, sans rien sauvegarder :
    # le but est uniquement de générer du trafic visible dans Prometheus / Grafana

    if not API_KEY:
        raise ValueError(
            "API_KEY introuvable dans le fichier .env"
        )

    texts = load_simulation_texts(source)

    logger.info("=" * 60)
    logger.info("DEMARRAGE DE LA SIMULATION DE TRAFIC")
    logger.info("API : %s", API_URL)
    logger.info("SOURCE : %s (%d avis)", source, len(texts))
    logger.info("INTERVALLE : %.1f s | AVIS PAR REQUETE : %d", interval, batch_size)
    logger.info("=" * 60)

    tick = 0

    with requests.Session() as session:

        while True:

            tick += 1
            tick_start = time.time()
            batch = random.sample(texts, min(batch_size, len(texts)))

            try:

                predictions = predict_batch(session, batch)

                labels = [
                    label
                    for label, values in predictions[0].items()
                    if values["prediction"] == 1
                ]

                logger.info(
                    "Requête %d : %d avis prédits en %.2f s | 1er avis -> %s",
                    tick,
                    len(batch),
                    time.time() - tick_start,
                    ", ".join(labels) or "aucune catégorie"
                )

            except (
                requests.RequestException,
                ValueError,
                KeyError,
                TypeError
            ) as error:

                logger.error(
                    "Requête %d en échec : %s",
                    tick,
                    error
                )

            time.sleep(
                max(0.0, interval - (time.time() - tick_start))
            )


def parse_args():

    parser = argparse.ArgumentParser(
        description="Prédictions automatiques via l'API FastAPI."
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Envoie des avis aléatoires à intervalle régulier (au lieu du traitement batch)."
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Secondes entre deux requêtes (mode simulation)."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Nombre d'avis envoyés par requête (mode simulation)."
    )

    parser.add_argument(
        "--source",
        choices=list(SIMULATION_SOURCES),
        default="all",
        help="Dataset d'où sont tirés les avis (mode simulation)."
    )

    return parser.parse_args()


# ============================================================
# 12. LANCEMENT
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    try:

        if args.simulate:

            run_simulation(
                args.interval,
                args.batch_size,
                args.source
            )

        else:

            run_inference()

    except KeyboardInterrupt:

        logger.info("Arrêt demandé par l'utilisateur.")

    except Exception:

        logger.exception(
            "ERREUR CRITIQUE DE LA PIPELINE D'INFERENCE"
        )

        raise
