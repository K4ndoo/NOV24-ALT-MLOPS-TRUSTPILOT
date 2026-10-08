import os
import math
import time

import numpy as np
import torch
from prometheus_client import Counter, Histogram
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "models/camembert"


# ============================================================
# MÉTRIQUES PROMETHEUS
# ============================================================

# Nombre total de textes prédits
PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Nombre total de textes prédits"
)

# Nombre total d'erreurs lors des prédictions
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Nombre total d'erreurs lors des prédictions"
)

# Temps nécessaire pour effectuer les prédictions
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Temps nécessaire pour effectuer les prédictions"
)

# Taille des commentaires en nombre de caractères
COMMENT_LENGTH = Histogram(
    "model_comment_length_chars",
    "Longueur des commentaires en nombre de caractères",
    buckets=[
        10,
        25,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
    ]
)

# Répartition des prédictions par label et par classe
PREDICTIONS_BY_LABEL = Counter(
    "model_predictions_by_label",
    "Répartition des prédictions par label",
    ["label", "prediction"]
)


class Predictor:

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.thresholds = None
        self.labels = None

    def load(self):
        model_file_safetensors = os.path.join(
            MODEL_DIR,
            "model.safetensors"
        )

        model_file_bin = os.path.join(
            MODEL_DIR,
            "pytorch_model.bin"
        )

        if (
            not os.path.exists(model_file_safetensors)
            and not os.path.exists(model_file_bin)
        ):
            print(
                "Poids du modèle introuvables localement. "
                "Téléchargement depuis DagsHub (DVC)..."
            )
            self._download_from_dvc()

        print("Chargement du tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR
        )

        print("Chargement du modèle...")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR
        )

        self.thresholds = self.model.config.thresholds
        self.labels = self.model.config.label_names

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Envoi du modèle sur {self.device}...")

        self.model.to(self.device)
        self.model.eval()

        print("Modèle chargé")

    def _download_from_dvc(self):
        import subprocess

        if not os.path.exists(f"{MODEL_DIR}.dvc"):
            raise RuntimeError(
                f"'{MODEL_DIR}.dvc' introuvable. "
                "Le modèle n'a jamais été entraîné/promu, "
                "ou le fichier n'a pas été récupéré via git."
            )

        # Configuration de l'authentification DVC
        subprocess.run(
            [
                "dvc",
                "remote",
                "modify",
                "origin",
                "--local",
                "user",
                os.getenv(
                    "MLFLOW_TRACKING_USERNAME",
                    ""
                ),
            ],
            check=True,
        )

        subprocess.run(
            [
                "dvc",
                "remote",
                "modify",
                "origin",
                "--local",
                "password",
                os.getenv(
                    "MLFLOW_TRACKING_PASSWORD",
                    ""
                ),
            ],
            check=True,
        )

        print(
            "Récupération du modèle promu depuis "
            "DagsHub (dvc pull)..."
        )

        subprocess.run(
            [
                "dvc",
                "pull",
                f"{MODEL_DIR}.dvc"
            ],
            check=True,
        )

        print("Téléchargement terminé avec succès !")

    def predict(self, texts, batch_size=32):

        if self.model is None:
            PREDICTION_ERRORS.inc()

            raise RuntimeError(
                "Le modèle n'est pas chargé. "
                "L'API a-t-elle bien démarré ?"
            )

        # ====================================================
        # DÉBUT MESURE LATENCE
        # ====================================================

        start_time = time.perf_counter()

        try:

            # ====================================================
            # MÉTRIQUE : TAILLE DES COMMENTAIRES
            # ====================================================

            for text in texts:
                COMMENT_LENGTH.observe(len(text))

            # ====================================================
            # NOMBRE TOTAL DE TEXTES
            # ====================================================

            PREDICTIONS_TOTAL.inc(len(texts))

            results = []

            num_batches = math.ceil(
                len(texts) / batch_size
            )

            # ====================================================
            # TRAITEMENT PAR BATCH
            # ====================================================

            for i in range(num_batches):

                batch_texts = texts[
                    i * batch_size:
                    (i + 1) * batch_size
                ]

                # ------------------------------------------------
                # Tokenisation
                # ------------------------------------------------

                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt"
                )

                # ------------------------------------------------
                # Envoi vers CPU / GPU
                # ------------------------------------------------

                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items()
                }

                # ------------------------------------------------
                # Inférence
                # ------------------------------------------------

                with torch.no_grad():

                    logits = self.model(
                        **inputs
                    ).logits

                # ------------------------------------------------
                # Conversion logits → probabilités
                # ------------------------------------------------

                probs = (
                    torch.sigmoid(logits)
                    .cpu()
                    .numpy()
                )

                # =================================================
                # TRAITEMENT DES PRÉDICTIONS
                # =================================================

                for p in probs:

                    result = {}

                    for j, label in enumerate(self.labels):

                        prediction = int(
                            p[j] > self.thresholds[j]
                        )

                        probability = float(p[j])
                        threshold = float(
                            self.thresholds[j]
                        )

                        # -----------------------------------------
                        # MÉTRIQUE :
                        # répartition des prédictions
                        # -----------------------------------------

                        PREDICTIONS_BY_LABEL.labels(
                            label=label,
                            prediction=str(prediction)
                        ).inc()

                        # -----------------------------------------
                        # Résultat API
                        # -----------------------------------------

                        result[label] = {
                            "prediction": prediction,
                            "probability": probability,
                            "threshold": threshold
                        }

                    results.append(result)

            return results

        except Exception:

            # Une erreur pendant la prédiction
            PREDICTION_ERRORS.inc()

            raise

        finally:

            # ====================================================
            # FIN MESURE LATENCE
            # ====================================================

            elapsed_time = (
                time.perf_counter() - start_time
            )

            PREDICTION_LATENCY.observe(
                elapsed_time
            )


# ============================================================
# INSTANCE GLOBALE
# ============================================================

# Chargée au démarrage de FastAPI
predictor = Predictor()
