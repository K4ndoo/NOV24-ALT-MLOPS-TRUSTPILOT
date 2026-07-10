import numpy as np
import torch
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL_DIR = "models/camembert"


class Predictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.thresholds = None
        self.labels = None

    def load(self):
        import os
        model_file_safetensors = os.path.join(MODEL_DIR, "model.safetensors")
        model_file_bin = os.path.join(MODEL_DIR, "pytorch_model.bin")
        if not os.path.exists(model_file_safetensors) and not os.path.exists(model_file_bin):
            print("Poids du modèle introuvables localement. Téléchargement depuis DagsHub (DVC)...")
            self._download_from_dvc()

        print("Chargement du tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        print("Chargement du modèle...")
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.thresholds = self.model.config.thresholds
        self.labels = self.model.config.label_names

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Envoi du modèle sur {self.device}...")
        self.model.to(self.device)
        self.model.eval()

        print("Modèle chargé")

    def _download_from_dvc(self):
        import os
        import subprocess

        if not os.path.exists(f"{MODEL_DIR}.dvc"):
            raise RuntimeError(
                f"'{MODEL_DIR}.dvc' introuvable. Le modèle n'a jamais été entraîné/promu, "
                "ou le fichier n'a pas été récupéré via git."
            )

        # on configure l'auth DVC à partir des identifiants DagsHub déjà utilisés pour MLflow
        subprocess.run(
            ["dvc", "remote", "modify", "origin", "--local", "user", os.getenv("MLFLOW_TRACKING_USERNAME", "")],
            check=True,
        )
        subprocess.run(
            ["dvc", "remote", "modify", "origin", "--local", "password", os.getenv("MLFLOW_TRACKING_PASSWORD", "")],
            check=True,
        )

        print("Récupération du modèle promu depuis DagsHub (dvc pull)...")
        subprocess.run(["dvc", "pull", f"{MODEL_DIR}.dvc"], check=True)
        print("Téléchargement terminé avec succès !")

    def predict(self, texts, batch_size=32):
        if self.model is None:
            raise RuntimeError("Le modèle n'est pas chargé. L'API a-t-elle bien démarré ?")
        
        results = []
        num_batches = math.ceil(len(texts) / batch_size)

        for i in range(num_batches):
            batch_texts = texts[i * batch_size : (i + 1) * batch_size]
            
            # Tokenisation batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.sigmoid(logits).cpu().numpy()

            for p in probs:
                result = {}
                for j, label in enumerate(self.labels):
                    result[label] = {
                        "prediction": int(p[j] > self.thresholds[j]),
                        "probability": float(p[j]),
                        "threshold": float(self.thresholds[j])
                    }
                results.append(result)

        return results


# instance globale (chargée au démarrage)
predictor = Predictor()