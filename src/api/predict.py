import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "./../../models/camembert"


class Predictor:

    def __init__(self):
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

    def predict(self, texts):

        # Tokenisation batch
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.sigmoid(logits).cpu().numpy()

        results = []

        for p in probs:
            result = {}

            for i, label in enumerate(self.labels):
                result[label] = {
                    "prediction": int(p[i] > self.thresholds[i]),
                    "probability": float(p[i]),
                    "threshold": float(self.thresholds[i])
                }

            results.append(result)

        return results


# instance globale (chargée au démarrage)
predictor = Predictor()