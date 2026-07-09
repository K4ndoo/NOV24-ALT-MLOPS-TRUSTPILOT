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
            print("Poids du modèle introuvables localement. Téléchargement depuis MLflow (Dagshub)...")
            self._download_from_mlflow()

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

    def _download_from_mlflow(self):
        import os
        import mlflow
        from mlflow.tracking.client import MlflowClient
        import tempfile
        import shutil

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        
        experiment_name = "TrustPilot_CamemBERT_Classification"
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise RuntimeError(f"L'expérience MLflow '{experiment_name}' est introuvable.")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], 
            order_by=["attributes.start_time DESC"], 
            max_results=1
        )
        if not runs:
            raise RuntimeError("Aucun run trouvé dans MLflow. Le modèle n'a jamais été entraîné.")
        
        run_id = runs[0].info.run_id
        print(f"Téléchargement du modèle depuis le run MLflow: {run_id}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            download_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, 
                artifact_path="camembert_model", 
                dst_path=tmpdir
            )
            os.makedirs(MODEL_DIR, exist_ok=True)
            for item in os.listdir(download_path):
                s = os.path.join(download_path, item)
                d = os.path.join(MODEL_DIR, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
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