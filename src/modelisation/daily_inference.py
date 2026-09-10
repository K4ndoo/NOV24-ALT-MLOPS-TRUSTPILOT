import os
import numpy as np
import pandas as pd
import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

MODEL_DIR = "models/camembert"
current_date = datetime.now().strftime("%Y-%m-%d")
DATASET_PATH = f"data/daily_inference/dataset_inference_{current_date}.csv"
TEXT_COL = "clean_comment"
BATCH_SIZE = 32
EXPERIMENT_NAME = "TrustPilot_Daily_Monitoring"

def main():
    if not os.path.exists(MODEL_DIR):
        raise RuntimeError(f"Le modèle introuvable dans {MODEL_DIR}. Lance 'dvc pull' d'abord.")
    
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError(f"Le dataset d'inférence est introuvable ({DATASET_PATH}). Lance le scraper d'abord.")

    print("Chargement du dataset quotidien...")
    df = pd.read_csv(DATASET_PATH)
    if len(df) == 0:
        raise ValueError("🚨 CRITICAL ERROR: 0 avis récupérés aujourd'hui ! Le scraper a échoué (blocage WAF Trustpilot ou changement de design). L'inférence est annulée.")
        
    import re
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        text = re.sub(r"(http|www)\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"(@|#)\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df[TEXT_COL] = df["Commentaire"].apply(clean_text)

    print(f"Chargement du modèle depuis {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    thresholds = np.array(model.config.thresholds)
    label_cols = model.config.label_names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    def tokenize(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=256)

    def pack_labels(batch):
        batch["labels"] = np.stack(
            [batch[col] for col in label_cols], axis=1
        ).astype(np.float32)
        return batch

    print("Préparation des données...")
    ds = Dataset.from_pandas(df[[TEXT_COL] + label_cols])
    ds = ds.map(tokenize, batched=True, num_proc=1)
    ds = ds.map(pack_labels, batched=True, num_proc=1)
    ds = ds.remove_columns(label_cols + [TEXT_COL])
    ds.set_format("torch")

    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=data_collator)

    all_labels = []
    all_probs = []

    print("Inférence en cours...")
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(labels.numpy())
            all_probs.append(probs)

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    preds = (all_probs > thresholds).astype(int)

    print("\n===== RÉSULTATS DU JOUR =====")
    print(classification_report(all_labels, preds, target_names=label_cols, zero_division=0))

    f1_micro = f1_score(all_labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_labels, preds, average="macro", zero_division=0)
    
    print(f"F1 micro du jour: {f1_micro:.3f} | F1 macro du jour: {f1_macro:.3f}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("CRITICAL ERROR: La variable MLFLOW_TRACKING_URI est manquante ! Les métriques seraient enregistrées localement et perdues. Vérifiez vos Secrets GitHub.")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Envoi des métriques à DagsHub (MLflow)...")
    with mlflow.start_run(run_name=f"Daily_Inference_{pd.Timestamp.now().strftime('%Y-%m-%d')}"):
        mlflow.log_metrics({
            "daily_f1_micro": f1_micro,
            "daily_f1_macro": f1_macro,
            "reviews_scraped": len(df)
        })
        
    print("Inférence terminée avec succès !")

if __name__ == "__main__":
    main()
