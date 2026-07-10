import os
import tempfile
import numpy as np
import pandas as pd
import torch
import mlflow
from mlflow.tracking.client import MlflowClient
from dotenv import load_dotenv

load_dotenv()

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# CONFIG

TEXT_COL = "clean_comment"
BATCH_SIZE = 32
EXPERIMENT_NAME = "TrustPilot_CamemBERT_Classification"


def get_latest_candidate(tmp_dir):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise RuntimeError(f"Expérience MLflow '{EXPERIMENT_NAME}' introuvable.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("Aucun run trouvé dans MLflow. Lance d'abord train.py.")

    run_id = runs[0].info.run_id
    print(f"Récupération du modèle candidat depuis le run MLflow : {run_id}")
    model_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="camembert_model", dst_path=tmp_dir
    )
    return model_dir, run_id


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir, run_id = get_latest_candidate(tmp_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

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

        def prepare(df):
            ds = Dataset.from_pandas(df[[TEXT_COL] + label_cols])
            ds = ds.map(tokenize,    batched=True, num_proc=4)
            ds = ds.map(pack_labels, batched=True, num_proc=4)
            ds = ds.remove_columns(label_cols + [TEXT_COL])
            ds.set_format("torch")
            return ds

        def predict(ds):
            loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=data_collator)

            all_labels = []
            all_probs = []

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

            return all_labels, preds

        def evaluate(name, df):
            print(f"\n===== {name} =====")

            ds = prepare(df)
            labels, preds = predict(ds)

            print(classification_report(labels, preds, target_names=label_cols, zero_division=0))

            for i, label in enumerate(label_cols):
                cm = confusion_matrix(labels[:, i], preds[:, i])
                print(f"\nMatrice de confusion — {label}")
                print(cm)

            f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
            f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
            print(f"F1 micro: {f1_micro:.3f} | F1 macro: {f1_macro:.3f}")
            return f1_micro, f1_macro

        # LOAD DATASETS
        df_main = pd.read_csv('data/labelled_dataset/dataset_avis.csv')
        df_gold = pd.read_csv('data/test_dataset/100_avis_annote.csv', sep=";")

        dataset = Dataset.from_pandas(df_main[[TEXT_COL] + label_cols])
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        df_test = dataset["test"].to_pandas()

        # EVALUATION
        # Split de test : c'est la métrique utilisée par deploy.py pour décider de la promotion.
        test_f1_micro, test_f1_macro = evaluate("TEST SPLIT", df_test)

        # Jeu annoté à la main : indicatif uniquement, ne sert jamais à la décision de promotion.
        gold_f1_micro, gold_f1_macro = evaluate("GOLD DATASET", df_gold)

        with mlflow.start_run(run_id=run_id):
            # on logue le split de test en priorité : c'est la métrique de décision
            mlflow.log_metrics({
                "test_f1_micro": test_f1_micro,
                "test_f1_macro": test_f1_macro,
            })
            # le Gold est loggé aussi, mais à titre indicatif uniquement
            mlflow.log_metrics({
                "gold_f1_micro": gold_f1_micro,
                "gold_f1_macro": gold_f1_macro,
            })

        print(f"\nF1 micro (test split, utilisé pour la décision de promotion) : {test_f1_micro:.3f}")
        print(f"F1 micro (Gold, indicatif) : {gold_f1_micro:.3f}")


if __name__ == "__main__":
    main()
