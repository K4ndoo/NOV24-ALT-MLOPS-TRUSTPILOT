import numpy as np
import pandas as pd
import torch
import mlflow
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.metrics import f1_score


# CONFIGURATION
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("TrustPilot_CamemBERT_Classification")

MODEL_NAME = "camembert-base"
TEXT_COL = "clean_comment"
LABEL_COLS = ["qualité produit", "service livraison", "service client"]

USE_FP16 = torch.cuda.is_available() # demi-précision automatique si GPU dispo


# GLOBAL COMPONENTS FOR MULTIPROCESSING
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch[TEXT_COL], truncation=True, max_length=256)

def pack_labels(batch):
    batch["labels"] = np.stack(
        [batch[col] for col in LABEL_COLS], axis=1
    ).astype(np.float32)
    return batch

def prepare(ds):
    ds = ds.map(tokenize,     batched=True, num_proc=4) # num_proc=1 si sur Windows ou si problèmes de multiprocessing
    ds = ds.map(pack_labels,  batched=True, num_proc=4) # num_proc=1 si sur Windows ou si problèmes de multiprocessing
    ds = ds.remove_columns(LABEL_COLS + [TEXT_COL])
    ds.set_format("torch")
    return ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }

def main():
    # LOAD DATA
    df = pd.read_csv('data/labelled_dataset/dataset_avis.csv')

    # PRÉ-PROCESSING DES NOUVEAUX AVIS (SCRAPING)
    import re
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        text = re.sub(r"(http|www)\S+", "", text) # URLs
        text = re.sub(r"\S+@\S+", "", text) # emails
        text = re.sub(r"(@|#)\w+", "", text) # mentions @ et #
        text = re.sub(r"\s+", " ", text).strip() # espaces multiples
        return text

    mask = df['clean_comment'].isna() | (df['clean_comment'] == "")
    if mask.any():
        df.loc[mask, 'clean_comment'] = df.loc[mask, 'Commentaire'].apply(clean_text)
        # On sauvegarde le dataset pour que evaluation.py puisse lire les textes nettoyés
        df.to_csv('data/labelled_dataset/dataset_avis.csv', index=False)

    # MODE DÉMO (Pour la soutenance)
    FAST_TRACK = os.getenv("FAST_TRACK", "false").lower() == "true"
    if FAST_TRACK:
        print("MODE FAST_TRACK ACTIVÉ : Entraînement sur 50 exemples (Démo) !")
        df = df.head(50)

    dataset = Dataset.from_pandas(df[[TEXT_COL] + LABEL_COLS])

    # Le split de test n'est pas utilisé ici : il est reconstruit (même seed) et évalué
    # indépendamment par evaluation.py, qui est seul responsable du calcul des métriques.
    dataset = dataset.train_test_split(test_size=0.2,  seed=42)
    dataset = dataset["train"].train_test_split(test_size=0.15, seed=42)
    train_ds = dataset["train"]
    val_ds = dataset["test"]


    # TOKENISATION
    train_ds = prepare(train_ds)
    val_ds = prepare(val_ds)

    data_collator = DataCollatorWithPadding(tokenizer)


    # MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    )


    # TRAINING & MLFLOW TRACKING
    with mlflow.start_run(run_name="Camembert_Base_Run1"), tempfile.TemporaryDirectory() as tmp_dir:

        training_args = TrainingArguments(
            output_dir=tmp_dir,

            # Évaluation & sauvegarde
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",

            # Hyperparamètres
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1 if FAST_TRACK else 10,
            weight_decay=0.01,

            # Affichage dans Airflow (force HuggingFace à écrire des logs texte souvent)
            logging_strategy="steps",
            logging_steps=10 if not FAST_TRACK else 2,

            # Vitesse
            fp16=USE_FP16,
            dataloader_num_workers=4, # mettre à 0 si sur Windows ou si problèmes de multiprocessing
            # Logging
            report_to="none",
        )

        # on log les hyperparamètres manuellement
        mlflow.log_params({
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "epochs": training_args.num_train_epochs,
            "weight_decay": training_args.weight_decay,
            "model_name": MODEL_NAME
        })
        
        # Log de la taille du dataset pour suivre son enrichissement via le scraper
        mlflow.log_metric("dataset_size", len(df))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()


        # FIND BEST THRESHOLDS
        predictions = trainer.predict(val_ds)
        val_logits = predictions.predictions
        val_labels = predictions.label_ids
        val_probs = 1 / (1 + np.exp(-val_logits))

        thresholds = np.arange(0.1, 0.91, 0.01)
        best_thresholds = []

        for i in range(val_labels.shape[1]):
            preds_matrix = (val_probs[:, i : i + 1] > thresholds).astype(int)
            f1_scores = [
                f1_score(val_labels[:, i], preds_matrix[:, j], zero_division=0)
                for j, _ in enumerate(thresholds)
            ]
            best_thresholds.append(float(thresholds[np.argmax(f1_scores)]))

        print("Best thresholds:", best_thresholds)

        # on log les thresholds dans MLflow
        for i, col in enumerate(LABEL_COLS):
            mlflow.log_metric(f"threshold_{col}", best_thresholds[i])


        # Nettoyage des checkpoints intermédiaires très lourds (450Mo chacun) générés par HuggingFace
        # pendant l'entraînement, sinon ils se retrouvent dans l'artefact loggé dans MLflow
        import shutil
        import glob
        for checkpoint_dir in glob.glob(f"{tmp_dir}/checkpoint-*"):
            shutil.rmtree(checkpoint_dir)

        # SAVE MODEL + THRESHOLDS (candidat : uniquement dans MLflow. L'évaluation (test split + Gold)
        # est faite par evaluation.py, et la promotion vers models/camembert + DVC par deploy.py)
        model.config.thresholds = best_thresholds
        model.config.label_names = LABEL_COLS

        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        mlflow.log_artifacts(tmp_dir, artifact_path="camembert_model")

    print("\nEntraînement terminé. Modèle candidat loggé dans MLflow.")
    print("Lance evaluation.py puis deploy.py pour évaluer et éventuellement promouvoir ce candidat.")

if __name__ == "__main__":
    main()
