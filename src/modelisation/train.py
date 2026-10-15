import os
import re
import glob
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
import mlflow

from dotenv import load_dotenv
from datasets import Dataset
from sklearn.metrics import f1_score

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
)
mlflow.set_experiment("TrustPilot_CamemBERT_Classification")

MODEL_NAME = "camembert-base"

TEXT_COL = "clean_comment"
RAW_TEXT_COL = "Commentaire"

LABEL_COLS = [
    "qualité produit",
    "service livraison",
    "service client",
]

DATA_PATH = "data/labelled_dataset/dataset_avis.csv"

USE_FP16 = torch.cuda.is_available()

FAST_TRACK = (
    os.getenv("FAST_TRACK", "false").lower() == "true"
)

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def clean_text(text):
    """Nettoie un commentaire client."""
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"(http|www)\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"(@|#)\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(batch):
    """Tokenise les commentaires."""
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        max_length=256,
    )


def pack_labels(batch):
    """Construit les labels multi-labels."""
    batch["labels"] = np.stack(
        [batch[col] for col in LABEL_COLS],
        axis=1,
    ).astype(np.float32)

    return batch


def prepare(ds):
    """Prépare un dataset pour Transformers."""
    ds = ds.map(
        tokenize,
        batched=True,
        num_proc=1,
    )

    ds = ds.map(
        pack_labels,
        batched=True,
        num_proc=1,
    )

    ds = ds.remove_columns(
        LABEL_COLS + [TEXT_COL]
    )

    ds.set_format("torch")

    return ds


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):
    """Calcule les scores F1 micro et macro."""
    logits, labels = eval_pred

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    return {
        "f1_micro": f1_score(
            labels,
            preds,
            average="micro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            labels,
            preds,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# DATA PREPROCESSING
# ============================================================

def load_and_prepare_dataframe():
    """Charge et prépare les données d'entraînement."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Vérification des colonnes nécessaires
    required_cols = LABEL_COLS + [RAW_TEXT_COL]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans le dataset : {missing_cols}"
        )

    # Initialiser clean_comment comme une colonne texte.
    # Évite l'erreur Pandas d'affectation dans une colonne float64.
    if TEXT_COL not in df.columns:
        df[TEXT_COL] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )
    else:
        df[TEXT_COL] = df[TEXT_COL].astype("string")

    # Identifier les commentaires à nettoyer
    mask = (
        df[TEXT_COL].isna()
        | df[TEXT_COL].str.strip().eq("").fillna(False)
    )

    if mask.any():
        df.loc[mask, TEXT_COL] = (
            df.loc[mask, RAW_TEXT_COL]
            .apply(clean_text)
            .astype("string")
        )

        # Persister les textes nettoyés pour evaluation.py
        df.to_csv(DATA_PATH, index=False)

        print(
            f"{int(mask.sum())} commentaires nettoyés "
            f"et sauvegardés."
        )

    # Éliminer les commentaires vides
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df = df[df[TEXT_COL].str.strip().ne("")].copy()

    # Convertir les labels en valeurs numériques
    for col in LABEL_COLS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Éliminer les exemples dont les labels sont manquants
    df = df.dropna(subset=LABEL_COLS).copy()

    # Vérifier que les labels sont binaires
    for col in LABEL_COLS:
        invalid = ~df[col].isin([0, 1])

        if invalid.any():
            raise ValueError(
                f"La colonne '{col}' contient des valeurs "
                "différentes de 0 et 1."
            )

        df[col] = df[col].astype("int64")

    if len(df) < 5:
        raise ValueError(
            "Dataset insuffisant après nettoyage : "
            f"{len(df)} exemples disponibles."
        )

    print(f"Nombre d'exemples disponibles : {len(df)}")

    return df


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Chargement des données
    # --------------------------------------------------------

    df = load_and_prepare_dataframe()

    if FAST_TRACK:
        print(
            "MODE FAST_TRACK ACTIVÉ : "
            "entraînement de démonstration sur 50 exemples."
        )
        df = df.head(50).copy()

    if len(df) < 5:
        raise ValueError(
            "Pas assez d'exemples pour effectuer le split."
        )

    # --------------------------------------------------------
    # 2. Création des datasets
    # --------------------------------------------------------

    dataset = Dataset.from_pandas(
        df[[TEXT_COL] + LABEL_COLS],
        preserve_index=False,
    )

    split = dataset.train_test_split(
        test_size=0.2,
        seed=42,
    )

    train_val = split["train"].train_test_split(
        test_size=0.15,
        seed=42,
    )

    train_ds = prepare(train_val["train"])
    val_ds = prepare(train_val["test"])

    print(f"Taille train : {len(train_ds)}")
    print(f"Taille validation : {len(val_ds)}")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # 3. Chargement du modèle
    # --------------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    )

    # --------------------------------------------------------
    # 4. Paramètres d'entraînement
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="Camembert_Base_Run1"
    ), tempfile.TemporaryDirectory() as tmp_dir:

        training_args = TrainingArguments(
            output_dir=tmp_dir,

            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",
            greater_is_better=True,

            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,

            num_train_epochs=(
                1 if FAST_TRACK else 10
            ),

            weight_decay=0.01,

            logging_strategy="steps",
            logging_steps=(
                2 if FAST_TRACK else 10
            ),

            fp16=USE_FP16,
            dataloader_num_workers=0,

            report_to="none",
            save_total_limit=1,

        )

        mlflow.log_params({
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "epochs": training_args.num_train_epochs,
            "weight_decay": training_args.weight_decay,
            "model_name": MODEL_NAME,
            "fast_track": FAST_TRACK,
            "dataset_size": len(df),
        })

        mlflow.log_metric(
            "dataset_size",
            len(df),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=2
                )
            ],
        )

        # ----------------------------------------------------
        # 5. Entraînement
        # ----------------------------------------------------

        trainer.train()

        # ----------------------------------------------------
        # 6. Recherche des meilleurs seuils
        # ----------------------------------------------------

        predictions = trainer.predict(val_ds)

        val_logits = predictions.predictions
        val_labels = predictions.label_ids

        val_probs = 1.0 / (
            1.0 + np.exp(-val_logits)
        )

        thresholds = np.arange(
            0.1,
            0.91,
            0.01,
        )

        best_thresholds = []

        for i in range(val_labels.shape[1]):
            scores = []

            for threshold in thresholds:
                preds = (
                    val_probs[:, i] > threshold
                ).astype(int)

                score = f1_score(
                    val_labels[:, i],
                    preds,
                    zero_division=0,
                )

                scores.append(score)

            best_threshold = float(
                thresholds[np.argmax(scores)]
            )

            best_thresholds.append(
                best_threshold
            )

        print(
            "Meilleurs seuils :",
            best_thresholds,
        )

        for i, col in enumerate(LABEL_COLS):
            mlflow.log_metric(
                f"threshold_{i}",
                best_thresholds[i],
            )

        # ----------------------------------------------------
        # 7. Sauvegarde du modèle dans MLflow
        # ----------------------------------------------------

        # Supprimer les checkpoints intermédiaires
        for checkpoint_dir in glob.glob(
            os.path.join(tmp_dir, "checkpoint-*")
        ):
            shutil.rmtree(checkpoint_dir)

        model = trainer.model

        model.config.thresholds = best_thresholds
        model.config.label_names = LABEL_COLS

        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        mlflow.log_artifacts(
            tmp_dir,
            artifact_path="camembert_model",
        )

    print(
        "\nEntraînement terminé. "
        "Modèle candidat enregistré dans MLflow."
    )

    print(
        "Lance evaluation.py puis deploy.py pour évaluer "
        "et éventuellement promouvoir ce candidat."
    )


if __name__ == "__main__":
    main()
