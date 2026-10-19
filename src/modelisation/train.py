import os
import re
import glob
import shutil
import tempfile
import gc

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

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

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

MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4

USE_FP16 = torch.cuda.is_available()
FAST_TRACK = os.getenv("FAST_TRACK", "false").lower() == "true"

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
    """Tokenise les commentaires avec une longueur limitée."""
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def pack_labels(batch):
    """Construit les labels multi-labels."""
    batch["labels"] = np.stack(
        [batch[col] for col in LABEL_COLS],
        axis=1,
    ).astype(np.float32)

    return batch


def prepare(ds):
    """Prépare le dataset pour Transformers."""
    ds = ds.map(
        tokenize,
        batched=True,
        batch_size=32,
        num_proc=1,
        remove_columns=[TEXT_COL],
    )

    ds = ds.map(
        pack_labels,
        batched=True,
        batch_size=32,
        num_proc=1,
    )

    ds = ds.remove_columns(LABEL_COLS)
    ds.set_format("torch")

    return ds


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):
    """Calcule les scores F1 micro et macro."""
    logits, labels = eval_pred

    # Protection contre les débordements numériques
    logits = np.clip(logits, -30, 30)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

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

    # Ne charge que les colonnes utiles, si elles existent.
    # clean_comment est facultative car elle peut être créée.
    header = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()

    required_cols = LABEL_COLS + [RAW_TEXT_COL]
    missing_cols = [col for col in required_cols if col not in header]

    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans le dataset : {missing_cols}"
        )

    usecols = required_cols.copy()
    if TEXT_COL in header:
        usecols.append(TEXT_COL)

    df = pd.read_csv(DATA_PATH, usecols=usecols)

    if TEXT_COL not in df.columns:
        df[TEXT_COL] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )
    else:
        df[TEXT_COL] = df[TEXT_COL].astype("string")

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

        # Sauvegarde des commentaires nettoyés
        df.to_csv(DATA_PATH, index=False)

        print(
            f"{int(mask.sum())} commentaires nettoyés et sauvegardés."
        )

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df = df[df[TEXT_COL].str.strip().ne("")].copy()

    for col in LABEL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=LABEL_COLS).copy()

    for col in LABEL_COLS:
        invalid = ~df[col].isin([0, 1])

        if invalid.any():
            raise ValueError(
                f"La colonne '{col}' contient des valeurs différentes "
                "de 0 et 1."
            )

        df[col] = df[col].astype("int64")

    if len(df) < 5:
        raise ValueError(
            f"Dataset insuffisant après nettoyage : {len(df)} exemples."
        )

    print(f"Nombre d'exemples disponibles : {len(df)}")

    return df


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_best_thresholds(logits, labels):
    """Recherche le meilleur seuil F1 pour chaque catégorie."""
    logits = np.clip(logits, -30, 30)
    probs = 1.0 / (1.0 + np.exp(-logits))

    thresholds = np.arange(0.1, 0.91, 0.01)
    best_thresholds = []

    for i in range(labels.shape[1]):
        scores = []

        for threshold in thresholds:
            preds = (probs[:, i] >= threshold).astype(int)

            scores.append(
                f1_score(
                    labels[:, i],
                    preds,
                    zero_division=0,
                )
            )

        best_thresholds.append(
            float(thresholds[np.argmax(scores)])
        )

    return best_thresholds


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Chargement des données
    # --------------------------------------------------------

    df = load_and_prepare_dataframe()

    if FAST_TRACK:
        print("MODE FAST_TRACK : maximum 50 exemples, 1 époque.")
        df = df.head(50).copy()

    if len(df) < 5:
        raise ValueError("Pas assez d'exemples pour effectuer le split.")

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

    # Libération des objets intermédiaires inutiles
    del dataset, split, train_val, df
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if USE_FP16 else None,
    )

    # --------------------------------------------------------
    # 3. Chargement du modèle
    # --------------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    )

    # Réduction de la mémoire des activations
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

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

            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,

            gradient_checkpointing=True,

            # Adafactor limite l'état mémoire de l'optimiseur
            optim="adafactor",

            num_train_epochs=1 if FAST_TRACK else 10,
            weight_decay=0.01,

            logging_strategy="steps",
            logging_steps=2 if FAST_TRACK else 10,

            fp16=USE_FP16,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,

            # Évite de charger les logits de tout le dataset
            # en mémoire pendant les évaluations.
            eval_accumulation_steps=1,

            report_to="none",
            save_total_limit=1,

            seed=42,
        )

        mlflow.log_params({
            "learning_rate": training_args.learning_rate,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "max_length": MAX_LENGTH,
            "epochs": training_args.num_train_epochs,
            "weight_decay": training_args.weight_decay,
            "optimizer": "adafactor",
            "model_name": MODEL_NAME,
            "fast_track": FAST_TRACK,
            "dataset_size": len(train_ds) + len(val_ds),
        })

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

        print("Début de l'entraînement...")
        trainer.train()

        # ----------------------------------------------------
        # 6. Recherche des meilleurs seuils
        # ----------------------------------------------------

        print("Recherche des meilleurs seuils...")

        predictions = trainer.predict(val_ds)

        best_thresholds = find_best_thresholds(
            predictions.predictions,
            predictions.label_ids,
        )

        del predictions
        gc.collect()

        print("Meilleurs seuils :", best_thresholds)

        for i, col in enumerate(LABEL_COLS):
            mlflow.log_metric(
                f"threshold_{i}",
                best_thresholds[i],
            )

        # ----------------------------------------------------
        # 7. Sauvegarde du modèle
        # ----------------------------------------------------

        for checkpoint_dir in glob.glob(
            os.path.join(tmp_dir, "checkpoint-*")
        ):
            shutil.rmtree(checkpoint_dir)

        model = trainer.model
        model.config.thresholds = best_thresholds
        model.config.label_names = LABEL_COLS
        model.config.use_cache = True

        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        mlflow.log_artifacts(
            tmp_dir,
            artifact_path="camembert_model",
        )

        print("Modèle sauvegardé dans MLflow.")

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