import numpy as np
import pandas as pd
import torch

from pathlib import Path
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


# CONFIG

MODEL_NAME = "camembert-base"
TEXT_COL = "clean_comment"
LABEL_COLS = ["qualité produit", "service livraison", "service client"]

#OUTPUT_DIR = "./../../models/camembert"
BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = BASE_DIR / "models" / "camembert"
USE_FP16 = torch.cuda.is_available() # demi-précision automatique si GPU dispo


# LOAD DATA

#df = pd.read_csv('./../../data/labelled_dataset/dataset_avis.csv')
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "labelled_dataset" / "dataset_avis.csv"

df = pd.read_csv(DATA_PATH)
dataset = Dataset.from_pandas(df[[TEXT_COL] + LABEL_COLS])

dataset = dataset.train_test_split(test_size=0.2,  seed=42)
test_ds = dataset["test"]
dataset = dataset["train"].train_test_split(test_size=0.15, seed=42)
train_ds = dataset["train"]
val_ds = dataset["test"]


# TOKENISATION

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

train_ds = prepare(train_ds)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

data_collator = DataCollatorWithPadding(tokenizer)


# MODEL

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_COLS),
    problem_type="multi_label_classification",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


# TRAINING

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Évaluation & sauvegarde
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",

    # Hyperparamètres
    learning_rate=2e-5,
    per_device_train_batch_size=16, # à ajuster selon la RAM disponible
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,

    # Vitesse
    fp16=USE_FP16, 
    dataloader_num_workers=4, # mettre à 0 si sur Windows ou si problèmes de multiprocessing
    # Logging
    logging_strategy="epoch",
    report_to="none",
)

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


# ÉVALUATION FINALE SUR LE JEU DE TEST

test_predictions = trainer.predict(test_ds)
test_logits = test_predictions.predictions
test_labels = test_predictions.label_ids
test_probs = 1 / (1 + np.exp(-test_logits))

test_preds = np.stack(
    [(test_probs[:, i] > best_thresholds[i]).astype(int) for i in range(len(LABEL_COLS))],
    axis=1,
)

print("Test F1 micro:", f1_score(test_labels, test_preds, average="micro",  zero_division=0))
print("Test F1 macro:", f1_score(test_labels, test_preds, average="macro",  zero_division=0))
for i, col in enumerate(LABEL_COLS):
    print(f"  {col}: F1 = {f1_score(test_labels[:, i], test_preds[:, i], zero_division=0):.3f}")


# SAVE MODEL + THRESHOLDS

model.config.thresholds = best_thresholds
model.config.label_names = LABEL_COLS

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModèle sauvegardé dans : {OUTPUT_DIR}")
