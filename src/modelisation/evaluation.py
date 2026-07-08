import numpy as np
import pandas as pd
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


# CONFIG

#MODEL_DIR = "./../../models/camembert"
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models" / "camembert"
TEXT_COL = "clean_comment"
BATCH_SIZE = 32


# LOAD MODEL + THRESHOLDS

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

THRESHOLDS = model.config.thresholds
LABEL_COLS = model.config.label_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")


# UTILS

def tokenize(batch):
    return tokenizer(batch[TEXT_COL], truncation=True, max_length=256)

def pack_labels(batch):
    batch["labels"] = np.stack(
        [batch[col] for col in LABEL_COLS], axis=1
    ).astype(np.float32)
    return batch

def prepare(df):
    ds = Dataset.from_pandas(df[[TEXT_COL] + LABEL_COLS])
    ds = ds.map(tokenize,    batched=True, num_proc=4)
    ds = ds.map(pack_labels, batched=True, num_proc=4)
    ds = ds.remove_columns(LABEL_COLS + [TEXT_COL])
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

    thresholds = np.array(THRESHOLDS)
    preds = (all_probs > thresholds).astype(int)

    return all_labels, preds


def evaluate(name, df):
    print(f"\n===== {name} =====")

    ds = prepare(df)
    labels, preds = predict(ds)

    print(classification_report(labels, preds, target_names=LABEL_COLS, zero_division=0))

    for i, label in enumerate(LABEL_COLS):
        cm = confusion_matrix(labels[:, i], preds[:, i])
        print(f"\nMatrice de confusion — {label}")
        print(cm)


# LOAD DATASETS

#df_main = pd.read_csv('./../../data/labelled_dataset/dataset_avis.csv')
#df_gold = pd.read_csv('./../../data/test_dataset/100_avis_annote.csv', sep=";")
df_main = pd.read_csv(
    BASE_DIR / "data" / "labelled_dataset" / "dataset_avis.csv"
)

df_gold = pd.read_csv(
    BASE_DIR / "data" / "test_dataset" / "100_avis_annote.csv",
    sep=";"
)

from sklearn.model_selection import train_test_split
_, df_test = train_test_split(df_main, test_size=0.2, random_state=42)


# EVALUATION

evaluate("TEST SPLIT",   df_test)
evaluate("GOLD DATASET", df_gold)
