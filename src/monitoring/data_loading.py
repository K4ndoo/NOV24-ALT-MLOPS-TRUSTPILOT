import pandas as pd

from src.monitoring.text import clean_text

INFERENCE_PATH = "data/daily_inference/dataset_inference.csv"
LABEL_COLS = ["qualité produit", "service livraison", "service client"]


def prepare(raw):
    """Normalise un jeu d'avis brut : texte nettoyé recalculé, note, labels, dates, enseigne."""
    df = pd.DataFrame(index=raw.index)
    df["comment_id"] = raw["comment_id"].astype(str)
    df["text"] = raw["Commentaire"].apply(clean_text)
    df["n_words"] = df["text"].str.split().str.len().fillna(0).astype(int)
    df["star"] = pd.to_numeric(raw["star"], errors="coerce").astype("Int64")
    for col in LABEL_COLS:
        df[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype(int)
    if "scraping_date" in raw.columns:
        df["scraping_date"] = pd.to_datetime(raw["scraping_date"], errors="coerce").dt.normalize()
    else:
        df["scraping_date"] = pd.NaT
    # Deux formats coexistent dans l'entraînement ('2020-06-25' et '2026-06-17T21:47:35.000Z') : format="mixed".
    df["published"] = (
        pd.to_datetime(raw["date"], utc=True, errors="coerce", format="mixed").dt.tz_localize(None).dt.normalize()
    )
    df["company"] = raw["company"]
    return df


def load_reviews(path):
    return prepare(pd.read_csv(path))


def load_inference(path=INFERENCE_PATH):
    """Avis scrapés pour l'inférence quotidienne : tout le fichier d'inférence."""
    return load_reviews(path)
