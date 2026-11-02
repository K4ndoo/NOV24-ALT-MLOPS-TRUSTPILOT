import re
import pandas as pd

# Copie de clean_text de src/modelisation/daily_inference.py.
# On ne l'importe pas, ce script tourne en mode script dans la CI, sans `src` sur le chemin.
# tests/monitoring/test_clean_text.py vérifie que les deux versions restent identiques.
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"(http|www)\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"(@|#)\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
