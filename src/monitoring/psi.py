import numpy as np
import pandas as pd

# Seuils du PSI : < 0,1 pas de dérive ; 0,1 à 0,25 dérive modérée ; > 0,25 dérive forte.
# Ils deviennent deux tests Evidently par variable (voir reports.py).
THRESHOLD_MODERATE = 0.1
THRESHOLD_STRONG = 0.25
N_LENGTH_BINS = 10


def length_bin_edges(reference_lengths, n_bins=N_LENGTH_BINS):
    """Calcul des bornes de longueur (pour créer des tranches de volume égal sur la référence).

    Evidently regroupant les valeurs numériques sur tous les datasets à la fois, on contourne le problème : on pré-découpe nos propres groupes de longueur sur la référence, qu'on lui passe ensuite en tant que variables catégorielles.
    """
    inner = np.unique(np.floor(np.quantile(reference_lengths, np.linspace(0, 1, n_bins + 1)[1:-1])))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def bin_lengths(lengths, edges):
    """Tranche de longueur de chaque avis, sous forme d'étiquette lisible ('13-18 mots')."""
    labels = []
    for low, high in zip(edges[:-1], edges[1:]):
        first = 0 if np.isinf(low) else int(low) + 1
        labels.append(f"{first}+ mots" if np.isinf(high) else f"{first}-{int(high)} mots")
    return pd.cut(lengths, bins=edges, labels=labels).astype(str)
