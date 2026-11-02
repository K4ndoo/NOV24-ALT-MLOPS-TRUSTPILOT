import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import MeanValue, RowCount, ValueDrift
from evidently.presets import DataSummaryPreset
from evidently.tests import lt

from src.monitoring import distributions
from src.monitoring.data_loading import LABEL_COLS
from src.monitoring.psi import THRESHOLD_MODERATE, THRESHOLD_STRONG, bin_lengths
from src.monitoring.window import is_exploitable

TEXT_COL = "text"
STAR_GROUP_COL = "notes_regroupees"
LENGTH_BIN_COL = "longueur_tranche"
PSI_COLUMNS = [STAR_GROUP_COL, LENGTH_BIN_COL, *LABEL_COLS]
SUMMARY_COLUMNS = ["star", "n_words", *LABEL_COLS]
NON_EXPLOITABLE = "non_exploitable"

# Dérive du texte : AUC du classifieur qui distingue la référence de la fenêtre (0,5 = indiscernables).
# 0,55 est le seuil par défaut d'Evidently ; 0,7 est la convention du projet pour une dérive forte.
TEXT_THRESHOLD_MODERATE = 0.55
TEXT_THRESHOLD_STRONG = 0.7

STARS = [1, 2, 3, 4, 5]


def star_group(star):
    """Notes regroupées pour le PSI : 2, 3 et 4 étoiles sont pas assez fréquents pour être des tranches séparées."""
    if pd.isna(star):
        return None
    return "1" if star == 1 else "5" if star == 5 else "2-4"


def note_col(star):
    return f"note_{star}"


def label_share_col(label):
    return f"part_{label}"


def company_col(company):
    return f"enseigne_{company}"


def share_columns(companies):
    return [*(note_col(s) for s in STARS), *(label_share_col(c) for c in LABEL_COLS), *(company_col(c) for c in companies)]


def to_frame(df, edges, companies):
    """Colonnes envoyées à Evidently.

    - texte nettoyé (dérive du texte) ;
    - notes regroupées, longueur en déciles de la référence et labels 0/1 (PSI) ;
    - colonnes indicatrices 0/1 dont la moyenne est une part (notes, catégories, enseignes).
    Les avis sans texte n'ont ni longueur ni tranche : ils sont exclus de ces deux variables.
    """
    frame = pd.DataFrame(index=df.index)
    frame[TEXT_COL] = df["text"]
    frame["star"] = df["star"].astype(str)
    frame[STAR_GROUP_COL] = df["star"].map(star_group)
    for col in LABEL_COLS:
        frame[col] = df[col].astype(int)
    has_text = df["n_words"] > 0
    frame["n_words"] = df["n_words"].where(has_text)
    frame[LENGTH_BIN_COL] = bin_lengths(df["n_words"], edges).where(has_text)
    for star in STARS:
        frame[note_col(star)] = (df["star"] == star).astype(int)
    for col in LABEL_COLS:
        frame[label_share_col(col)] = df[col].astype(int)
    for company in companies:
        frame[company_col(company)] = (df["company"] == company).astype(int)
    return frame


def _definition(companies):
    return DataDefinition(
        text_columns=[TEXT_COL],
        categorical_columns=["star", *PSI_COLUMNS],
        numerical_columns=["n_words", *share_columns(companies)],
    )


def drift_metrics():
    """Dérive du texte et PSI des variables d'explication, chacun avec deux tests : échec = forte, alerte = modérée."""
    text = ValueDrift(
        column=TEXT_COL,
        method="abs_text_content_drift",
        threshold=TEXT_THRESHOLD_MODERATE,
        tests=[lt(TEXT_THRESHOLD_STRONG), lt(TEXT_THRESHOLD_MODERATE, is_critical=False)],
    )
    psi = [
        ValueDrift(
            column=col,
            method="psi",
            threshold=THRESHOLD_MODERATE,
            tests=[lt(THRESHOLD_STRONG), lt(THRESHOLD_MODERATE, is_critical=False)],
        )
        for col in PSI_COLUMNS
    ]
    return [text, *psi]


def raw_metrics(companies):
    """Données brutes du jour : parts (moyennes des colonnes indicatrices) et longueur moyenne."""
    return [MeanValue(column=col) for col in ["n_words", *share_columns(companies)]]


def comparison_report(reference, current, edges, companies, timestamp, name, metadata, tags):
    """Rapport de comparaison : volume, données brutes, distributions côte à côte, et dérive si assez d'avis."""
    exploitable = is_exploitable(len(current))
    metrics = [RowCount()]

    if len(current):
        metrics += [DataSummaryPreset(columns=SUMMARY_COLUMNS), *raw_metrics(companies)]
    if exploitable:
        metrics += drift_metrics()
    else:
        tags = [*tags, NON_EXPLOITABLE]
    if len(current):
        metadata = {
            **metadata,
            "mots_propres_au_recent": distributions.as_text(distributions.exclusive_words(current, reference)),
            "mots_propres_a_la_reference": distributions.as_text(distributions.exclusive_words(reference, current)),
        }
    definition = _definition(companies)
    return Report(metrics).run(
        current_data=Dataset.from_pandas(to_frame(current, edges, companies), data_definition=definition),
        reference_data=Dataset.from_pandas(to_frame(reference, edges, companies), data_definition=definition),
        timestamp=timestamp,
        name=name,
        metadata=metadata,
        tags=tags,
    )


def daily_drift_report(reference, window, edges, companies, day, start, promotion, excluded):
    """Dérive du jour : fenêtre de 7 jours d'avis d'inférence récents contre la référence figée."""
    return comparison_report(
        reference,
        window,
        edges,
        companies,
        timestamp=day.to_pydatetime(),
        name=f"Dérive {day:%Y-%m-%d}",
        metadata={
            "type": "derive",
            "fenetre": f"{start:%Y-%m-%d} au {day:%Y-%m-%d}",
            "avis_fenetre": str(len(window)),
            "avis_anciens_exclus": str(excluded),
            "reference": f"promotion {promotion.short} du {promotion.date}",
            "avis_reference": str(len(reference)),
        },
        tags=["derive"],
    )


def weekly_report(previous_version, added, edges, companies, promotion, previous_promotion):
    """Réentraînement : avis ajoutés par une promotion contre la version précédente du jeu d'entraînement."""
    return comparison_report(
        previous_version,
        added,
        edges,
        companies,
        timestamp=promotion.timestamp,
        name=f"Promotion {promotion.short} ({promotion.date})",
        metadata={
            "type": "reentrainement",
            "promotion": f"{promotion.short} du {promotion.date}",
            "comparee_a": f"{previous_promotion.short} du {previous_promotion.date}",
            "avis_ajoutes": str(len(added)),
            "avis_version_precedente": str(len(previous_version)),
        },
        tags=["reentrainement"],
    )
