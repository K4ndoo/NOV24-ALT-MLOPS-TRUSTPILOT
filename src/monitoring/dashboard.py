"""Projets et panneaux du tableau de bord Evidently UI."""

from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import bar_plot_panel, line_plot_panel, text_panel

from src.monitoring.data_loading import LABEL_COLS
from src.monitoring.reports import (
    LENGTH_BIN_COL,
    STAR_GROUP_COL,
    STARS,
    TEXT_COL,
    TEXT_THRESHOLD_MODERATE,
    TEXT_THRESHOLD_STRONG,
    company_col,
    label_share_col,
    note_col,
)

DRIFT_PROJECT = "Dérive des données"
WEEKLY_PROJECT = "Réentraînement hebdomadaire"

THRESHOLDS_TEXT = (
    f"Seuils · texte (AUC) : {TEXT_THRESHOLD_MODERATE} / {TEXT_THRESHOLD_STRONG} · PSI : 0.1 / 0.25 · "
    "verdict du jour : Reports > Tests"
)


def _value(metric, column, legend):
    return PanelMetric(legend=legend, metric=f"evidently:metric_v2:{metric}", metric_labels={"column": column})


def _pct(share):
    return f"{share * 100:.0f} %".replace(".", ",")


def reference_summary(reference):
    """Valeurs de la référence, rappelées sous les panneaux de données brutes."""
    stars = reference["star"].value_counts(normalize=True)
    has_text = reference["n_words"] > 0
    return {
        "notes": " ; ".join(f"{s}★ {_pct(stars.get(s, 0))}" for s in STARS),
        "categories": " ; ".join(f"{c} {_pct(reference[c].mean())}" for c in LABEL_COLS),
        "longueur": f"{reference.loc[has_text, 'n_words'].mean():.0f} mots",
        "enseignes": " ; ".join(f"{c} {_pct(s)}" for c, s in reference["company"].value_counts(normalize=True).items()),
    }


def _drift_panels(reference, companies):
    ref = reference_summary(reference)
    return [
        text_panel(
            title="Avis récents (7 jours) vs données d'entraînement",
            description=THRESHOLDS_TEXT,
        ),
        line_plot_panel(
            title="Dérive du texte des avis",
            values=[_value("ValueDrift", TEXT_COL, "texte (AUC)")],
            size="full",
        ),
        line_plot_panel(
            title="PSI : notes (1 / 2-4 / 5) et longueur",
            values=[_value("ValueDrift", STAR_GROUP_COL, "notes"), _value("ValueDrift", LENGTH_BIN_COL, "longueur")],
            size="half",
        ),
        line_plot_panel(
            title="PSI : catégories",
            values=[_value("ValueDrift", col, col) for col in LABEL_COLS],
            size="half",
        ),
        bar_plot_panel(
            title="Répartition des notes",
            description=f"Réf. : {ref['notes']}",
            values=[_value("MeanValue", note_col(s), f"{s}★") for s in STARS],
            size="half",
            stacked=True,
        ),
        bar_plot_panel(
            title="Répartition par enseigne",
            description=f"Réf. : {ref['enseignes']}",
            values=[_value("MeanValue", company_col(c), c) for c in companies],
            size="half",
            stacked=True,
        ),
        line_plot_panel(
            title="Part de chaque catégorie",
            description=f"Réf. : {ref['categories']}",
            values=[_value("MeanValue", label_share_col(c), c) for c in LABEL_COLS],
            size="half",
        ),
        line_plot_panel(
            title="Longueur moyenne des avis (mots)",
            description=f"Réf. : {ref['longueur']}",
            values=[_value("MeanValue", "n_words", "longueur moyenne")],
            size="half",
        ),
        line_plot_panel(
            title="Avis dans la fenêtre (minimum 50)",
            values=[PanelMetric(legend="avis", metric="evidently:metric_v2:RowCount")],
            size="full",
        ),
    ]


def _weekly_panels(reference, companies):
    return [
        text_panel(
            title="Avis ajoutés à chaque réentraînement vs version précédente",
            description=THRESHOLDS_TEXT,
        ),
        line_plot_panel(
            title="Dérive du texte des avis ajoutés",
            values=[_value("ValueDrift", TEXT_COL, "texte (AUC)")],
            size="half",
        ),
        line_plot_panel(
            title="PSI des avis ajoutés : notes et longueur",
            values=[_value("ValueDrift", STAR_GROUP_COL, "notes"), _value("ValueDrift", LENGTH_BIN_COL, "longueur")],
            size="half",
        ),
        line_plot_panel(
            title="Avis ajoutés par promotion",
            values=[PanelMetric(legend="avis ajoutés", metric="evidently:metric_v2:RowCount")],
            size="full",
        ),
    ]


PROJECTS = {
    DRIFT_PROJECT: (
        "Dérive des avis scrapés chaque jour par rapport au jeu d'entraînement du modèle en production.",
        _drift_panels,
    ),
    WEEKLY_PROJECT: (
        "Ce que chaque promotion du modèle a ajouté au jeu d'entraînement.",
        _weekly_panels,
    ),
}


def get_or_create_project(workspace, name, reference, companies):
    """Projet existant du même nom, ou nouveau projet avec ses panneaux (valeurs de la référence incluses)."""
    existing = workspace.search_project(name)
    if existing:
        return existing[0]
    description, panels = PROJECTS[name]
    project = workspace.create_project(name, description=description)
    for panel in panels(reference, companies):
        project.dashboard.add_panel(panel)
    return project
