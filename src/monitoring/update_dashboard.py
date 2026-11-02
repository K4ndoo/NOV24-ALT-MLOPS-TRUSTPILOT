"""Met le tableau de bord Evidently UI à jour jusqu'à une date (la veille par défaut).

Usage : python -m src.monitoring.update_dashboard [--until AAAA-MM-JJ] [--rev REV] [--inference CSV] [--rebuild]

Ajoute un rapport de dérive par jour de scraping manquant, et un rapport
par promotion du modèle manquante. Les rapports déjà présents ne sont pas recalculés (pas de doublon).
L'espace Evidently est entièrement régénérable : --rebuild le recrée à partir des données versionnées.
Lancé chaque jour par le DAG Airflow dags/monitoring_drift.py.
"""

import argparse
from datetime import date, timedelta

import pandas as pd
from evidently.ui.workspace import Workspace

from src.monitoring.dashboard import DRIFT_PROJECT, WEEKLY_PROJECT, get_or_create_project
from src.monitoring.data_loading import INFERENCE_PATH, load_inference, load_reviews
from src.monitoring.psi import length_bin_edges
from src.monitoring.reference import fetch_reference, list_promotions
from src.monitoring.reports import daily_drift_report, weekly_report
from src.monitoring.weekly import added_reviews
from src.monitoring.window import recent_only, select_window

WORKSPACE_DIR = "evidently_workspace"


def _edges(reference):
    return length_bin_edges(reference.loc[reference["n_words"] > 0, "n_words"])


def existing_names(workspace, project):
    return {workspace.get_run(project.id, run_id).name for run_id in workspace.list_runs(project.id)}


def update_daily(workspace, until, rev="HEAD", inference_path=INFERENCE_PATH):
    """Rapports quotidiens manquants, du premier jour de scraping jusqu'à `until`."""
    promotion = list_promotions(rev)[0]  # dernière promotion : le modèle en production
    reference = load_reviews(fetch_reference(promotion))
    inference = load_inference(inference_path)
    recent = recent_only(inference)
    companies = sorted(inference["company"].dropna().unique())
    project = get_or_create_project(workspace, DRIFT_PROJECT, reference, companies)
    done = existing_names(workspace, project)
    edges = _edges(reference)
    added = 0
    for day in pd.date_range(inference["scraping_date"].min(), until, freq="D"):
        window, start, _ = select_window(recent, day)
        if f"Dérive {day:%Y-%m-%d}" not in done:
            excluded = len(select_window(inference, day)[0]) - len(window)
            snapshot = daily_drift_report(reference, window, edges, companies, day, start, promotion, excluded)
            workspace.add_run(project.id, snapshot)
            added += 1
    print(
        f"{DRIFT_PROJECT} : {added} rapport(s) ajouté(s), référence {promotion.short} du {promotion.date}, "
        f"{len(inference) - len(recent)} avis anciens exclus"
    )


def update_weekly(workspace, rev="HEAD"):
    """Un rapport par promotion manquante, comparée à la promotion d'avant."""
    promotions = list_promotions(rev)
    versions = {}

    def version(promotion):
        if promotion.commit not in versions:
            versions[promotion.commit] = load_reviews(fetch_reference(promotion))
        return versions[promotion.commit]

    latest = version(promotions[0])
    companies = sorted(latest["company"].dropna().unique())
    project = get_or_create_project(workspace, WEEKLY_PROJECT, latest, companies)
    done = existing_names(workspace, project)
    added = 0
    for current, previous in zip(promotions, promotions[1:]):
        if f"Promotion {current.short} ({current.date})" in done:
            continue
        previous_version = version(previous)
        new = added_reviews(previous_version, version(current))
        snapshot = weekly_report(previous_version, new, _edges(previous_version), companies, current, previous)
        workspace.add_run(project.id, snapshot)
        added += 1
    print(f"{WEEKLY_PROJECT} : {added} rapport(s) ajouté(s), {len(promotions)} promotions")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # La veille par défaut : un rapport n'est jamais recalculé, celui du jour serait figé
    # avant que l'inférence quotidienne n'ait ajouté les avis du jour.
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    parser.add_argument("--until", default=yesterday, help="dernier jour à couvrir (AAAA-MM-JJ)")
    parser.add_argument("--rev", default="HEAD", help="révision git où chercher les promotions (ex. origin/main)")
    parser.add_argument("--inference", default=INFERENCE_PATH, help="CSV des avis d'inférence")
    parser.add_argument("--rebuild", action="store_true", help="recréer les projets à partir de zéro")
    parser.add_argument("--workspace", default=WORKSPACE_DIR)
    args = parser.parse_args()

    workspace = Workspace.create(args.workspace)
    if args.rebuild:
        for name in (DRIFT_PROJECT, WEEKLY_PROJECT):
            for project in workspace.search_project(name):
                workspace.delete_project(project.id)
    update_daily(workspace, pd.Timestamp(args.until).normalize(), args.rev, args.inference)
    update_weekly(workspace, args.rev)


if __name__ == "__main__":
    main()
