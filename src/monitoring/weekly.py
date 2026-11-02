"""Réentraînement hebdomadaire : ce que chaque promotion du modèle a ajouté au jeu d'entraînement."""


def added_reviews(previous, current):
    """Avis présents dans `current` dont l'identifiant n'existait pas dans `previous`."""
    return current[~current["comment_id"].isin(previous["comment_id"])]
