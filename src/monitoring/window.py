import pandas as pd

WINDOW_DAYS = 7
MIN_REVIEWS = 50
# Un avis publié plus de 7 jours avant son scraping vient d'un rattrapage (premier scraping d'une
# enseigne : 751 avis le 2026-09-18, dont certains de 2025) : il ne fait pas partie du flux du jour.
MAX_PUBLICATION_LAG_DAYS = 7


def recent_only(df):
    """Avis publiés au plus 7 jours avant leur scraping (date de publication inconnue : gardé)."""
    lag = (df["scraping_date"] - df["published"]).dt.days
    return df[lag.isna() | (lag <= MAX_PUBLICATION_LAG_DAYS)]


def select_window(df, report_date, days=WINDOW_DAYS):
    """Avis dont la date de scraping tombe dans les `days` derniers jours, date de rapport incluse.

    Renvoie (avis de la fenêtre, premier jour, dernier jour).
    """
    end = pd.Timestamp(report_date).normalize()
    start = end - pd.Timedelta(days=days - 1)
    in_window = df["scraping_date"].between(start, end)
    return df[in_window], start, end


def is_exploitable(n_recent):
    """Une fenêtre de moins de 50 avis est non exploitable, quelle que soit la valeur du PSI."""
    return n_recent >= MIN_REVIEWS
