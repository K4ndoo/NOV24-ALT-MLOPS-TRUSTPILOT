import numpy as np
import pandas as pd

from src.monitoring.data_loading import LABEL_COLS
from src.monitoring.distributions import exclusive_words
from src.monitoring.psi import bin_lengths, length_bin_edges
from src.monitoring.reports import (
    NON_EXPLOITABLE,
    PSI_COLUMNS,
    STAR_GROUP_COL,
    TEXT_COL,
    comparison_report,
    note_col,
    star_group,
)
from src.monitoring.weekly import added_reviews
from src.monitoring.window import recent_only, select_window

TEXTS = ["livraison rapide produit conforme", "colis jamais reçu", "service client absent", "qualité correcte prix bas"]
COMPANIES = ["BlackMarket", "Zalando"]


def _reviews(n, star_shares, seed, start="2026-10-01", days=1, texts=TEXTS):
    """Jeu d'avis synthétique au format de data_loading.prepare."""
    rng = np.random.default_rng(seed)
    scraped = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    df = pd.DataFrame(
        {
            "comment_id": [f"{seed}-{i}" for i in range(n)],
            "text": rng.choice(texts, n),
            "n_words": rng.integers(1, 120, n),
            "star": pd.array(rng.choice([1, 2, 3, 4, 5], n, p=star_shares), dtype="Int64"),
            "scraping_date": scraped,
            "published": scraped,
            "company": rng.choice(COMPANIES, n),
        }
    )
    for col in LABEL_COLS:
        df[col] = rng.integers(0, 2, n)
    return df


UNIFORM = [0.2, 0.2, 0.2, 0.2, 0.2]


def _report(reference, current):
    edges = length_bin_edges(reference["n_words"])
    timestamp = pd.Timestamp("2026-10-25").to_pydatetime()
    return comparison_report(reference, current, edges, COMPANIES, timestamp, "test", {}, [])


def _test_statuses(snapshot, column):
    """Statuts des deux tests de dérive d'une variable : (seuil fort, seuil modéré)."""
    tests = [t for t in snapshot.dict()["tests"] if f"Value drift for {column}:" in t["name"]]
    statuses = sorted(((t["name"], str(t["status"])) for t in tests), key=lambda t: t[0])
    strong = [s for n, s in statuses if "0.250" in n or "0.700" in n]
    moderate = [s for n, s in statuses if "0.100" in n or "0.550" in n]
    return strong[0], moderate[0]


def _metric_values(snapshot, prefix):
    return {m["metric_name"]: m["value"] for m in snapshot.dict()["metrics"] if m["metric_name"].startswith(prefix)}


def test_fenetre_de_7_jours_bornes_incluses():
    dates = pd.to_datetime(["2026-10-18", "2026-10-19", "2026-10-25", "2026-10-26", None])
    window, start, end = select_window(pd.DataFrame({"scraping_date": dates}), "2026-10-25")
    assert (start, end) == (pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-25"))
    assert list(window["scraping_date"]) == [pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-25")]


def test_avis_anciens_exclus_des_fenetres():
    df = pd.DataFrame(
        {
            "scraping_date": pd.to_datetime(["2026-09-18"] * 4),
            "published": pd.to_datetime(["2026-09-18", "2026-09-11", "2026-09-10", None]),
        }
    )
    # publié le jour même et 7 jours avant : gardés ; 8 jours avant : exclu ; date inconnue : gardé
    assert list(recent_only(df).index) == [0, 1, 3]


def test_notes_regroupees():
    assert [star_group(s) for s in [1, 2, 3, 4, 5]] == ["1★", "2-4★", "2-4★", "2-4★", "5★"]


def test_tranches_de_longueur_calculees_sur_la_reference():
    labels = bin_lengths(pd.Series([1, 50, 100, 500]), length_bin_edges(pd.Series(range(1, 101))))
    assert labels.iloc[0].startswith("0-")
    assert labels.iloc[-1].endswith("+ mots")
    assert labels.nunique() == 3


def test_verdict_pas_de_derive_sur_deux_jeux_identiques():
    reference = _reviews(800, UNIFORM, seed=1)
    snapshot = _report(reference, reference.copy())
    assert _test_statuses(snapshot, STAR_GROUP_COL) == ("TestStatus.SUCCESS", "TestStatus.SUCCESS")


def test_verdict_derive_forte_quand_les_notes_1_etoile_explosent():
    reference = _reviews(800, UNIFORM, seed=2)
    current = _reviews(300, [0.8, 0.05, 0.05, 0.05, 0.05], seed=3)
    assert _test_statuses(_report(reference, current), STAR_GROUP_COL) == ("TestStatus.FAIL", "TestStatus.WARNING")


def test_derive_du_texte_detectee_quand_le_vocabulaire_change():
    reference = _reviews(800, UNIFORM, seed=4)
    current = _reviews(200, UNIFORM, seed=5, texts=["écran rayé batterie morte", "téléphone reconditionné parfait"])
    assert _test_statuses(_report(reference, current), TEXT_COL) == ("TestStatus.FAIL", "TestStatus.WARNING")


def test_fenetre_sous_50_avis_sans_derive_et_marquee_non_exploitable():
    snapshot = _report(_reviews(800, UNIFORM, seed=6), _reviews(20, UNIFORM, seed=7))
    assert not _metric_values(snapshot, "ValueDrift")
    assert NON_EXPLOITABLE in snapshot.to_snapshot_model().tags


def test_fenetre_vide_sans_erreur():
    reference = _reviews(800, UNIFORM, seed=8)
    snapshot = _report(reference, reference.iloc[0:0])
    assert NON_EXPLOITABLE in snapshot.to_snapshot_model().tags


def test_derive_du_texte_et_un_psi_par_variable_quand_la_fenetre_est_exploitable():
    snapshot = _report(_reviews(800, UNIFORM, seed=9), _reviews(200, UNIFORM, seed=10))
    assert len(_metric_values(snapshot, "ValueDrift")) == len(PSI_COLUMNS) + 1


def test_parts_du_jour_calculees_sur_la_fenetre():
    current = _reviews(200, UNIFORM, seed=11)
    values = _metric_values(_report(_reviews(800, UNIFORM, seed=12), current), "MeanValue")
    share_5 = next(v for name, v in values.items() if f"column={note_col(5)}" in name)
    assert share_5 == (current["star"] == 5).mean()


def test_avis_ajoutes_entre_deux_promotions():
    previous = _reviews(100, UNIFORM, seed=13)
    new = _reviews(30, UNIFORM, seed=14)
    assert len(added_reviews(previous, pd.concat([previous, new]))) == 30


def test_mots_propres_a_un_seul_jeu():
    a = pd.DataFrame({"text": ["veepee livraison"] * 5})
    b = pd.DataFrame({"text": ["backmarket livraison"] * 5})
    assert exclusive_words(a, b) == [("veepee", 5)]
