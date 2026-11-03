import os
import sys
import time
import subprocess
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[2]

# API FastAPI (API_URL = adresse vue par Streamlit, PUBLIC_API_URL = adresse vue par le navigateur)
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:8000").rstrip("/")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://localhost:8001")

HEADERS = {"X-API-Key": API_KEY}

# Datasets utilisés pour tirer un avis au hasard
DATASETS = {
    "Entraînement": BASE_DIR / "data/labelled_dataset/dataset_avis.csv",
    "Inférence quotidienne": BASE_DIR / "data/daily_inference/dataset_inference.csv",
}

# Sources du simulateur (valeurs de --source de automatic_prediction.py)
SIMULATION_SOURCES = {
    "Entraînement + inférence": "all",
    "Entraînement": "train",
    "Inférence quotidienne": "inference",
}

LABEL_COLS = ["qualité produit", "service livraison", "service client"]

# Logs du simulateur (dossier ignoré par git)
LOG_FILE = BASE_DIR / "logs/traffic_simulation.log"

DEFAULT_REVIEW = "Le produit est de bonne qualité mais la livraison a pris trois semaines et le service client ne répond pas."


st.set_page_config(
    page_title="Démo API - Projet Trustpilot",
    layout="centered",
)


# ============================================================
# 2. APPELS API
# ============================================================

def get_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def get_model_info():
    try:
        response = requests.get(f"{API_URL}/model/info", headers=HEADERS, timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def predict(text):
    start = time.time()
    response = requests.post(
        f"{API_URL}/predict",
        json={"texts": [text]},
        headers=HEADERS,
        timeout=60,
    )
    response.raise_for_status()
    return response.json(), time.time() - start


# ============================================================
# 3. DATASETS
# ============================================================

@st.cache_data(show_spinner="Chargement du dataset...")
def load_reviews(dataset_name):
    path = DATASETS[dataset_name]
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df = df[df["Commentaire"].fillna("").astype(str).str.strip() != ""]
    return df.reset_index(drop=True)


def draw_random_review():
    df = load_reviews(st.session_state.random_source)

    if df is None or df.empty:
        st.session_state.random_error = f"Dataset introuvable : {DATASETS[st.session_state.random_source]}"
        return

    row = df.sample(1).iloc[0]
    st.session_state.review_text = str(row["Commentaire"]).strip()
    st.session_state.review_meta = row.to_dict() | {"dataset": st.session_state.random_source}
    st.session_state.random_error = None


def clear_review_meta():
    # L'avis a été modifié à la main : ce n'est plus celui du dataset
    st.session_state.review_meta = None


# ============================================================
# 4. SIMULATEUR DE TRAFIC
# ============================================================

@st.cache_resource
def simulation_state():
    # Partagé entre toutes les sessions Streamlit : un seul simulateur à la fois
    return {"process": None, "params": None, "log": None}


def simulation_running():
    process = simulation_state()["process"]
    return process is not None and process.poll() is None


def start_simulation(interval, batch_size, source):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w", encoding="utf-8")

    process = subprocess.Popen(
        [
            sys.executable, "-m", "src.modelisation.automatic_prediction",
            "--simulate",
            "--interval", str(interval),
            "--batch-size", str(batch_size),
            "--source", source,
        ],
        cwd=BASE_DIR,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=os.environ | {"PYTHONUNBUFFERED": "1", "API_URL": API_URL},
    )

    simulation_state().update(
        process=process,
        log=log,
        params={"interval": interval, "batch_size": batch_size, "source": source},
    )


def stop_simulation():
    process = simulation_state()["process"]
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    log = simulation_state()["log"]
    if log is not None:
        log.close()


def on_toggle_simulation():
    if st.session_state.simulation_on:
        start_simulation(
            st.session_state.sim_interval,
            st.session_state.sim_batch_size,
            SIMULATION_SOURCES[st.session_state.sim_source],
        )
    else:
        stop_simulation()


def read_simulation_logs():
    if not LOG_FILE.exists():
        return []
    return LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()


# ============================================================
# 5. AFFICHAGE DES RESULTATS
# ============================================================

def probability_bar(probability, threshold, detected):
    # Barre de probabilité avec un repère vertical à l'emplacement du seuil
    color = "#21a366" if detected else "#9aa0a6"
    return f"""
    <div style="position:relative;height:14px;border-radius:7px;
                background:rgba(128,128,128,0.2);margin:4px 0 2px 0;">
        <div style="width:{probability * 100:.1f}%;height:100%;border-radius:7px;background:{color};"></div>
        <div title="Seuil" style="position:absolute;top:-4px;left:{threshold * 100:.1f}%;
                    width:2px;height:22px;background:#e8453c;"></div>
    </div>
    """


def show_prediction(result, latency):
    prediction = result["predictions"][0]

    st.markdown("#### Résultat de la prédiction")

    cols = st.columns(len(prediction))
    for col, (label, values) in zip(cols, prediction.items()):
        detected = values["prediction"] == 1
        with col:
            st.markdown(f"**{label.capitalize()}**")
            st.markdown(f"## {values['probability']:.1%}")
            st.markdown(
                probability_bar(values["probability"], values["threshold"], detected),
                unsafe_allow_html=True,
            )
            st.caption(f"Seuil : {values['threshold']:.0%}")
            if detected:
                st.success("OUI", icon="✅")
            else:
                st.error("NON", icon="✖️")

    detected_labels = [label for label, values in prediction.items() if values["prediction"] == 1]
    st.caption(
        f"Catégories prédites : **{', '.join(detected_labels) or 'aucune'}** · "
        f"Temps de réponse de l'API : {latency * 1000:.0f} ms · "
        "Le trait rouge indique le seuil de décision de chaque catégorie."
    )

    with st.expander("Réponse JSON brute de l'API"):
        st.json(result)


# ============================================================
# 6. PAGE
# ============================================================

st.session_state.setdefault("review_text", DEFAULT_REVIEW)
st.session_state.setdefault("review_meta", None)
st.session_state.setdefault("random_error", None)
st.session_state.setdefault("last_prediction", None)
st.session_state.setdefault("sim_interval", 10)
st.session_state.setdefault("sim_batch_size", 1)

# --- Barre latérale : état de l'API et liens utiles ---
with st.sidebar:
    st.header("État de l'API")
    health = get_health()

    if health is None:
        st.error(f"API injoignable ({API_URL})")
    elif not health.get("model_loaded"):
        st.warning("API en ligne, modèle non chargé")
    else:
        st.success("API en ligne, modèle chargé")
        info = get_model_info()
        if info:
            st.caption(f"Device : `{info['device']}`")
            st.dataframe(
                pd.DataFrame({"Catégorie": info["labels"], "Seuil": info["thresholds"]}),
                hide_index=True,
                column_config={"Seuil": st.column_config.NumberColumn(format="%.2f")},
            )

    st.header("Liens")
    st.markdown(
        f"- [Swagger de l'API]({PUBLIC_API_URL}/docs)\n"
        f"- [Métriques brutes /metrics]({PUBLIC_API_URL}/metrics)\n"
        f"- [Prometheus]({PROMETHEUS_URL})\n"
        f"- [Grafana]({GRAFANA_URL})\n"
        f"- [Evidently (monitoring du drift)]({EVIDENTLY_URL})"
    )

st.title("Classification d'avis Trustpilot")

# --- Saisie de l'avis ---
st.text_area(
    "Avis client",
    key="review_text",
    height=150,
    on_change=clear_review_meta,
)

meta = st.session_state.review_meta
if meta:
    stars = int(meta["star"]) if pd.notna(meta.get("star")) else 0
    caption = f"Avis tiré du dataset « {meta['dataset']} » · {meta.get('company', '?')} · {'★' * stars}{'☆' * (5 - stars)}"
    if meta["dataset"] == "Entraînement" and all(pd.notna(meta.get(label)) for label in LABEL_COLS):
        labels = [label for label in LABEL_COLS if int(meta[label]) == 1]
        caption += f" · Étiquettes d'entraînement : {', '.join(labels) or 'aucune'}"
    st.caption(caption)

col_random, col_source, col_predict = st.columns([1.2, 1.6, 1])

with col_random:
    st.button("Avis aléatoire", on_click=draw_random_review, use_container_width=True)

with col_source:
    st.selectbox(
        "Dataset",
        list(DATASETS),
        key="random_source",
        label_visibility="collapsed",
    )

with col_predict:
    predict_clicked = st.button("PRÉDIRE", type="primary", use_container_width=True)

if st.session_state.random_error:
    st.error(st.session_state.random_error)

if predict_clicked:
    text = st.session_state.review_text.strip()
    if not text:
        st.warning("Veuillez saisir un avis.")
    else:
        try:
            with st.spinner("Prédiction en cours..."):
                st.session_state.last_prediction = predict(text)
        except requests.RequestException as error:
            st.session_state.last_prediction = None
            st.error(f"Erreur lors de l'appel à l'API : {error}")

if st.session_state.last_prediction:
    show_prediction(*st.session_state.last_prediction)


# --- Simulateur de trafic ---
st.divider()
st.header("Simulateur de trafic")
st.write(
    "Envoie automatiquement des avis du dataset à l'API à intervalle régulier"
)

running = simulation_running()

st.session_state.simulation_on = running
if running:
    params = simulation_state()["params"]
    st.session_state.sim_interval = params["interval"]
    st.session_state.sim_batch_size = params["batch_size"]
    st.session_state.sim_source = next(k for k, v in SIMULATION_SOURCES.items() if v == params["source"])

col_interval, col_batch, col_source = st.columns(3)
with col_interval:
    st.number_input("Intervalle (secondes)", min_value=1, max_value=600, key="sim_interval", disabled=running)
with col_batch:
    st.number_input("Avis par requête", min_value=1, max_value=32, key="sim_batch_size", disabled=running)
with col_source:
    st.selectbox("Source des avis", list(SIMULATION_SOURCES), key="sim_source", disabled=running)

st.toggle("Simulation ON / OFF", key="simulation_on", on_change=on_toggle_simulation)


@st.fragment(run_every=2)
def simulation_monitor():
    logs = read_simulation_logs()
    succeeded = sum(" avis prédits en " in line for line in logs)
    failed = sum(" en échec : " in line for line in logs)

    if simulation_running():
        st.success(
            f"Simulation en cours · {succeeded} requête(s) réussie(s) · {failed} en échec",
            icon="🟢",
        )
    elif simulation_state()["process"] is not None:
        st.info(f"Simulation arrêtée · {succeeded} requête(s) réussie(s) · {failed} en échec", icon="⏹️")
    else:
        st.caption("Simulation à l'arrêt.")

    # Le processus s'est arrêté tout seul: on relance la page pour remettre l'interrupteur à OFF
    if st.session_state.simulation_on and not simulation_running():
        st.rerun(scope="app")

    if logs:
        st.code("\n".join(logs[-12:]), language=None)


simulation_monitor()
