"""
ATLAS-ML :: Frontend dashboard (Streamlit)
===========================================
Interactive front end for the ATLAS-ML project. It talks to the FastAPI
backend (backend/main.py) and falls back to the local CSV files when the
backend is not running, so the demo never breaks.

Sections:
  1. Rankings explorer   - pick a lunar base, see its ranked Earth analogs
  2. Model results       - regression / classification / clustering tables
                           and plots produced by the model scripts
  3. Predict playground  - move the five pillar-distance sliders and get a
                           predicted Analog Score from the trained model

Run:  streamlit run frontend/app.py        (from the repo root)
      (optional backend: uvicorn backend.main:app --port 8000)
"""

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
API = os.environ.get("ATLAS_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="ATLAS-ML", layout="wide")
st.title("ATLAS-ML: Lunar Analog Site Ranking with Machine Learning")
st.caption(
    "Which places on Earth can stand in for the planned permanent Moon bases? "
    "ATLAS profiles 4 lunar polar bases and 14 Earth sites from real NASA/ESA "
    "data, then machine learning models rank, classify and cluster the matches."
)


def api_get(path: str):
    try:
        r = requests.get(f"{API}{path}", timeout=3)
        if r.ok:
            return r.json()
    except requests.RequestException:
        pass
    return None


@st.cache_data
def load_pairs() -> pd.DataFrame:
    data = api_get("/api/pairs")
    if data:
        return pd.DataFrame(data)
    return pd.read_csv(ROOT / "data" / "ml_pairs.csv")


pairs = load_pairs()

# ---------------------------------------------------------------- rankings
st.header("1. Rankings explorer")
base = st.selectbox("Lunar base", sorted(pairs["moon_base"].unique()))
sub = pairs[pairs["moon_base"] == base].sort_values("analog_score", ascending=False)
st.dataframe(
    sub[["earth_site", "analog_score", "suitability"]].reset_index(drop=True),
    use_container_width=True,
)
st.bar_chart(sub.set_index("earth_site")["analog_score"])

# ------------------------------------------------------------------ models
st.header("2. Model results")
metrics_dir = ROOT / "outputs" / "metrics"
plot_dir = ROOT / "outputs" / "plots"

col1, col2 = st.columns(2)
with col1:
    st.subheader("Regression: predicting the Analog Score")
    path = metrics_dir / "regression_results.csv"
    if path.exists():
        st.dataframe(pd.read_csv(path), use_container_width=True)
    img = plot_dir / "rf_feature_importance.png"
    if img.exists():
        st.image(str(img), caption="Which pillars drive the score?")
with col2:
    st.subheader("Classification: suitability class")
    path = metrics_dir / "classification_results.csv"
    if path.exists():
        st.dataframe(pd.read_csv(path), use_container_width=True)
    img = plot_dir / "confusion_matrix.png"
    if img.exists():
        st.image(str(img), caption="Confusion matrix (best classifier, CV)")

st.subheader("Clustering: environmental regimes of the Earth sites")
path = metrics_dir / "site_clusters.csv"
if path.exists():
    st.dataframe(pd.read_csv(path), use_container_width=True)
img = plot_dir / "cluster_scatter.png"
if img.exists():
    st.image(str(img))

# ----------------------------------------------------------------- predict
st.header("3. Predict playground")
st.write(
    "Set the five pillar distances of a hypothetical base-site pair "
    "(0 = identical, 1 = completely different) and predict its Analog Score."
)
cols = st.columns(5)
values = {}
for col, name in zip(
    cols, ["dist_slope", "dist_flat", "dist_temperature", "dist_relief", "dist_soil"]
):
    values[name] = col.slider(name.replace("dist_", ""), 0.0, 1.0, 0.3, 0.01)

if st.button("Predict Analog Score"):
    result = None
    try:
        r = requests.post(f"{API}/api/predict", json=values, timeout=3)
        if r.ok:
            result = r.json()
    except requests.RequestException:
        pass
    if result:
        st.success(
            f"Predicted Analog Score: **{result['predicted_score']}** "
            f"(source: {result['source']})"
        )
    else:
        st.warning("Backend is not running, start it with: uvicorn backend.main:app")
