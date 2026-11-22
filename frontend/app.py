"""
ATLAS-ML :: Frontend dashboard (Streamlit), mission-control edition
====================================================================
Interactive front end for the ATLAS-ML project, restyled in the original
ATLAS mission-control console design system (dark glass panels, one orange
accent for action and best match, one mint accent for positive and
runner-up, Inter for UI text, DM Mono for telemetry and numbers).
It talks to the FastAPI backend (backend/main.py) and falls back to the
local CSV files when the backend is not running, so the demo never breaks.

Sections:
  1. Rankings explorer   - pick a lunar base, see its ranked Earth analogs
  2. Model results       - regression / classification / clustering tables
                           and plots produced by the model scripts
  3. Predict playground  - move the five pillar-distance sliders and get a
                           predicted Analog Score from the trained model

Run:  streamlit run frontend/app.py        (from the repo root)
      (optional backend: uvicorn backend.main:app --port 8000)
      (the repo .streamlit/config.toml carries the matching dark theme)
"""

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
API = os.environ.get("ATLAS_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="ATLAS-ML", page_icon="🌕", layout="wide")

# ------------------------------------------------------- mission-control CSS
# Design tokens lifted from the original ATLAS console (webapp styles.css):
# bg #04070a, glass #0d1218 / rgba(13,18,24,.82), borders rgba(255,255,255,.08),
# text #eef1f4, muted #98a3ad, dim #77828d, orange #ff7a1a, mint #4fe3c1,
# live green #58d977, radius 10px, micro labels uppercase + letter-spaced.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

.stApp{
  background:
    radial-gradient(900px 480px at 78% -5%, rgba(255,122,26,.055) 0%, transparent 60%),
    radial-gradient(760px 520px at 12% 108%, #0a141e 0%, transparent 60%),
    radial-gradient(700px 400px at 70% 20%, #0c1722 0%, transparent 60%),
    #04070a;
  color:#eef1f4;
  font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  -webkit-font-smoothing:antialiased;
}
[data-testid="stAppViewContainer"]{background:transparent}
[data-testid="stAppViewContainer"] > .main{background:transparent}
.block-container{padding-top:1.6rem; padding-bottom:2.5rem; max-width:1460px}

/* hide default Streamlit chrome */
[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"]{display:none}
[data-testid="stDecoration"]{display:none}
[data-testid="stStatusWidget"]{display:none}
#MainMenu{visibility:hidden}
footer{visibility:hidden}

/* sidebar: dark glass */
[data-testid="stSidebar"]{
  background:rgba(8,11,15,.94);
  border-right:1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] *{color:#eef1f4}

/* typography */
h1,h2,h3,h4{color:#eef1f4; font-family:'Inter',sans-serif; letter-spacing:.01em}
p, li, label, [data-testid="stMarkdownContainer"] p{color:#d7dee5}
[data-testid="stCaptionContainer"] p{color:#98a3ad}

/* console micro labels + section titles */
.mc-k{
  font-size:10px; letter-spacing:.18em; text-transform:uppercase;
  color:#ff7a1a; font-weight:700; margin:0 0 2px 0;
}
.mc-k.dim{color:#77828d}
.mc-h{font-size:26px; font-weight:700; color:#eef1f4; margin:0 0 10px 0; line-height:1.15}
.mc-sub-h{font-size:16.5px; font-weight:600; color:#eef1f4; margin:6px 0 8px 0}

/* top strip */
.mc-topbar{
  display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding:9px 14px; margin-bottom:14px;
  background:rgba(8,11,15,.88); border:1px solid rgba(255,255,255,.08); border-radius:10px;
}
.mc-brand{font-size:10.5px; letter-spacing:.18em; font-weight:700; color:#eef1f4; text-transform:uppercase}
.mc-brand .org{color:#ff7a1a}
.mc-top-right{display:flex; align-items:center; gap:18px}
.mc-course{font-size:9px; letter-spacing:.16em; color:#77828d; text-transform:uppercase; font-weight:600}
.mc-live{display:flex; align-items:center; gap:7px; font-size:10.5px; font-weight:700; letter-spacing:.12em; color:#eef1f4}
.live-dot{
  width:8px; height:8px; border-radius:50%; background:#58d977;
  box-shadow:0 0 8px rgba(88,217,119,.85); animation:mc-pulse 2.4s infinite;
}
@keyframes mc-pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* hero */
.mc-title{font-size:52px; font-weight:800; line-height:1.02; color:#eef1f4; margin:2px 0 6px 0}
.mc-title .org{color:#ff7a1a}
.mc-tag{font-size:15.5px; line-height:1.55; color:#98a3ad; max-width:880px; margin:0 0 16px 0}

/* glass stat cards */
.mc-stats{display:flex; gap:10px; flex-wrap:wrap; margin:4px 0 8px 0}
.mc-stat{
  flex:1 1 150px; background:rgba(13,18,24,.82); border:1px solid rgba(255,255,255,.08);
  border-radius:10px; padding:11px 14px 10px;
  box-shadow:0 10px 28px rgba(0,0,0,.35);
}
.mc-stat .k{font-size:9px; letter-spacing:.16em; color:#77828d; text-transform:uppercase; font-weight:700}
.mc-stat .v{font-family:'DM Mono', ui-monospace, monospace; font-size:25px; color:#eef1f4; margin-top:4px; line-height:1.1}
.mc-stat.hot{border-color:rgba(255,122,26,.5)}
.mc-stat.hot .v{color:#ff7a1a}
.mc-stat.mint .v{color:#4fe3c1}

/* best match + runner-up cards */
.mc-best{
  background:linear-gradient(135deg, rgba(255,122,26,.16) 0%, rgba(13,18,24,.86) 58%);
  border:1px solid rgba(255,122,26,.5); border-radius:10px; padding:15px 18px 13px;
  box-shadow:0 10px 28px rgba(0,0,0,.45); height:100%;
}
.mc-runner{
  background:rgba(13,18,24,.82);
  border:1px solid rgba(79,227,193,.35); border-radius:10px; padding:15px 18px 13px;
  box-shadow:0 10px 28px rgba(0,0,0,.45); height:100%;
}
.mc-chip{
  display:inline-block; font-size:9px; font-weight:700; letter-spacing:.1em;
  color:#ff7a1a; background:rgba(255,122,26,.14); border:1px solid rgba(255,122,26,.45);
  padding:2px 8px; border-radius:4px; text-transform:uppercase;
}
.mc-chip.mint{color:#4fe3c1; background:rgba(79,227,193,.1); border-color:rgba(79,227,193,.4)}
.mc-card-name{font-size:24px; font-weight:700; color:#eef1f4; margin-top:9px; line-height:1.2}
.mc-card-score{font-family:'DM Mono', ui-monospace, monospace; font-size:42px; line-height:1.05; color:#ff7a1a; margin-top:2px}
.mc-runner .mc-card-score{color:#4fe3c1; font-size:34px}
.mc-card-meta{font-size:10px; letter-spacing:.1em; color:#98a3ad; text-transform:uppercase; font-weight:600; margin-top:5px}

/* widget panels */
[data-testid="stDataFrame"]{
  border:1px solid rgba(255,255,255,.08); border-radius:10px; overflow:hidden;
  background:rgba(13,18,24,.82);
}
[data-testid="stImage"] img{
  border:1px solid rgba(255,255,255,.08); border-radius:10px; background:#0d1218;
}
[data-testid="stImage"] [data-testid="stCaptionContainer"] p{color:#98a3ad; font-size:12px}

/* selectbox */
[data-testid="stSelectbox"] label p{
  font-size:9.5px; letter-spacing:.16em; text-transform:uppercase; color:#77828d; font-weight:700;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div{
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.12);
  border-radius:8px; color:#eef1f4;
}
[data-testid="stSelectbox"] [data-baseweb="select"] svg{fill:#98a3ad}
ul[data-baseweb="menu"]{background:#0d1218; border:1px solid rgba(255,255,255,.1)}
ul[data-baseweb="menu"] li{color:#eef1f4}

/* sliders */
[data-testid="stSlider"] label p{
  font-family:'DM Mono', ui-monospace, monospace; font-size:12.5px; color:#98a3ad;
  text-transform:uppercase; letter-spacing:.06em;
}

/* primary action button: orange */
.stButton > button{
  background:#ff7a1a; color:#ffffff; border:1px solid #ff7a1a; border-radius:8px;
  font-weight:700; font-size:13.5px; letter-spacing:.04em; padding:.55rem 1.3rem;
  box-shadow:0 6px 20px rgba(255,122,26,.35);
}
.stButton > button:hover{background:#ff8b3d; border-color:#ff8b3d; color:#ffffff}
.stButton > button:active{background:#e56a10; border-color:#e56a10; color:#ffffff}
.stButton > button:focus:not(:active){
  background:#ff7a1a; border-color:#ffb37e; color:#ffffff;
  box-shadow:0 0 0 3px rgba(255,122,26,.35);
}

/* alerts: dark glass */
[data-testid="stAlert"]{
  background:rgba(13,18,24,.88); border:1px solid rgba(255,255,255,.1);
  border-radius:10px; color:#eef1f4;
}
[data-testid="stAlert"] p{color:#eef1f4; font-size:15px}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] strong{
  font-family:'DM Mono', ui-monospace, monospace; color:#4fe3c1;
}

/* metric widget in DM Mono */
[data-testid="stMetricValue"]{font-family:'DM Mono', ui-monospace, monospace; color:#eef1f4}
[data-testid="stMetricLabel"] p{
  font-size:9px; letter-spacing:.16em; text-transform:uppercase; color:#77828d; font-weight:700;
}

hr{border-color:rgba(255,255,255,.08)}
::-webkit-scrollbar{width:9px; height:9px}
::-webkit-scrollbar-thumb{background:#26313b; border-radius:5px}
::-webkit-scrollbar-track{background:transparent}
</style>
""",
    unsafe_allow_html=True,
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


# ------------------------------------------------------------ console header
def _headline_stats():
    """Real headline numbers: counts from the pairs data, best scores from
    the saved model metrics (with the verified values as fallback)."""
    best_r2, best_acc = "0.982", "91.1%"
    try:
        reg = pd.read_csv(ROOT / "outputs" / "metrics" / "regression_results.csv")
        best_r2 = f"{reg['cv_R2'].max():.3f}"
    except Exception:
        pass
    try:
        cls = pd.read_csv(ROOT / "outputs" / "metrics" / "classification_results.csv")
        best_acc = f"{cls['cv_accuracy'].max() * 100:.1f}%"
    except Exception:
        pass
    return best_r2, best_acc


_best_r2, _best_acc = _headline_stats()

st.markdown(
    f"""
<div class="mc-topbar">
  <div class="mc-brand">ATLAS-ML <span class="org">//</span> MISSION CONTROL</div>
  <div class="mc-top-right">
    <div class="mc-course">ML LAB AI-3202 · LAB 14 CCA · DAWOOD UNIVERSITY</div>
    <div class="mc-live"><span class="live-dot"></span>LIVE</div>
  </div>
</div>
<div class="mc-k">Analog Terrain &amp; Lunar Alignment System · ML Edition</div>
<div class="mc-title">ATLAS<span class="org">-ML</span></div>
<p class="mc-tag">Which places on Earth can stand in for the planned permanent Moon bases?
ATLAS profiles 4 lunar polar bases and 14 Earth sites from real NASA/ESA data, then machine
learning models rank, classify and cluster the matches.</p>
<div class="mc-stats">
  <div class="mc-stat"><div class="k">Base-Site Pairs</div><div class="v">{len(pairs)}</div></div>
  <div class="mc-stat"><div class="k">Earth Sites</div><div class="v">{pairs['earth_site'].nunique()}</div></div>
  <div class="mc-stat"><div class="k">Lunar Bases</div><div class="v">{pairs['moon_base'].nunique()}</div></div>
  <div class="mc-stat hot"><div class="k">Best Analog Score</div><div class="v">{pairs['analog_score'].max():.1f}</div></div>
  <div class="mc-stat mint"><div class="k">Best Regression R2</div><div class="v">{_best_r2}</div></div>
  <div class="mc-stat mint"><div class="k">Best Classification</div><div class="v">{_best_acc}</div></div>
</div>
""",
    unsafe_allow_html=True,
)


def section_head(number: str, title: str):
    st.markdown(
        f'<div class="mc-k">{number} // Console</div><div class="mc-h">{title}</div>',
        unsafe_allow_html=True,
    )


def sub_head(text: str):
    st.markdown(f'<div class="mc-sub-h">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- rankings
section_head("01", "Rankings explorer")
base = st.selectbox("Lunar base", sorted(pairs["moon_base"].unique()))
sub = pairs[pairs["moon_base"] == base].sort_values("analog_score", ascending=False)

_top = sub.iloc[0]
_runner = sub.iloc[1]
_card_col, _runner_col = st.columns(2)
with _card_col:
    st.markdown(
        f"""
<div class="mc-best">
  <span class="mc-chip">★ Best Match</span>
  <div class="mc-card-name">{_top['earth_site']}</div>
  <div class="mc-card-score">{_top['analog_score']:.1f}</div>
  <div class="mc-card-meta">Analog Score · {_top['suitability']} suitability · for {base}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with _runner_col:
    st.markdown(
        f"""
<div class="mc-runner">
  <span class="mc-chip mint">Runner-Up</span>
  <div class="mc-card-name">{_runner['earth_site']}</div>
  <div class="mc-card-score">{_runner['analog_score']:.1f}</div>
  <div class="mc-card-meta">Analog Score · {_runner['suitability']} suitability · for {base}</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.dataframe(
    sub[["earth_site", "analog_score", "suitability"]].reset_index(drop=True),
    use_container_width=True,
)
st.bar_chart(sub.set_index("earth_site")["analog_score"], color="#ff7a1a")

# ------------------------------------------------------------------ models
section_head("02", "Model results")
metrics_dir = ROOT / "outputs" / "metrics"
plot_dir = ROOT / "outputs" / "plots"

col1, col2 = st.columns(2)
with col1:
    sub_head("Regression: predicting the Analog Score")
    path = metrics_dir / "regression_results.csv"
    if path.exists():
        st.dataframe(pd.read_csv(path), use_container_width=True)
    img = plot_dir / "rf_feature_importance.png"
    if img.exists():
        st.image(str(img), caption="Which pillars drive the score?")
with col2:
    sub_head("Classification: suitability class")
    path = metrics_dir / "classification_results.csv"
    if path.exists():
        st.dataframe(pd.read_csv(path), use_container_width=True)
    img = plot_dir / "confusion_matrix.png"
    if img.exists():
        st.image(str(img), caption="Confusion matrix (best classifier, CV)")

sub_head("Clustering: environmental regimes of the Earth sites")
path = metrics_dir / "site_clusters.csv"
if path.exists():
    st.dataframe(pd.read_csv(path), use_container_width=True)
_clu_col, _den_col = st.columns(2)
with _clu_col:
    img = plot_dir / "cluster_scatter.png"
    if img.exists():
        st.image(str(img), caption="K-Means regimes of the Earth sites (PCA view)")
with _den_col:
    img = plot_dir / "dendrogram.png"
    if img.exists():
        st.image(str(img), caption="Hierarchical clustering dendrogram (Ward)")

# ----------------------------------------------------------------- predict
section_head("03", "Predict playground")
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
