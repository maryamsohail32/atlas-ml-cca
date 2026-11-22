"""
ATLAS-ML :: Backend API (FastAPI)
==================================
Serves the ATLAS-ML datasets, model results and live predictions to the
frontend. The prediction endpoint loads the best trained regressor
(saved by src/models/regression.py); if the model artifact is not
present yet it falls back to the original ATLAS formula so the API
never breaks during development:

    Analog Score = 100 * exp(-sqrt(sum(w_i * d_i^2)))

with pillar weights slope .30, flat .20, temperature .20, soil .15,
relief .15 (soil weight is redistributed when soil data is missing).

Endpoints:
  GET  /                    health + counts
  GET  /api/pairs           all 56 base-site pairs with features + score
  GET  /api/rankings        rankings for one base (?base=Shackleton)
  GET  /api/sites           the 14 Earth sites with cluster assignments
  GET  /api/metrics         model comparison tables (once models are run)
  POST /api/predict         {distances...} -> predicted Analog Score

Run:  uvicorn backend.main:app --port 8000   (from the repo root)
"""

import json
import math
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
METRICS = ROOT / "outputs" / "metrics"
MODEL_FILE = ROOT / "outputs" / "models" / "best_regressor.joblib"

WEIGHTS = {
    "dist_slope": 0.30,
    "dist_flat": 0.20,
    "dist_temperature": 0.20,
    "dist_soil": 0.15,
    "dist_relief": 0.15,
}

app = FastAPI(title="ATLAS-ML API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def load_pairs() -> pd.DataFrame:
    path = DATA / "ml_pairs.csv"
    if not path.exists():
        raise HTTPException(503, "ml_pairs.csv missing: run src/data_pipeline.py")
    return pd.read_csv(path)


@app.get("/")
def health():
    pairs = load_pairs()
    return {
        "status": "ok",
        "pairs": int(len(pairs)),
        "bases": sorted(pairs["moon_base"].unique().tolist()),
        "model_loaded": MODEL_FILE.exists(),
    }


@app.get("/api/pairs")
def pairs():
    return load_pairs().to_dict(orient="records")


@app.get("/api/rankings")
def rankings(base: str):
    df = load_pairs()
    sub = df[df["moon_base"] == base].sort_values("analog_score", ascending=False)
    if sub.empty:
        raise HTTPException(404, f"unknown base: {base}")
    return sub.to_dict(orient="records")


@app.get("/api/sites")
def sites():
    path = METRICS / "site_clusters.csv"
    if not path.exists():
        raise HTTPException(503, "site_clusters.csv missing: run clustering.py")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/api/metrics")
def metrics():
    out = {}
    for name in ["regression_results", "classification_results"]:
        path = METRICS / f"{name}.csv"
        if path.exists():
            out[name] = pd.read_csv(path).to_dict(orient="records")
    cluster_path = METRICS / "clustering_results.json"
    if cluster_path.exists():
        out["clustering_results"] = json.loads(cluster_path.read_text())
    return out


class PredictRequest(BaseModel):
    dist_slope: float
    dist_flat: float
    dist_temperature: float
    dist_relief: float
    dist_soil: float | None = None  # None = no soil coverage (e.g. Antarctica)


@app.post("/api/predict")
def predict(req: PredictRequest):
    soil_missing = 1 if req.dist_soil is None else 0
    soil_value = req.dist_soil if req.dist_soil is not None else 0.5  # training median

    if MODEL_FILE.exists():
        model = joblib.load(MODEL_FILE)
        features = pd.DataFrame(
            [
                {
                    "dist_slope": req.dist_slope,
                    "dist_flat": req.dist_flat,
                    "dist_temperature": req.dist_temperature,
                    "dist_relief": req.dist_relief,
                    "dist_soil": soil_value,
                    "soil_missing": soil_missing,
                }
            ]
        )
        score = float(model.predict(features)[0])
        return {"predicted_score": round(score, 1), "source": "trained ML model"}

    # Fallback: the original ATLAS formula (weights renormalised if no soil).
    weights = dict(WEIGHTS)
    values = {
        "dist_slope": req.dist_slope,
        "dist_flat": req.dist_flat,
        "dist_temperature": req.dist_temperature,
        "dist_relief": req.dist_relief,
        "dist_soil": req.dist_soil,
    }
    if req.dist_soil is None:
        total = sum(w for k, w in weights.items() if k != "dist_soil")
        weights = {
            k: (w / total if k != "dist_soil" else 0.0) for k, w in weights.items()
        }
        values["dist_soil"] = 0.0
    d = math.sqrt(sum(weights[k] * values[k] ** 2 for k in weights))
    return {"predicted_score": round(100 * math.exp(-d), 1), "source": "ATLAS formula"}
