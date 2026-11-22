# ATLAS-ML Backend API

FastAPI service that serves the datasets, model results and live
predictions to the frontend.

## Run

From the repo root (after the data pipeline and model scripts have run):

```bash
uvicorn backend.main:app --port 8000
```

Interactive API docs: http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check, pair count, whether the trained model is loaded |
| GET | `/api/pairs` | All 56 base-site pairs with features, score and suitability |
| GET | `/api/rankings?base=Shackleton` | Ranked Earth analogs for one Moon base |
| GET | `/api/sites` | The 14 Earth sites with K-Means and Hierarchical cluster labels |
| GET | `/api/metrics` | Regression, classification and clustering result tables |
| POST | `/api/predict` | Predict an Analog Score from five pillar distances |

## Prediction behaviour

`POST /api/predict` loads the best trained regressor from
`outputs/models/best_regressor.joblib`. If that file does not exist yet,
it falls back to the original ATLAS formula
(`100 * exp(-weighted Euclidean distance)`), so the API works at every
stage of the project. Send `"dist_soil": null` for pairs with no soil
coverage (Antarctica, Etosha Pan); the pillar is then handled the same
way as in the training data.
