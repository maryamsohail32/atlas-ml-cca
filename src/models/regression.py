"""
ATLAS-ML :: Regression models  (Labs 2, 5 and 8 concepts)
==========================================================
Goal: learn a *surrogate* of the ATLAS scoring engine. Given the five
per-pillar dissimilarity distances of a Moon-base / Earth-site pair,
predict its Analog Score (0-100).

Why this is a valid ML task: the original engine computes the score with
a fixed hand-weighted formula (Analog Score = 100 * exp(-D)). A learned
model can approximate that mapping from examples alone, which is exactly
what a surrogate model does in engineering: a fast, trainable stand-in
for an expensive or fixed computation. Comparing model families also
shows which pillars actually drive the score (feature importance).

Models compared:
  * Linear Regression            (Lab 2)
  * Random Forest Regressor      (Lab 5, ensemble)
  * K-Nearest Neighbours         (Lab 8, k = 5)

Evaluation: held-out test split (80/20) + 5-fold cross-validation.
Metrics: MAE, RMSE, R^2. Results -> outputs/metrics/regression_results.csv
The best model is saved to outputs/models/best_regressor.joblib and its
Random Forest feature importances are plotted.

Run:  python src/models/regression.py   (after src/data_pipeline.py)
"""

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_pipeline import PILLARS, build_pair_dataset  # noqa: E402

FEATURES = PILLARS + ["soil_missing"]
TARGET = "analog_score"
RANDOM_STATE = 42


def get_models() -> dict:
    """Model zoo. Distance-based and linear models get scaled features."""
    return {
        "Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "KNN Regressor (k=5)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5)),
            ]
        ),
    }


def evaluate(model, X, y) -> dict:
    """Cross-validated metrics over the full dataset (n = 56 is small,
    so 5-fold CV predictions are more trustworthy than one split)."""
    pred = cross_val_predict(model, X, y, cv=5)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": r2_score(y, pred),
    }


def main() -> None:
    df = build_pair_dataset()
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    rows, fitted = [], {}
    for name, model in get_models().items():
        cv = evaluate(model, X, y)
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "cv_MAE": round(cv["MAE"], 3),
                "cv_RMSE": round(cv["RMSE"], 3),
                "cv_R2": round(cv["R2"], 4),
                "test_R2": round(r2_score(y_test, test_pred), 4),
            }
        )
        fitted[name] = model
        print(f"{name:22s} CV R2={cv['R2']:.4f}  CV RMSE={cv['RMSE']:.3f}")

    results = pd.DataFrame(rows).sort_values("cv_R2", ascending=False)
    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "regression_results.csv", index=False)

    # Save the best model (by CV R^2) for the backend to serve.
    best_name = results.iloc[0]["model"]
    model_dir = ROOT / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], model_dir / "best_regressor.joblib")
    print("Best regressor:", best_name)

    # Feature importance plot (Random Forest).
    rf = fitted["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot.barh(ax=ax, color="#2b6cb0")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance: predicting the Analog Score")
    fig.tight_layout()
    plot_dir = ROOT / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "rf_feature_importance.png", dpi=150)
    print("Saved outputs/metrics/regression_results.csv and importance plot")


if __name__ == "__main__":
    main()
