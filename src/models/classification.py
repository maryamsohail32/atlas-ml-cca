"""
ATLAS-ML :: Classification models  (Labs 3, 4 and 6 concepts)
==============================================================
Goal: classify a Moon-base / Earth-site pair into a suitability class
directly from its five pillar distances:

    High      Analog Score >= 60     (13 pairs)
    Moderate  Analog Score 45 - 60   (36 pairs)
    Low       Analog Score <  45     ( 7 pairs)

The classes are imbalanced (Moderate dominates), which is handled with
stratified cross-validation and class_weight="balanced" where the
algorithm supports it, and reported honestly with per-class metrics
instead of a single accuracy number.

Models compared:
  * Logistic Regression   (Lab 3)
  * Decision Tree         (Lab 4)
  * Support Vector Machine (Lab 6)

Outputs: outputs/metrics/classification_results.csv,
         outputs/metrics/classification_report.txt (best model),
         outputs/plots/confusion_matrix.png (best model, CV predictions).

Run:  python src/models/classification.py   (after src/data_pipeline.py)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_pipeline import PILLARS, build_pair_dataset  # noqa: E402

FEATURES = PILLARS + ["soil_missing"]
TARGET = "suitability"
RANDOM_STATE = 42


def get_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "SVM (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
    }


def main() -> None:
    df = build_pair_dataset()
    X, y = df[FEATURES], df[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, cv_preds = [], {}
    for name, model in get_models().items():
        pred = cross_val_predict(model, X, y, cv=cv)
        cv_preds[name] = pred
        acc = accuracy_score(y, pred)
        rows.append({"model": name, "cv_accuracy": round(acc, 4)})
        print(f"{name:22s} CV accuracy = {acc:.4f}")

    results = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False)
    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "classification_results.csv", index=False)

    best_name = results.iloc[0]["model"]
    print("Best classifier:", best_name)

    report = classification_report(y, cv_preds[best_name], digits=3)
    (out_dir / "classification_report.txt").write_text(
        f"Best model: {best_name} (5-fold stratified CV predictions)\n\n{report}"
    )

    labels = ["High", "Moderate", "Low"]
    cm = confusion_matrix(y, cv_preds[best_name], labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix: {best_name}")
    fig.tight_layout()
    plot_dir = ROOT / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "confusion_matrix.png", dpi=150)
    print("Saved classification results, report and confusion matrix")


if __name__ == "__main__":
    main()
