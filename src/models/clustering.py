"""
ATLAS-ML :: Clustering models  (Labs 10 and 11 concepts)
=========================================================
Goal: group the 14 Earth analog sites into environmental regimes using
only their physical features (elevation, slope, flatness, temperature,
seasonality, soil) - no labels are given to the algorithms.

Two unsupervised methods are compared:
  * K-Means                 (Lab 10) - k chosen by silhouette score scan
  * Hierarchical Agglomerative (Lab 11) - Ward linkage, dendrogram

Agreement between the two partitions is measured with the Adjusted Rand
Index (ARI): 1.0 = identical grouping, 0 = chance-level agreement.

Outputs: outputs/metrics/clustering_results.json,
         outputs/metrics/site_clusters.csv,
         outputs/plots/dendrogram.png, outputs/plots/cluster_scatter.png.

Run:  python src/models/clustering.py   (after src/data_pipeline.py)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_pipeline import SITE_FEATURES, build_site_dataset  # noqa: E402

RANDOM_STATE = 42


def choose_k(X_scaled) -> tuple[int, dict]:
    """Silhouette scan for k = 2..6; returns the best k and all scores."""
    scores = {}
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(
            X_scaled
        )
        scores[k] = round(float(silhouette_score(X_scaled, labels)), 4)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def main() -> None:
    sites = build_site_dataset()
    X = StandardScaler().fit_transform(sites[SITE_FEATURES])

    best_k, sil_scores = choose_k(X)
    print("Silhouette by k:", sil_scores, "-> best k =", best_k)

    km_labels = KMeans(
        n_clusters=best_k, n_init=10, random_state=RANDOM_STATE
    ).fit_predict(X)
    hc_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)

    ari = float(adjusted_rand_score(km_labels, hc_labels))
    print(f"K-Means vs Hierarchical agreement (ARI) = {ari:.3f}")

    out = sites[["site_name", "country", "terrain_type"]].copy()
    out["kmeans_cluster"] = km_labels
    out["hierarchical_cluster"] = hc_labels
    metrics_dir = ROOT / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(metrics_dir / "site_clusters.csv", index=False)

    summary = {
        "n_sites": int(len(sites)),
        "features": SITE_FEATURES,
        "silhouette_by_k": sil_scores,
        "best_k": best_k,
        "kmeans_silhouette": round(float(silhouette_score(X, km_labels)), 4),
        "hierarchical_silhouette": round(float(silhouette_score(X, hc_labels)), 4),
        "adjusted_rand_index": round(ari, 4),
    }
    (metrics_dir / "clustering_results.json").write_text(json.dumps(summary, indent=2))

    # --- Plots --------------------------------------------------------------
    plot_dir = ROOT / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    dendrogram(linkage(X, method="ward"), labels=sites["site_name"].tolist(), ax=ax)
    ax.set_title("Hierarchical clustering of Earth analog sites (Ward linkage)")
    ax.set_ylabel("Distance")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "dendrogram.png", dpi=150)

    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=km_labels, cmap="tab10", s=90)
    for (x, y), name in zip(coords, sites["site_name"]):
        ax.annotate(name.split(" (")[0], (x, y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"K-Means regimes of Earth analog sites (k = {best_k})")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(plot_dir / "cluster_scatter.png", dpi=150)
    print("Saved clustering metrics, assignments and plots")


if __name__ == "__main__":
    main()
