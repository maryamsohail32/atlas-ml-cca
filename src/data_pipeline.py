"""
ATLAS-ML :: Data Pipeline  (Lab 1 concept: Data Preprocessing)
==============================================================
Builds the machine-learning datasets used by every model in this project
from the raw ATLAS science outputs in ``data/``:

  * ``data/scores_ranking.csv``  - 56 Moon-base / Earth-site pairs, each with
    five per-pillar dissimilarity distances and the final Analog Score.
  * ``data/earth_profiles.csv``  - physical profile of the 14 Earth sites.
  * ``data/moon_profiles.csv``   - physical profile of the 4 lunar bases.

Outputs (written to ``data/``):
  * ``ml_pairs.csv``  - one row per base-site pair, model-ready:
    features = the five pillar distances, target = ``analog_score``,
    class label = ``suitability`` (High / Moderate / Low).
  * ``ml_sites.csv``  - one row per Earth site with numeric physical
    features, used by the clustering models.

Preprocessing steps demonstrated here (Lab 1):
  1. Missing-value handling: ``dist_soil`` is missing for 12 of 56 pairs
     because SoilGrids has no coverage for Antarctica and Etosha Pan.
     We median-impute it AND add a ``soil_missing`` indicator column so
     the models can still learn that the value was absent.
  2. Label engineering: the continuous score is banded into suitability
     classes (High >= 60, Moderate 45-60, Low < 45).
  3. Feature selection: only numeric, leakage-free columns are kept.

Run:  python src/data_pipeline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# The five scoring pillars of the ATLAS engine, in weight order.
PILLARS = ["dist_slope", "dist_flat", "dist_temperature", "dist_relief", "dist_soil"]

# Physical features used to cluster the Earth sites.
SITE_FEATURES = [
    "elev_mean_m",
    "slope_mean_deg",
    "pct_slope_under_10deg",
    "temp_mean_c",
    "temp_seasonal_range_c",
    "sand_pct",
    "clay_pct",
]


def suitability_band(score: float) -> str:
    """Band a continuous Analog Score into a suitability class."""
    if score >= 60:
        return "High"
    if score >= 45:
        return "Moderate"
    return "Low"


def build_pair_dataset() -> pd.DataFrame:
    """One row per Moon-base / Earth-site pair (56 rows), model-ready."""
    scores = pd.read_csv(DATA / "scores_ranking.csv")

    df = scores[["moon_base", "earth_site"] + PILLARS + ["analog_score"]].copy()

    # --- Missing-value handling -------------------------------------------
    # Soil distance is missing where SoilGrids has no coverage (the two
    # Antarctic sites and Etosha Pan: 3 sites x 4 bases = 12 rows).
    df["soil_missing"] = df["dist_soil"].isna().astype(int)
    soil_median = df["dist_soil"].median()
    df["dist_soil"] = df["dist_soil"].fillna(soil_median)

    # --- Label engineering -------------------------------------------------
    df["suitability"] = df["analog_score"].apply(suitability_band)

    assert df[PILLARS].notna().all().all(), "NaNs remain after imputation"
    assert len(df) == 56, f"expected 56 pairs, got {len(df)}"
    return df


def build_site_dataset() -> pd.DataFrame:
    """One row per Earth site (14 rows) with numeric physical features."""
    earth = pd.read_csv(DATA / "earth_profiles.csv")
    df = earth[["site_name", "country", "terrain_type"] + SITE_FEATURES].copy()

    # Soil percentages are missing for the same no-coverage sites.
    for col in ["sand_pct", "clay_pct"]:
        df[col] = df[col].fillna(df[col].median())

    assert df[SITE_FEATURES].notna().all().all(), "NaNs remain in site features"
    assert len(df) == 14, f"expected 14 sites, got {len(df)}"
    return df


def main() -> None:
    pairs = build_pair_dataset()
    pairs.to_csv(DATA / "ml_pairs.csv", index=False)

    sites = build_site_dataset()
    sites.to_csv(DATA / "ml_sites.csv", index=False)

    print("ml_pairs.csv :", pairs.shape, "rows x cols")
    print(pairs["suitability"].value_counts().to_dict())
    print("ml_sites.csv :", sites.shape, "rows x cols")
    print("Pair features:", PILLARS + ["soil_missing"], "| target: analog_score")


if __name__ == "__main__":
    main()
