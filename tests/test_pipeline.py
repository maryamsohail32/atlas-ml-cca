"""
Smoke tests for the ATLAS-ML data pipeline.
Run:  pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import PILLARS, build_pair_dataset, build_site_dataset


def test_pair_dataset_shape_and_no_nans():
    df = build_pair_dataset()
    assert len(df) == 56  # 4 bases x 14 sites
    assert df[PILLARS].notna().all().all()


def test_scores_in_valid_range():
    df = build_pair_dataset()
    assert df["analog_score"].between(0, 100).all()


def test_suitability_bands():
    df = build_pair_dataset()
    assert set(df["suitability"]) == {"High", "Moderate", "Low"}
    high = df[df["suitability"] == "High"]
    assert (high["analog_score"] >= 60).all()


def test_site_dataset():
    df = build_site_dataset()
    assert len(df) == 14
    assert df.notna().all().all()
