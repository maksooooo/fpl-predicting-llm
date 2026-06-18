"""Performance-regression guard.

Loads the trained model + test predictions and asserts the model still clears
known quality bars. Skips cleanly if the artifacts haven't been generated yet
(run `./run_pipeline.sh` first). This catches silent regressions where a code
change quietly makes the model worse.
"""
import os
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "histgb_model.pkl")
TEST = os.path.join(ROOT, "data", "processed", "test_data_with_targets.csv")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(MODEL) and os.path.exists(TEST)),
    reason="Model/test artifacts not built yet — run ./run_pipeline.sh")


@pytest.fixture(scope="module")
def artifacts():
    with open(MODEL, "rb") as f:
        bundle = pickle.load(f)
    df = pd.read_csv(TEST)
    return bundle["model"], bundle["features"], df


def test_model_beats_mae_threshold(artifacts):
    model, feats, df = artifacts
    pred = model.predict(df[feats])
    mae = mean_absolute_error(df["target_next_gw_points"], pred)
    assert mae < 0.90, f"Overall test MAE regressed to {mae:.4f}"


def test_model_beats_naive_baselines(artifacts):
    """The model must beat 'repeat last GW' and 'repeat 3-game form'."""
    model, feats, df = artifacts
    y = df["target_next_gw_points"]
    pred = model.predict(df[feats])
    model_mae = mean_absolute_error(y, pred)
    last_gw = mean_absolute_error(y, df["lag_1_total_points"].fillna(0))
    form = mean_absolute_error(y, df["rolling_3_avg_total_points"].fillna(0))
    assert model_mae < last_gw
    assert model_mae < form


def test_featured_player_mae_reasonable(artifacts):
    """On players who featured, MAE should stay within a sane band."""
    model, feats, df = artifacts
    played = df["minutes"] > 0
    pred = model.predict(df[feats][played])
    mae = mean_absolute_error(df["target_next_gw_points"][played], pred)
    assert 1.0 < mae < 2.2, f"Featured-player MAE drifted to {mae:.4f}"


def test_predictions_are_finite(artifacts):
    model, feats, df = artifacts
    pred = model.predict(df[feats])
    assert np.isfinite(pred).all()
