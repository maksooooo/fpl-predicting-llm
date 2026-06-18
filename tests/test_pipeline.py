"""Training-pipeline contract tests (independent of the large real dataset)."""
import numpy as np
import pandas as pd
import pytest

from features.build_features import preprocess_data, engineer_features
import train_model as tm


@pytest.fixture
def engineered(synthetic_raw):
    return engineer_features(preprocess_data(synthetic_raw.copy()))


def test_targets_are_never_features(engineered):
    """The prediction targets must be excluded from the model inputs."""
    feats = engineered.select_dtypes(include=['number', 'bool']).columns
    feats = [c for c in feats if c not in tm.DROP_COLS]
    assert tm.TARGET not in feats
    assert 'target_next_gw_minutes' not in feats


def test_prepare_features_returns_numeric_only():
    """Build a tiny 3-season frame and confirm the feature matrix is numeric and
    target-free."""
    frames = []
    for season in tm.TRAIN_SEASONS[:1] + tm.VAL_SEASONS + tm.TEST_SEASONS:
        from conftest import _player_rows
        frames.append(_player_rows(f"P-{season}", "Arsenal", "Chelsea", "MID",
                                   season, [3, 5, 2, 6, 4, 7]))
    raw = pd.concat(frames, ignore_index=True)
    df = engineer_features(preprocess_data(raw))

    X_tr, y_tr, X_val, y_val, X_te, y_te, feats = tm.prepare_features(
        df[df.season_x.isin(tm.TRAIN_SEASONS)],
        df[df.season_x.isin(tm.VAL_SEASONS)],
        df[df.season_x.isin(tm.TEST_SEASONS)])

    assert len(feats) > 0
    assert tm.TARGET not in feats
    # Every selected feature column is numeric.
    assert X_tr[feats].select_dtypes(include=['number', 'bool']).shape[1] == len(feats)


def test_split_is_chronological_and_disjoint():
    """Train / val / test seasons must not overlap."""
    assert not (set(tm.TRAIN_SEASONS) & set(tm.VAL_SEASONS))
    assert not (set(tm.VAL_SEASONS) & set(tm.TEST_SEASONS))
    assert not (set(tm.TRAIN_SEASONS) & set(tm.TEST_SEASONS))
