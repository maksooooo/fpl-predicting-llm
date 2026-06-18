"""Feature-engineering correctness and target-leakage tests."""
import numpy as np
import pandas as pd
import pytest

from features.build_features import preprocess_data, engineer_features


@pytest.fixture
def engineered(synthetic_raw):
    return engineer_features(preprocess_data(synthetic_raw.copy()))


def _player(df, name):
    return df[df['name'] == name].sort_values('GW').reset_index(drop=True)


def test_target_is_next_gw_points(engineered):
    """target_next_gw_points[n] must equal total_points[n+1] for the player."""
    a = _player(engineered, "Player A")
    # Player A points were [2,5,1,8,3,6]; last row dropped (no next), so targets
    # for GW1..GW5 are the *following* GW's points: 5,1,8,3,6.
    assert list(a['target_next_gw_points']) == [5, 1, 8, 3, 6]


def test_lag_and_rolling_use_only_the_past(engineered):
    """Rolling/lag features must never include the current or future match."""
    a = _player(engineered, "Player A")
    pts = [2, 5, 1, 8, 3, 6]
    # lag_1 at row i is the previous match's points (NaN for the first).
    assert np.isnan(a.loc[0, 'lag_1_total_points'])
    assert list(a['lag_1_total_points'].iloc[1:]) == pts[:-1][:len(a) - 1]
    # rolling_3 at row i = mean of up to 3 *previous* matches (current excluded).
    for i in range(len(a)):
        expected = np.mean(pts[max(0, i - 3):i]) if i > 0 else np.nan
        got = a.loc[i, 'rolling_3_avg_total_points']
        if i == 0:
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected)


def test_no_future_leakage_when_future_is_mutated(synthetic_raw):
    """Gold-standard leakage check: changing a FUTURE match must not change a
    past row's features."""
    base = engineer_features(preprocess_data(synthetic_raw.copy()))

    mutated_raw = synthetic_raw.copy()
    # Blow up Player A's last gameweek (GW6) points to an absurd value.
    mask = (mutated_raw['name'] == "Player A") & (mutated_raw['GW'] == 6)
    mutated_raw.loc[mask, ['total_points', 'bps', 'ict_index']] = 999
    mutated = engineer_features(preprocess_data(mutated_raw))

    feat_cols = ['rolling_3_avg_total_points', 'rolling_5_avg_total_points',
                 'lag_1_total_points', 'ewma_total_points', 'season_ppg']
    a_base = _player(base, "Player A")[feat_cols].iloc[:4].reset_index(drop=True)
    a_mut = _player(mutated, "Player A")[feat_cols].iloc[:4].reset_index(drop=True)
    pd.testing.assert_frame_equal(a_base, a_mut)


def test_position_labels_normalised(engineered):
    """'GKP' must be folded into 'GK'."""
    assert 'GKP' not in set(engineered['position'])
    assert (engineered[engineered['name'] == "Player B"]['position'] == 'GK').all()


def test_position_one_hot_is_consistent(engineered):
    """Exactly one position flag is set per row and it matches `position`."""
    flags = engineered[['is_gk', 'is_def', 'is_mid', 'is_fwd']]
    assert (flags.sum(axis=1) == 1).all()
    mid = engineered[engineered['position'] == 'MID']
    assert (mid['is_mid'] == 1).all()


def test_next_was_home_matches_next_fixture(engineered):
    """next_was_home[n] is the venue of match n+1 (known from the schedule)."""
    a = _player(engineered, "Player A")
    # was_home alternates True/False from GW1; next_was_home shifts that by one.
    assert list(a['next_was_home']) == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_no_target_rows_have_nan_target(engineered):
    """Rows with an unknown (final-match) target must be dropped."""
    assert engineered['target_next_gw_points'].notna().all()
    # 2 players x 6 GWs - 2 final matches = 10 rows.
    assert len(engineered) == 10
