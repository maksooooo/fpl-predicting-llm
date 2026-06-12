import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import os

# Columns we roll into "recent form" features. All are shifted by one match
# first so the current GW's outcome never leaks into its own feature row.
ROLLING_COLS = [
    'total_points', 'minutes', 'bps', 'ict_index', 'threat', 'creativity',
    'influence', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
    'saves',
]
ROLLING_WINDOWS = (3, 5)


def load_data(filepath):
    """Load the raw FPL dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    """Parse dates, normalise labels, and sort chronologically per player."""
    print("Preprocessing data...")
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])

    # Normalise the position labels: the raw data mixes 'GK' and 'GKP'.
    df['position'] = df['position'].replace({'GKP': 'GK'})

    # Sort chronologically *within* each player so shift()/rolling() are safe.
    df = df.sort_values(by=['name', 'season_x', 'kickoff_time']).reset_index(drop=True)
    return df


def _shifted_group(df, col):
    """Return ``col`` shifted one match forward inside each player-season."""
    return df.groupby(['name', 'season_x'])[col].shift(1)


def engineer_features(df):
    """Create the target plus leakage-safe lag, rolling and context features."""
    print("Engineering features...")
    grp = df.groupby(['name', 'season_x'])

    # 1. Target: the player's points in the *next* gameweek.
    df['target_next_gw_points'] = grp['total_points'].shift(-1)

    # 2. Simple lag features (last match's output).
    df['lag_1_total_points'] = grp['total_points'].shift(1)
    df['lag_1_minutes'] = grp['minutes'].shift(1)

    # 3. Rolling "form" features over the last 3 and 5 matches. We shift by one
    #    first so the window covers *past* matches only (no target leakage).
    for col in ROLLING_COLS:
        shifted = _shifted_group(df, col)
        roll = shifted.groupby([df['name'], df['season_x']])
        for w in ROLLING_WINDOWS:
            df[f'rolling_{w}_avg_{col}'] = (
                roll.rolling(window=w, min_periods=1).mean()
                .reset_index(level=[0, 1], drop=True)
            )

    # 4. Consistency: std of recent points (low std = dependable returns).
    shifted_pts = _shifted_group(df, 'total_points')
    df['rolling_5_std_total_points'] = (
        shifted_pts.groupby([df['name'], df['season_x']])
        .rolling(window=5, min_periods=1).std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
    )

    # 5. Minutes reliability: share of recent matches with a "starter" workload
    #    (>=60 mins). A strong proxy for whether the player is nailed on.
    started = (_shifted_group(df, 'minutes') >= 60).astype(float)
    df['rolling_5_start_rate'] = (
        started.groupby([df['name'], df['season_x']])
        .rolling(window=5, min_periods=1).mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # 6. Scoring rate per 90 over recent form (rewards efficient minutes).
    df['rolling_5_points_per_90'] = (
        df['rolling_5_avg_total_points'] /
        df['rolling_5_avg_minutes'].clip(lower=1.0) * 90
    ).clip(upper=30)

    # 7. Fixture context for the match we are predicting. Home/away and the
    #    opponent are part of the published schedule, so the *next* fixture's
    #    venue is known ahead of time -- legitimate, not leakage.
    df['next_was_home'] = grp['was_home'].shift(-1).astype('float')
    df['was_home_int'] = df['was_home'].astype(int)

    # 8. One-hot position flags (position is a huge driver of FPL scoring and
    #    was previously dropped entirely).
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        df[f'is_{pos.lower()}'] = (df['position'] == pos).astype(int)

    # 9. Drop rows with no known target (last match of each player-season).
    print(f"Shape before dropping NaN targets: {df.shape}")
    df = df.dropna(subset=['target_next_gw_points']).reset_index(drop=True)
    print(f"Shape after dropping NaN targets: {df.shape}")

    return df


def save_data(df, output_path):
    """Save the engineered dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving features to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")


if __name__ == "__main__":
    input_file = "cleaned_merged_seasons.csv"
    output_file = "data/processed/engineered_features.csv"

    df = load_data(input_file)
    df = preprocess_data(df)
    df = engineer_features(df)
    save_data(df, output_file)
