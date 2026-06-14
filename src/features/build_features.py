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


def build_team_context(df):
    """Add fixture-difficulty features from team-level rolling form.

    For every player row we attach (a) their own team's recent attacking and
    defensive form and (b) the *next* opponent's attacking/defensive form
    entering the gameweek we are predicting. All rolling values are shifted so
    they only ever use matches played *before* the fixture in question -- the
    schedule (who/where) is known in advance, the form is computed from the
    past, so there is no leakage.
    """
    print("Building team/opponent strength features...")

    # Goals for/against from the player's team perspective.
    df['team_goals_for'] = np.where(df['was_home'], df['team_h_score'], df['team_a_score'])
    df['team_goals_against'] = np.where(df['was_home'], df['team_a_score'], df['team_h_score'])

    # Collapse the squad to one row per team-match, then roll each team's form.
    tm = (df.dropna(subset=['team_h_score'])
            .drop_duplicates(['season_x', 'team_x', 'GW'])
            [['season_x', 'team_x', 'GW', 'kickoff_time',
              'team_goals_for', 'team_goals_against']]
            .sort_values(['team_x', 'season_x', 'kickoff_time']))

    for src, dst in [('team_goals_for', 'team_att_form'),
                     ('team_goals_against', 'team_def_form')]:
        shifted = tm.groupby(['team_x', 'season_x'])[src].shift(1)
        tm[dst] = (shifted.groupby([tm['team_x'], tm['season_x']])
                   .rolling(window=5, min_periods=1).mean()
                   .reset_index(level=[0, 1], drop=True))

    form_cols = ['season_x', 'team_x', 'GW', 'team_att_form', 'team_def_form']

    # (a) Own-team form at the current gameweek.
    df = df.merge(tm[form_cols], on=['season_x', 'team_x', 'GW'], how='left')

    # (b) Next opponent's form entering the predicted fixture.
    grp = df.groupby(['name', 'season_x'])
    df['next_opp_name'] = grp['opp_team_name'].shift(-1)
    df['next_gw'] = grp['GW'].shift(-1)
    opp = tm[form_cols].rename(columns={
        'team_x': 'next_opp_name', 'GW': 'next_gw',
        'team_att_form': 'next_opp_att_form', 'team_def_form': 'next_opp_def_form'})
    df = df.merge(opp, on=['season_x', 'next_opp_name', 'next_gw'], how='left')

    df = df.drop(columns=['team_goals_for', 'team_goals_against',
                          'next_opp_name', 'next_gw'])
    return df


def engineer_features(df):
    """Create the target plus leakage-safe lag, rolling and context features."""
    print("Engineering features...")
    grp = df.groupby(['name', 'season_x'])

    # 1. Target: the player's points in the *next* gameweek. We also keep their
    #    next-GW minutes as a secondary target for the hurdle model's
    #    "will they even play?" stage. (Excluded from model inputs in
    #    train_model.DROP_COLS so it can never leak.)
    df['target_next_gw_points'] = grp['total_points'].shift(-1)
    df['target_next_gw_minutes'] = grp['minutes'].shift(-1)

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

    # 6b. Exponentially-weighted form -- weights recent matches more heavily
    #     than a flat rolling mean, capturing momentum.
    df['ewma_total_points'] = (
        shifted_pts.groupby([df['name'], df['season_x']])
        .apply(lambda s: s.ewm(span=3, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
    )

    # 6c. Season-to-date points-per-game (expanding mean of past matches).
    df['season_ppg'] = (
        shifted_pts.groupby([df['name'], df['season_x']])
        .expanding().mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
    )

    # 6d. Rest: days since the player's previous match (fatigue / freshness).
    prev_kickoff = grp['kickoff_time'].shift(1)
    df['days_since_last'] = (
        (df['kickoff_time'] - prev_kickoff).dt.days.clip(lower=0, upper=30)
    )

    # 7. Fixture context for the match we are predicting. Home/away and the
    #    opponent are part of the published schedule, so the *next* fixture's
    #    venue is known ahead of time -- legitimate, not leakage.
    df['next_was_home'] = grp['was_home'].shift(-1).astype('float')
    df['was_home_int'] = df['was_home'].astype(int)

    # 8. One-hot position flags (position is a huge driver of FPL scoring and
    #    was previously dropped entirely).
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        df[f'is_{pos.lower()}'] = (df['position'] == pos).astype(int)

    # 8b. Team & next-opponent strength (fixture difficulty).
    df = build_team_context(df)

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
