"""Shared pytest fixtures.

Adds ``src/`` to the import path and builds a small, fully-controlled synthetic
raw dataset so the feature/leakage tests are fast and deterministic (no reliance
on the multi-hundred-MB real CSV).
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))


ROLL_COLS = ['total_points', 'minutes', 'bps', 'ict_index', 'threat',
             'creativity', 'influence', 'goals_scored', 'assists',
             'clean_sheets', 'goals_conceded', 'saves', 'bonus']


def _player_rows(name, team, opp, position, season, points_seq):
    """One row per gameweek for a single player with controllable points."""
    n = len(points_seq)
    base = pd.Timestamp("2023-08-12")
    rows = pd.DataFrame({
        'name': name, 'team_x': team, 'opp_team_name': opp,
        'opponent_team': 1, 'position': position, 'season_x': season,
        'element': 1, 'fixture': range(n), 'round': range(1, n + 1),
        'GW': range(1, n + 1),
        'kickoff_time': [base + pd.Timedelta(days=7 * i) for i in range(n)],
        'was_home': [i % 2 == 0 for i in range(n)],
        'value': [55 + i for i in range(n)],
        'total_points': points_seq,
        'minutes': [90 if p is not None else 0 for p in points_seq],
        'team_h_score': [2] * n, 'team_a_score': [1] * n,
        'selected': 1000, 'transfers_in': 0, 'transfers_out': 0,
        'transfers_balance': 0,
    })
    for c in ROLL_COLS:
        if c not in rows:
            rows[c] = [float(p) for p in points_seq]
    return rows


@pytest.fixture
def synthetic_raw():
    """Two players over several gameweeks, with known integer point sequences."""
    a = _player_rows("Player A", "Arsenal", "Chelsea", "MID", "2023-24",
                     [2, 5, 1, 8, 3, 6])
    b = _player_rows("Player B", "Chelsea", "Arsenal", "GKP", "2023-24",
                     [6, 0, 2, 4, 7, 1])
    df = pd.concat([a, b], ignore_index=True)
    # Shuffle to prove the pipeline sorts chronologically itself.
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)
