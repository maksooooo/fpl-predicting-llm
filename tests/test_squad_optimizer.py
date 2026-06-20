"""Validate that the squad optimiser respects every FPL rule it claims to."""
import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA = os.path.join(ROOT, "data", "processed", "test_data_with_targets.csv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(TEST_DATA),
    reason="Test predictions not built yet — run ./run_pipeline.sh")


@pytest.fixture(scope="module")
def squad():
    from squad_optimizer import build_optimal_squad
    s, summary = build_optimal_squad(TEST_DATA, gw=10)
    return s, summary


def test_squad_has_15_players_in_required_positions(squad):
    s, _ = squad
    assert len(s) == 15
    counts = s['position'].value_counts().to_dict()
    assert counts == {'MID': 5, 'DEF': 5, 'FWD': 3, 'GK': 2}


def test_squad_within_budget(squad):
    s, _ = squad
    assert s['value'].sum() <= 1000   # £100.0m in tenths


def test_max_three_players_per_club(squad):
    s, _ = squad
    assert s['team_x'].value_counts().max() <= 3


def test_starting_eleven_and_legal_formation(squad):
    s, _ = squad
    xi = s[s['starter']]
    assert len(xi) == 11
    c = xi['position'].value_counts().to_dict()
    assert c.get('GK', 0) == 1
    assert 3 <= c.get('DEF', 0) <= 5
    assert 2 <= c.get('MID', 0) <= 5
    assert 1 <= c.get('FWD', 0) <= 3


def test_exactly_one_captain_who_starts(squad):
    s, _ = squad
    caps = s[s['captain']]
    assert len(caps) == 1
    assert bool(caps.iloc[0]['starter']) is True


def test_bench_are_not_starters(squad):
    s, _ = squad
    assert (s['starter'].sum() == 11) and ((~s['starter']).sum() == 4)
